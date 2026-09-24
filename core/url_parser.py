"""URL recognition + cleaning. Provider-driven — each Provider exposes its own
URL_PATTERNS list, so adding a new music source needs zero changes here.

Two public entry points:
- `parse(text, registry)` — return the first ParsedURL found in `text`.
- `parse_all(text, registry)` — return every distinct match.

Plus `looks_like_url(text)`, which tells "a link no provider claims" apart
from "not a link", so handlers can answer the first and ignore the second.

Plus a tracking-param scrubber used to render clean canonical links back to
the user (strips si=, utm_*, fbclid=, igshid=, ref=, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .shortlink import resolve as _resolve_shortlink

if TYPE_CHECKING:
    from providers.registry import Registry

# Providers whose "url" kind means "needs an HTTP resolve before we know
# what this is" (spoti.fi/spotify.link, on.soundcloud.com/snd.sc). Other
# providers never produce kind="url" at all.
_SHORTLINK_PROVIDERS = frozenset({"spotify", "soundcloud"})


# Catches *anything* URL-shaped, including bare domains ("foo.com/bar") and
# full schemes. Both entry points that take user text (inline query, DM
# message) use it to tell "no provider claims this link" apart from "this
# isn't a link at all" — a link we can't handle gets an answer, plain text
# gets ignored.
_URL_LIKE_RE = re.compile(
    r"""(
        https?://\S+                        |  # explicit scheme
        (?:[a-z0-9-]+\.)+[a-z]{2,}/\S+       |  # bare host + path
        spotify:[a-z]+:[A-Za-z0-9]+             # spotify URI
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def looks_like_url(text: str) -> bool:
    """True when `text` is nothing but a URL."""
    return bool(_URL_LIKE_RE.fullmatch((text or "").strip()))


@dataclass(frozen=True)
class ParsedURL:
    provider: str
    kind: str  # "track" | "album" | "playlist" | "artist" | "url"
    entity_id: str
    url: str  # canonicalized URL (no tracking params)


# Query keys we strip from any user-shared URL before echoing it back.
TRACKING_PARAMS = frozenset({
    "si", "context", "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content", "fbclid", "gclid", "msclkid",
    "ref", "ref_src", "referrer", "igshid", "feature",
    "_branch_match_id", "_branch_referrer",
})


def strip_tracking(url: str) -> str:
    """Drop known tracking query params from `url`. Leaves all other params
    intact (e.g. SoundCloud's `in=` set ref is removed; YouTube's `t=` start
    timestamp is preserved)."""
    if not url:
        return url
    try:
        p = urlparse(url)
    except ValueError:
        return url
    if not p.query:
        return url
    kept = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS]
    return urlunparse(p._replace(query=urlencode(kept)))


def parse(text: str, registry: "Registry") -> Optional[ParsedURL]:
    """First match wins. Returns None if no provider claims any URL in `text`."""
    if not text:
        return None
    for provider in registry.all():
        m = provider.parse_url(text)
        if m is not None:
            return ParsedURL(
                provider=provider.name,
                kind=m.kind,
                entity_id=m.entity_id,
                url=provider.canonical_url(m.kind, m.entity_id),
            )
    return None


def parse_all(text: str, registry: "Registry") -> list[ParsedURL]:
    """Every distinct match across all providers, in registry order."""
    if not text:
        return []
    seen: set[tuple[str, str, str]] = set()
    out: list[ParsedURL] = []
    for provider in registry.all():
        for m in provider.parse_all(text):
            key = (provider.name, m.kind, m.entity_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(ParsedURL(
                provider=provider.name,
                kind=m.kind,
                entity_id=m.entity_id,
                url=provider.canonical_url(m.kind, m.entity_id),
            ))
    return out


async def resolve_url_kind(
    parsed: ParsedURL, registry: "Registry"
) -> Optional[ParsedURL]:
    """Follow a shortlink through to what it actually points at.

    Pass-through unchanged unless `parsed.kind == "url"` for a provider
    that uses that kind for shortlinks (spoti.fi/spotify.link,
    on.soundcloud.com/snd.sc) — the one thing every caller needs to do
    with a `url`-kind match before it can dispatch on `.kind`, so it lives
    here once instead of once per handler.

    Returns None if the shortlink couldn't be resolved to anything a
    provider recognizes; the caller decides how to surface that.
    """
    if parsed.kind != "url" or parsed.provider not in _SHORTLINK_PROVIDERS:
        return parsed
    resolved = await _resolve_shortlink(parsed.entity_id)
    if resolved is None:
        return None
    reparsed = parse(resolved, registry)
    if reparsed is not None and reparsed.kind != "url":
        return reparsed
    if parsed.provider == "soundcloud":
        # SoundCloud's own url-kind pattern is a plain permalink — this
        # is already its terminal form, the provider's own
        # `resolve_kind` figures out track/playlist/user from here.
        return parsed.__class__(
            provider=parsed.provider,
            kind=parsed.kind,
            entity_id=resolved,
            url=resolved,
        )
    return None
