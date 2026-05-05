"""URL recognition + cleaning. Provider-driven — each Provider exposes its own
URL_PATTERNS list, so adding a new music source needs zero changes here.

Two public entry points:
- `parse(text, registry)` — return the first ParsedURL found in `text`.
- `parse_all(text, registry)` — return every distinct match.

Plus a tracking-param scrubber used to render clean canonical links back to
the user (strips si=, utm_*, fbclid=, igshid=, ref=, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from providers.registry import Registry


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
