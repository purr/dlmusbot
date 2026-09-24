"""Redirect resolver for known music shortlink domains."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import aiohttp

from .logging_setup import logger

# Hosts that serve nothing but redirect stubs — every path on them is a
# shortlink.
_SHORTLINK_DOMAINS = frozenset({
    "spoti.fi",
    "spotify.link",
    "on.soundcloud.com",
    "snd.sc",
})

# Hosts that serve real content *and* shortlinks, keyed by the path prefix
# that marks the shortlink. Spotify's share sheet now hands out
# `open.spotify.com/s/<code>`, which sits on the same host as canonical
# track/album URLs — so the host alone can't decide, the path has to.
_SHORTLINK_PATHS: dict[str, tuple[str, ...]] = {
    "open.spotify.com": ("/s/",),
}

# spotify.link and open.spotify.com/s/ serve HTTP 200 with a client-side JS
# redirect (no Location header), so `allow_redirects` alone never reaches the
# canonical URL. The target is embedded as an `og:url` meta tag in the page
# they do return. property/content can appear in either attribute order.
_OG_URL_PATTERNS = (
    re.compile(
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']',
        re.IGNORECASE,
    ),
)


def is_shortlink(url: str) -> bool:
    """True when `url` needs an HTTP round-trip before we know what it is."""
    p = urlparse(url)
    host = p.netloc.lower()
    if host in _SHORTLINK_DOMAINS:
        return True
    prefixes = _SHORTLINK_PATHS.get(host)
    return bool(prefixes and p.path.startswith(prefixes))


def _find_og_url(body: str) -> str | None:
    for pat in _OG_URL_PATTERNS:
        m = pat.search(body)
        if m:
            return html.unescape(m.group(1))
    return None


async def resolve(url: str) -> str | None:
    """Follow a shortlink to the canonical URL behind it.

    Returns `url` unchanged when it isn't a shortlink at all, and None when
    it is one we couldn't resolve (network error, or a page with no target
    in it). None is deliberately distinct from "returned the input": the
    caller surfaces an unresolved shortlink to the user instead of retrying
    the stub forever.
    """
    if not is_shortlink(url):
        return url
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                final = str(r.url)
                if not is_shortlink(final):
                    return final
                body = await r.text()
        target = _find_og_url(body)
        if target is None:
            logger.error("shortlink {} resolved to no canonical target", url)
        return target
    except Exception as e:
        logger.error("shortlink resolve failed for {} ({}): {}", url, type(e).__name__, e)
        return None
