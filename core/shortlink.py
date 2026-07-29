"""Redirect resolver for known music shortlink domains."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import aiohttp

from .logging_setup import logger

_SHORTLINK_DOMAINS = frozenset({
    "spoti.fi",
    "spotify.link",
    "on.soundcloud.com",
    "snd.sc",
})

# spotify.link serves HTTP 200 with a client-side JS redirect (no Location
# header), so `allow_redirects` alone never reaches the canonical URL. The
# target is embedded as an `og:url` meta tag in the page it does return.
# property/content can appear in either attribute order.
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


def _find_og_url(body: str) -> str | None:
    for pat in _OG_URL_PATTERNS:
        m = pat.search(body)
        if m:
            return html.unescape(m.group(1))
    return None


async def resolve(url: str) -> str:
    """Follow redirect for known shortener domains; return canonical URL.
    Returns `url` unchanged on network error or if not a known shortener."""
    host = urlparse(url).netloc.lower()
    if host not in _SHORTLINK_DOMAINS:
        return url
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                final = str(r.url)
                if urlparse(final).netloc.lower() not in _SHORTLINK_DOMAINS:
                    return final
                body = await r.text()
        return _find_og_url(body) or final
    except Exception as e:
        logger.error("shortlink resolve failed for {} ({}): {}", url, type(e).__name__, e)
        return url
