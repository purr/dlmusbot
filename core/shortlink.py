"""HTTP redirect resolver for known music shortlink domains."""
from __future__ import annotations

from urllib.parse import urlparse

import aiohttp

_SHORTLINK_DOMAINS = frozenset({
    "spoti.fi",
    "spotify.link",
    "on.soundcloud.com",
    "snd.sc",
})


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
                return str(r.url)
    except Exception:
        return url
