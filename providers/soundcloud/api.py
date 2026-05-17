"""Thin async wrappers around api-v2.soundcloud.com.

The public `client_id` is auto-extracted from one of the JS bundles linked on
soundcloud.com if not supplied. Cached in-process, refreshed on 401.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlparse, urlunparse

import aiohttp

from core.exceptions import ProviderError, TrackNotFoundError
from core.http_retry import get_with_retry

API_V2 = "https://api-v2.soundcloud.com"
HOMEPAGE = "https://soundcloud.com/discover"

_CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*["\']([A-Za-z0-9]{20,40})["\']')
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+\.js[^"]*)"', re.I)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://soundcloud.com",
    "Referer": "https://soundcloud.com/",
}

log = logging.getLogger(__name__)


def _normalize_resolve_url(url: str) -> str:
    """Trim trailing slash(es) from the path before sending to /resolve.

    SC's resolve endpoint is strict: `/bladee1000/` 404s while
    `/bladee1000` returns the user object. Browser address bars often
    append a trailing slash, and our SC URL pattern intentionally
    captures any path char until `?`/`#`/whitespace, so the slash makes
    it into the resolve call. Strip it here so every caller (inline,
    dm, JobRunner) benefits.

    Preserves query string + fragment, only touches the path."""
    if not url:
        return url
    try:
        parts = urlparse(url)
    except ValueError:
        return url
    path = parts.path.rstrip("/")
    if not path:
        return url
    return urlunparse(parts._replace(path=path))


async def _probe_script_for_client_id(
    http: aiohttp.ClientSession, url: str
) -> Optional[str]:
    try:
        async with http.get(url, headers=DEFAULT_HEADERS) as r:
            if r.status != 200:
                return None
            body = await r.text()
    except aiohttp.ClientError:
        return None
    m = _CLIENT_ID_RE.search(body)
    return m.group(1) if m else None


async def fetch_client_id(http: aiohttp.ClientSession) -> str:
    """Scrape a working client_id from one of soundcloud.com's JS bundles."""
    async with http.get(HOMEPAGE, headers=DEFAULT_HEADERS) as r:
        if r.status != 200:
            raise ProviderError(f"soundcloud homepage {r.status}")
        html = await r.text()

    for src in _SCRIPT_SRC_RE.findall(html):
        cid = await _probe_script_for_client_id(http, src)
        if cid:
            return cid

    raise ProviderError("could not extract SoundCloud client_id from homepage")


class SoundCloudAPI:
    def __init__(self, client_id: Optional[str] = None):
        self._client_id = client_id
        self._http: Optional[aiohttp.ClientSession] = None
        self._cid_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession(
                headers=DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=30),
            )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    @property
    def http(self) -> aiohttp.ClientSession:
        assert self._http is not None, "SoundCloudAPI not started"
        return self._http

    async def _ensure_client_id(self, force: bool = False) -> str:
        if self._client_id and not force:
            return self._client_id
        async with self._cid_lock:
            if self._client_id and not force:
                return self._client_id
            self._client_id = await fetch_client_id(self.http)
            log.info("soundcloud client_id refreshed")
            return self._client_id

    async def _get_json(
        self, endpoint: str, params: Optional[dict] = None
    ) -> dict:
        """GET endpoint as JSON. Auto-retries with refreshed client_id on 401."""
        merged = dict(params or {})
        merged.setdefault("client_id", await self._ensure_client_id())
        try:
            r = await get_with_retry(
                self.http, endpoint, params=merged,
                # SoundCloud also rate-limits at high volume.
                max_attempts=4,
            )
        except aiohttp.ClientError as e:
            raise ProviderError(f"soundcloud network error: {e}") from e

        try:
            if r.status == 401:
                merged["client_id"] = await self._ensure_client_id(force=True)
                r2 = await get_with_retry(
                    self.http, endpoint, params=merged, max_attempts=2,
                )
                async with r2:
                    if r2.status != 200:
                        body = (await r2.text())[:200]
                        raise ProviderError(f"sc {endpoint} -> {r2.status}: {body}")
                    return await r2.json(content_type=None)
            if r.status == 404:
                raise TrackNotFoundError(f"soundcloud {endpoint} not found")
            if r.status != 200:
                body = (await r.text())[:200]
                raise ProviderError(f"sc {endpoint} -> {r.status}: {body}")
            return await r.json(content_type=None)
        finally:
            r.release()

    async def resolve(self, url: str) -> dict:
        """Resolve any soundcloud URL to its JSON entity (track / set / user).

        Strips a trailing slash from the URL path before forwarding —
        SC's resolve endpoint returns 404 for `/<slug>/` but accepts
        `/<slug>`. The trailing slash is a legitimate user-typed form
        (browser address bars often append one), so normalize here
        rather than failing the lookup."""
        return await self._get_json(
            f"{API_V2}/resolve", params={"url": _normalize_resolve_url(url)},
        )

    async def search_tracks(self, query: str, limit: int = 25) -> list[dict]:
        data = await self._get_json(
            f"{API_V2}/search/tracks",
            params={"q": query, "limit": min(limit, 50)},
        )
        return data.get("collection") or []

    async def transcoding_url(self, transcoding_url: str) -> str:
        """Resolve a transcoding URL to a streamable URL."""
        data = await self._get_json(transcoding_url)
        url = data.get("url")
        if not url:
            raise ProviderError(f"no streamable url in {data!r}")
        return url

    async def get_track_by_id(self, track_id: str) -> dict:
        return await self._get_json(f"{API_V2}/tracks/{track_id}")

    async def get_user_tracks(
        self, user_id: str | int, *, limit: int = 50,
    ) -> list[dict]:
        """Return a SoundCloud user's *own* uploads (no reposts).

        `/users/{id}/tracks` is the right endpoint here — `/reposts` is a
        separate one and explicitly excluded. SC caps a single page
        around 50; we don't paginate further to keep inline latency in
        check (50 is also the inline-result cap)."""
        try:
            data = await self._get_json(
                f"{API_V2}/users/{user_id}/tracks",
                params={"limit": str(min(limit, 50))},
            )
        except TrackNotFoundError:
            return []
        return list(data.get("collection") or [])

    async def get_tracks_by_ids(self, track_ids: list[str | int]) -> list[dict]:
        """Batch-resolve a list of track IDs in a single round-trip.

        SoundCloud's `/playlists` resolve only embeds full metadata for the
        first ~5 tracks; the rest come back as `{id, monetization_model,
        policy}` stubs. We need this batch endpoint to hydrate those into
        usable Track objects. SC caps a single `?ids=` call somewhere
        around 50 — we batch in chunks of 40 to stay well clear."""
        if not track_ids:
            return []
        out: list[dict] = []
        chunk = 40
        for i in range(0, len(track_ids), chunk):
            ids_csv = ",".join(str(t) for t in track_ids[i:i + chunk])
            data = await self._get_json(
                f"{API_V2}/tracks", params={"ids": ids_csv},
            )
            if isinstance(data, list):
                out.extend(data)
        return out
