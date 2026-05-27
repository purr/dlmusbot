"""Thin async wrappers around api-v2.soundcloud.com.

The public `client_id` is auto-extracted from one of the JS bundles linked on
soundcloud.com if not supplied. Cached in-process, refreshed on 401.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp

from core import disk_cache
from core.exceptions import ProviderError, TrackNotFoundError
from core.http_retry import get_with_retry

API_V2 = "https://api-v2.soundcloud.com"
HOMEPAGE = "https://soundcloud.com/discover"

# Persist scraped client_id between runs. SoundCloud rotates the public
# id rarely (weeks-to-months), so caching for 7 days + lazy-refresh on
# 401 keeps the bot fast and cuts churn against the homepage (which is
# AWS-WAF-fronted and intermittently serves 202 to bot-shaped UAs).
# Path is anchored relative to the repo root (this file's grandparent's
# parent), NOT cwd — so a different working directory (cron, systemd
# WorkingDirectory unset, pm2 with non-default cwd) can't silently
# scatter cache files across the filesystem.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CID_CACHE_PATH = _REPO_ROOT / "data" / "soundcloud_client_id.json"
_CID_CACHE_TTL_S = 7 * 24 * 3600

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


_CID_VALID_RE = re.compile(r"[A-Za-z0-9]{20,40}")


def _is_valid_client_id(cid: Any) -> bool:
    return isinstance(cid, str) and bool(_CID_VALID_RE.fullmatch(cid))


def _load_cached_client_id() -> tuple[Optional[str], bool]:
    """Returns (client_id_or_None, is_fresh) via the shared disk_cache
    helper. Validates against the SoundCloud public-id charset so a
    corrupted file can't ship garbage strings into every API request."""
    return disk_cache.load(
        _CID_CACHE_PATH,
        value_key="client_id",
        ttl_seconds=_CID_CACHE_TTL_S,
        validator=_is_valid_client_id,
    )


def _save_cached_client_id(cid: str) -> None:
    disk_cache.save(_CID_CACHE_PATH, value_key="client_id", value=cid)


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


# Browser impersonation profile for the Cloudflare-fronted endpoints.
# `aiohttp`'s TLS handshake doesn't match real Chrome — Cloudflare reads
# the JA3/JA4 fingerprint and serves 202 (or 403) to anything that
# doesn't look like a real browser. `curl_cffi` wraps libcurl-impersonate
# and reproduces Chrome's exact TLS ClientHello + HTTP/2 settings, so
# our requests are indistinguishable from a browser at the network
# level. Used ONLY for the homepage scrape (the Cloudflare wall);
# api-v2 endpoints don't need it and stay on the cheap aiohttp path.
_BROWSER_IMPERSONATE = "chrome131"


async def _probe_script_for_client_id_browserlike(
    session, url: str,
) -> Optional[str]:
    """Browser-impersonated GET on a JS bundle URL using the *shared*
    AsyncSession opened by `fetch_client_id`. Returns the client_id
    literal if found, else None. Sharing the session avoids paying a
    fresh TLS handshake per script URL — a 9-script homepage previously
    paid 9× the impersonation cost plus risked hitting AWS WAF's
    burst-rate trigger."""
    try:
        r = await session.get(url, timeout=10)
        if r.status_code != 200:
            return None
        body = r.text
    except Exception as e:
        log.debug("soundcloud script probe failed for %s: %s", url, e)
        return None
    m = _CLIENT_ID_RE.search(body)
    return m.group(1) if m else None


# AWS WAF backoff schedule. SoundCloud sits behind AWS WAF which
# rate-triggers a 202 "Just a moment..." JS challenge under bursty
# load. The challenge typically clears within 10-30s; these waits
# give it room to relax without making the operator stare at logs.
_WAF_RETRY_WAITS_S: tuple[float, ...] = (8.0, 15.0, 25.0)


async def fetch_client_id(http: aiohttp.ClientSession) -> str:
    """Scrape a working client_id from one of soundcloud.com's JS bundles.

    Uses `curl_cffi` to impersonate a real Chrome TLS handshake — the
    aiohttp/Python TLS fingerprint trips AWS WAF's bot wall and gets
    served a 202 JS challenge instead of the real homepage. The
    `http` parameter is kept in the signature for backward-
    compatibility but is intentionally not used here.

    If the WAF *is* in challenge mode (rate-triggered, transient),
    we retry up to len(_WAF_RETRY_WAITS_S) times with growing waits.
    """
    del http  # browser-impersonated session is opened internally
    try:
        from curl_cffi.requests import AsyncSession
    except ImportError as e:
        raise ProviderError(
            f"curl_cffi not installed — needed for soundcloud "
            f"homepage scrape (AWS WAF bot wall): {e}"
        ) from e

    # Single AsyncSession reused for the homepage fetch + every script
    # probe. New session per probe was paying a TLS handshake per call
    # and risked AWS WAF's burst-rate trigger.
    async with AsyncSession(impersonate=_BROWSER_IMPERSONATE) as s:
        html: Optional[str] = None
        for attempt, wait_s in enumerate(_WAF_RETRY_WAITS_S, start=1):
            try:
                r = await s.get(HOMEPAGE, timeout=10)
            except Exception as e:
                raise ProviderError(
                    f"soundcloud homepage fetch failed: {e}"
                ) from e
            if r.status_code == 200:
                html = r.text
                break
            if r.status_code == 202:
                log.warning(
                    "soundcloud WAF challenge (attempt %d/%d, waiting "
                    "%.0fs before retry)",
                    attempt, len(_WAF_RETRY_WAITS_S), wait_s,
                )
                if attempt < len(_WAF_RETRY_WAITS_S):
                    await asyncio.sleep(wait_s)
                continue
            raise ProviderError(
                f"soundcloud homepage {r.status_code} "
                f"(unexpected status with {_BROWSER_IMPERSONATE} "
                f"impersonation)"
            )
        if html is None:
            raise ProviderError(
                f"soundcloud WAF stayed in challenge mode across "
                f"{len(_WAF_RETRY_WAITS_S)} attempts"
            )

        for src in _SCRIPT_SRC_RE.findall(html):
            cid = await _probe_script_for_client_id_browserlike(s, src)
            if cid:
                return cid

    raise ProviderError("could not extract SoundCloud client_id from homepage")


class SoundCloudAPI:
    def __init__(self, client_id: Optional[str] = None):
        # Prefer caller-supplied id; otherwise load from disk so a
        # fresh process boot doesn't need to scrape the homepage at
        # all. Stale-cache fallback is handled by the same load fn.
        if not client_id:
            cached, fresh = _load_cached_client_id()
            if cached:
                client_id = cached
                log.info(
                    "soundcloud client_id loaded from disk cache "
                    "(%s)",
                    "fresh" if fresh else "stale — will refresh on 401",
                )
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

    async def ensure_client_id(self, force: bool = False) -> str:
        """Return the cached public client_id, scraping one on first call.
        Pass `force=True` after a 401 to invalidate + refetch."""
        return await self._ensure_client_id(force=force)

    async def _ensure_client_id(self, force: bool = False) -> str:
        if self._client_id and not force:
            return self._client_id
        async with self._cid_lock:
            if self._client_id and not force:
                return self._client_id
            # Remember the existing (possibly soon-stale) id so we can
            # fall back to it if the scrape fails. Cloudflare often
            # responds 202 to bot-shaped User-Agents; a 7-day-old
            # client_id almost always still works.
            previous = self._client_id
            try:
                self._client_id = await fetch_client_id(self.http)
                _save_cached_client_id(self._client_id)
                log.info("soundcloud client_id refreshed + cached to disk")
            except ProviderError as e:
                if previous:
                    log.warning(
                        "soundcloud client_id refresh failed (%s); "
                        "continuing with previous id",
                        e,
                    )
                    self._client_id = previous
                else:
                    raise
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
