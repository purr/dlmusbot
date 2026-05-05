"""SpotifyProvider — wraps the in-tree librespot downloader."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional, TypeVar

import aiohttp
from loguru import logger as _log

from core.exceptions import ProviderError
from core.models import Album, ArtistRef, DownloadResult, Playlist, Track
from ..base import StageCallback

from . import search as search_mod
from ._internal import (
    SpotifyDownloaderError,
    download_track_with_session,
)
from ._internal import api as api_mod
from ._internal.auth import DEFAULT_HEADERS, HTTP_TIMEOUT, get_access_token
from ._internal.exceptions import (
    HandshakeError,
    LoginError,
    MercuryError,
    TokenExpiredError,
    TrackUnavailableError,
)
from ._internal.ids import base62_to_gid, parse_track_id
from ._internal.librespot import Session
from ._internal.models import Album as _SpAlbum
from ._internal.models import Track as _SpTrack
from ..base import Provider

T = TypeVar("T")

# Errors that mean "the cached AP socket is dead — reconnect and retry".
_RECONNECTABLE = (
    HandshakeError, MercuryError, OSError, asyncio.IncompleteReadError,
    ConnectionError,
)

# How many times to retry a session-level call before giving up. AP
# sockets get rotated server-side without warning; one retry isn't
# always enough.
_SESSION_RETRY_ATTEMPTS = 3
_SESSION_RETRY_DELAY_S = 0.4  # multiplied by attempt number for gentle backoff

log = logging.getLogger(__name__)


# Token cache TTL — fallback when the auth response doesn't include a
# usable expiry. Real Spotify tokens last ~1h; we refresh after 50 min so
# in-flight requests don't race a server-side rotation.
TOKEN_TTL_SECONDS = 50 * 60

# Safety window — refresh the token this many seconds *before* its declared
# expiry so we don't ship a token that dies mid-request on the wire.
TOKEN_REFRESH_LEEWAY_S = 60

# Audio formats we'll ask librespot for, ordered highest-quality-first.
# Only OGG_VORBIS variants are listed because they're the only librespot-
# decryptable formats for free / standard accounts (FLAC + MP4 are
# PlayPlay/Widevine-locked — see spotify_test/NOTES.md). Keeping this at
# the top of the file so the bitrate ladder is visible without diving
# into the internal api module.
SPOTIFY_FORMAT_PRIORITY: list[str] = [
    "OGG_VORBIS_320",
    "OGG_VORBIS_160",
    "OGG_VORBIS_96",
]

def _decrypt_concurrency_from_cores() -> int:
    """Pick a safe decrypt concurrency based on host CPU count.

    Decrypt is pure-Python CPU work; too many parallel decrypts can stall small
    VPS hosts. Rule of thumb:
    - <=2 cores -> 1 (explicit requirement)
    - otherwise -> about half the cores, capped to 4.
    """
    cores = os.cpu_count() or 1
    if cores <= 2:
        return 1
    return max(1, min(4, cores // 2))

# Max tracks we'll hydrate per Spotify playlist. Each Mercury hop is ~50ms
# and serialised by Session._lock, so 50 tracks ≈ 2.5s — fits inside the
# 5 s inline-search deadline with headroom for the playlist GET itself.
PLAYLIST_TRACK_LIMIT = 50


def _album_from_internal(
    sp_album: "_SpAlbum",
    sp_tracks: list["_SpTrack"],
    *,
    total_tracks: Optional[int] = None,
) -> Album:
    """Build a core.Album (with embedded core.Tracks) from the librespot
    Album+Track pair returned by `parse_album_with_tracks`. `total_tracks`,
    if given, is the *uncapped* GID count from the album proto so the UI
    can show the real release length even when we only hydrated a subset."""
    artist_refs = [
        ArtistRef(
            name=a.name,
            artist_id=a.spotify_id,
            url=f"https://open.spotify.com/artist/{a.spotify_id}",
        )
        for a in sp_album.artists
    ]
    cover = sp_album.cover_url
    tracks: list[Track] = []
    for st in sp_tracks:
        t_artists = [
            ArtistRef(
                name=a.name,
                artist_id=a.spotify_id,
                url=f"https://open.spotify.com/artist/{a.spotify_id}",
            )
            for a in (list(st.artists) + list(st.featured_artists))
        ]
        tracks.append(Track(
            provider="spotify",
            track_id=st.spotify_id,
            title=st.name,
            artists=t_artists,
            album=sp_album.name,
            duration_seconds=st.duration_ms // 1000,
            artwork_url=cover,
            url=st.spotify_url,
            extra={"album_id": sp_album.spotify_id},
        ))
    return Album(
        provider="spotify",
        album_id=sp_album.spotify_id,
        title=sp_album.name,
        artists=artist_refs,
        artwork_url=cover,
        url=sp_album.spotify_url,
        tracks=tracks,
        total_tracks=total_tracks if total_tracks is not None else len(tracks),
    )


def _track_from_internal(st: _SpTrack) -> Track:
    """Convert a Mercury-fetched internal Track into the bot's core.Track."""
    artists = [
        ArtistRef(
            name=a.name,
            artist_id=a.spotify_id,
            url=f"https://open.spotify.com/artist/{a.spotify_id}",
        )
        for a in (list(st.artists) + list(st.featured_artists))
    ]
    album_id = st.album.spotify_id if st.album else None
    return Track(
        provider="spotify",
        track_id=st.spotify_id,
        title=st.name or "<unknown>",
        artists=artists,
        album=st.album.name if st.album else None,
        duration_seconds=st.duration_ms // 1000,
        artwork_url=(st.album.cover_url if st.album else None),
        url=st.spotify_url,
        isrc=st.isrc,
        extra={"album_id": album_id} if album_id else {},
    )


class SpotifyProvider(Provider):
    name = "spotify"
    label = "Spotify"

    URL_PATTERNS = [
        ("track", re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?track/([A-Za-z0-9]{22})")),
        ("album", re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?album/([A-Za-z0-9]{22})")),
        ("playlist", re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?playlist/([A-Za-z0-9]{22})")),
        ("artist", re.compile(r"open\.spotify\.com/(?:intl-[a-z]{2}/)?artist/([A-Za-z0-9]{22})")),
        ("track", re.compile(r"spotify:track:([A-Za-z0-9]{22})")),
        ("album", re.compile(r"spotify:album:([A-Za-z0-9]{22})")),
        ("playlist", re.compile(r"spotify:playlist:([A-Za-z0-9]{22})")),
        ("artist", re.compile(r"spotify:artist:([A-Za-z0-9]{22})")),
        # Shortened / campaign URLs — need an HTTP redirect to resolve.
        ("url", re.compile(r"(https?://(?:spoti\.fi|spotify\.link)/[^\s?#]+)")),
    ]

    def __init__(self, sp_dc: str):
        if not sp_dc:
            raise ValueError("SpotifyProvider requires sp_dc")
        self._sp_dc = sp_dc
        self._http: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._token_fetched_at: float = 0.0
        self._token_expires_at: float = 0.0
        # Serialise concurrent token refreshes — without this, N parallel
        # inline searches that all hit a stale token would each fire their
        # own /api/token request (and TOTP secret fetch), hammering Spotify
        # for nothing.
        self._token_lock = asyncio.Lock()
        # Long-lived librespot AP socket. Cached so we don't pay the DH+login
        # handshake (~hundreds of ms) on every track. Recreated lazily after
        # a connection drop or auth-token rotation.
        self._session: Optional[Session] = None
        self._session_lock = asyncio.Lock()
        self._decrypt_max_concurrency = _decrypt_concurrency_from_cores()
        self._decrypt_sem = asyncio.Semaphore(self._decrypt_max_concurrency)
        _log.info(
            "[spotify] decrypt concurrency={} (cores={})",
            self._decrypt_max_concurrency,
            os.cpu_count() or 1,
        )

    async def start(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession(
                headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT,
            )

    async def close(self) -> None:
        await self._drop_session()
        if self._http is not None:
            await self._http.close()
            self._http = None

    async def _access_token(self, *, force_refresh: bool = False) -> str:
        """Return a valid bearer token.

        Honours `accessTokenExpirationTimestampMs` from Spotify's /api/token
        response when present (so we refresh as the server rotates rather
        than guessing) and falls back to TOKEN_TTL_SECONDS otherwise. Pass
        `force_refresh=True` after a 401 to invalidate the cache and mint
        a fresh token even if the cached one *looks* valid by clock."""
        now = time.time()
        if (
            not force_refresh
            and self._token
            and now < self._token_expires_at
        ):
            return self._token
        async with self._token_lock:
            now = time.time()
            # Re-check inside the lock so the loser of a refresh race
            # picks up the freshly minted token instead of doing its own
            # round-trip.
            if (
                not force_refresh
                and self._token
                and now < self._token_expires_at
            ):
                return self._token
            assert self._http is not None
            token_info = await get_access_token(self._sp_dc, session=self._http)
            self._token = token_info["accessToken"]
            self._token_fetched_at = now
            expiry_ms = token_info.get("accessTokenExpirationTimestampMs")
            if isinstance(expiry_ms, (int, float)) and expiry_ms > 0:
                self._token_expires_at = (
                    expiry_ms / 1000.0 - TOKEN_REFRESH_LEEWAY_S
                )
            else:
                self._token_expires_at = now + TOKEN_TTL_SECONDS
            _log.info(
                "[spotify] minted access token (expires in {:.0f}s)",
                max(0.0, self._token_expires_at - now),
            )
            return self._token

    async def _with_token_retry(
        self, fn: Callable[[str], Awaitable[T]],
    ) -> T:
        """Run a token-using HTTP call, refreshing once on a 401. Spotify
        rotates bearer tokens server-side without warning, and our local
        cache only refreshes when the clock says it should — so a token
        that *looks* fresh can still 401 mid-flight. Catching the
        TokenExpiredError sentinel and retrying with a freshly minted
        token absorbs that case transparently."""
        token = await self._access_token()
        try:
            return await fn(token)
        except TokenExpiredError:
            log.info("spotify access token rejected (401), refreshing once")
            token = await self._access_token(force_refresh=True)
            return await fn(token)

    # ---- session caching -------------------------------------------------

    async def _drop_session(self) -> None:
        async with self._session_lock:
            sess, self._session = self._session, None
        if sess is not None:
            try:
                await sess.close()
            except Exception:
                log.debug("error closing stale Spotify session", exc_info=True)

    async def _get_session(self) -> Session:
        """Return a connected Session, opening one on first use or after a
        prior call left the cache empty. Always pushes a fresh access token
        onto the session so HTTP-side calls (track-playback, storage-resolve)
        don't 401 after the original handshake token expires."""
        token = await self._access_token()
        async with self._session_lock:
            assert self._http is not None
            if self._session is not None and self._session.is_connected:
                self._session.access_token = token
                return self._session
            sess = Session(token, http=self._http)
            try:
                await sess.connect()
            except (HandshakeError, LoginError, OSError):
                try:
                    await sess.close()
                except Exception:
                    pass
                raise
            self._session = sess
            return sess

    async def _with_session(
        self, fn: Callable[[Session], Awaitable[T]],
    ) -> T:
        """Run an operation against the cached session. AP sockets die for
        all sorts of reasons (idle drops, server-side rotations, transient
        TLS resets) so we retry up to `_SESSION_RETRY_ATTEMPTS` times,
        dropping + reconnecting between each. Brief backoff between
        attempts so we don't hammer the AP server when it's mid-restart."""
        last_err: Optional[BaseException] = None
        for attempt in range(1, _SESSION_RETRY_ATTEMPTS + 1):
            try:
                sess = await self._get_session()
                return await fn(sess)
            except _RECONNECTABLE as e:
                last_err = e
                if attempt >= _SESSION_RETRY_ATTEMPTS:
                    break
                log.warning(
                    "spotify session call failed (%s) attempt %d/%d; "
                    "reconnecting", e, attempt, _SESSION_RETRY_ATTEMPTS,
                )
                await self._drop_session()
                await asyncio.sleep(_SESSION_RETRY_DELAY_S * attempt)
        assert last_err is not None
        raise last_err

    # ---- url plumbing ----------------------------------------------------

    def canonical_url(self, kind: str, entity_id: str) -> str:
        if kind == "url":
            return entity_id  # entity_id is already the full short URL
        return f"https://open.spotify.com/{kind}/{entity_id}"

    def artist_url(self, artist_id: str) -> Optional[str]:
        if not artist_id:
            return None
        return f"https://open.spotify.com/artist/{artist_id}"

    # ---- search / fetch --------------------------------------------------

    async def search(self, query: str, limit: int = 25) -> list[Track]:
        assert self._http is not None
        try:
            return await self._with_token_retry(
                lambda tok: search_mod.search_tracks(
                    self._http, tok, query, limit,
                )
            )
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify search failed: {e}") from e

    async def get_track(self, entity_id: str) -> Track:
        # Use Mercury (AP server) instead of Web API /v1/tracks — the Web API
        # is rate-limited per app/IP and 429s under load, while the Mercury
        # path is the same one used for the actual download.
        track_id = parse_track_id(entity_id)
        gid_hex = base62_to_gid(track_id).hex()
        try:
            st = await self._with_session(lambda s: api_mod.fetch_track(s, gid_hex))
        except TrackUnavailableError as e:
            # Region-locked / removed / playback-redirect-mismatch — surface
            # as an `unavailable` permanent failure so the bot shows the
            # right popup instead of the generic retry kb.
            raise ProviderError(
                f"spotify track {entity_id} unavailable: {e}", reason="unavailable",
            ) from e
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify track fetch failed: {e}") from e
        return _track_from_internal(st)

    async def get_album(self, entity_id: str) -> Optional[Album]:
        # Mercury album endpoint — same /v1/albums data without the
        # TOTP-token rate cap that breaks the Web API. Mercury's album
        # proto only carries track GIDs (no per-track metadata), so we
        # then hydrate each via the track endpoint (~50 ms / track).
        gid_hex = base62_to_gid(entity_id).hex()

        async def _fetch(s: Session):
            sp_album, gid_list = await api_mod.fetch_album_track_gids(s, gid_hex)
            if sp_album is None:
                return None, [], 0
            sp_tracks: list[_SpTrack] = []
            for g in gid_list[:PLAYLIST_TRACK_LIMIT]:
                try:
                    st = await api_mod.fetch_track_basic(s, g)
                except SpotifyDownloaderError:
                    continue
                if st is not None:
                    sp_tracks.append(st)
            return sp_album, sp_tracks, len(gid_list)

        try:
            sp_album, sp_tracks, full_count = await self._with_session(_fetch)
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify album fetch failed: {e}") from e
        if sp_album is None:
            return None
        return _album_from_internal(sp_album, sp_tracks, total_tracks=full_count)

    async def get_artist(self, entity_id: str) -> Optional[Playlist]:
        # Mercury artist endpoint returns top-track gids per country (only
        # the first populated bucket is used — usually the global / US set
        # of ~10 most-streamed tracks). Each gid is then hydrated via
        # fetch_track_basic in the same Session.
        gid_hex = base62_to_gid(entity_id).hex()

        async def _fetch(s: Session):
            name, gid_list = await api_mod.fetch_artist_top_tracks(s, gid_hex)
            if not gid_list:
                return name, []
            sp_tracks: list[_SpTrack] = []
            for g in gid_list[:PLAYLIST_TRACK_LIMIT]:
                try:
                    st = await api_mod.fetch_track_basic(s, g)
                except SpotifyDownloaderError:
                    continue
                if st is not None:
                    sp_tracks.append(st)
            return name, sp_tracks

        try:
            name, sp_tracks = await self._with_session(_fetch)
        except TrackUnavailableError as e:
            raise ProviderError(
                f"spotify artist {entity_id} unavailable: {e}",
                reason="unavailable",
            ) from e
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify artist fetch failed: {e}") from e

        if not name:
            return None  # no artist record at all
        return Playlist(
            provider="spotify",
            playlist_id=entity_id,
            title=name,
            owner=name,
            url=f"https://open.spotify.com/artist/{entity_id}",
            tracks=[_track_from_internal(st) for st in sp_tracks],
            total_tracks=len(sp_tracks),
        )

    async def get_playlist(self, entity_id: str) -> Optional[Playlist]:
        # Two-step: spclient/playlist/v2 for the URI list, then per-track
        # Mercury fetches for metadata. Capped at PLAYLIST_TRACK_LIMIT so
        # absurdly long Spotify mixes don't blow past the inline-query
        # deadline (each Mercury hop is ~50ms, serialised by Session._lock).
        assert self._http is not None
        url = f"{search_mod.PLAYLIST_V2_BASE}/{entity_id}"

        async def _load(token: str) -> Optional[dict]:
            headers = {
                "Authorization": f"Bearer {token}",
                **search_mod.SPCLIENT_HEADERS,
            }
            try:
                async with self._http.get(url, headers=headers) as r:
                    if r.status == 401:
                        # Bubble up so _with_token_retry refreshes the
                        # token; without this a stale spclient token
                        # would surface to the user as "playlist not
                        # found" or generic 401.
                        raise TokenExpiredError(
                            f"spotify playlist {entity_id} 401 (token expired)"
                        )
                    if r.status == 404:
                        return None
                    if r.status != 200:
                        body = (await r.text())[:200]
                        raise ProviderError(
                            f"spotify playlist {entity_id} -> {r.status}: {body}"
                        )
                    return await r.json(content_type=None)
            except aiohttp.ClientError as e:
                raise ProviderError(
                    f"spotify playlist network error: {e}"
                ) from e

        data = await self._with_token_retry(_load)
        if data is None:
            return None

        title = (data.get("attributes") or {}).get("name") or "<unknown>"
        owner = data.get("ownerUsername")
        # spclient returns the playlist's true length up front, even if
        # we only hydrate a prefix below — preserve it for the UI header.
        full_length: Optional[int] = data.get("length")
        items = (data.get("contents") or {}).get("items") or []
        track_ids: list[str] = []
        for it in items:
            uri = (it or {}).get("uri") or ""
            if uri.startswith("spotify:track:"):
                tid = uri.rsplit(":", 1)[-1]
                if tid:
                    track_ids.append(tid)
            if len(track_ids) >= PLAYLIST_TRACK_LIMIT:
                break
        if full_length is None:
            # spclient sometimes omits length on small playlists; the URI
            # list itself is then the source of truth (uncapped count).
            uri_count = sum(
                1 for it in items
                if (it or {}).get("uri", "").startswith("spotify:track:")
            )
            full_length = uri_count

        async def _hydrate(s: Session) -> list[Track]:
            out: list[Track] = []
            for tid in track_ids:
                gid = base62_to_gid(tid).hex()
                try:
                    st = await api_mod.fetch_track_basic(s, gid)
                except SpotifyDownloaderError:
                    continue
                if st is None:
                    continue
                out.append(_track_from_internal(st))
            return out

        try:
            tracks = await self._with_session(_hydrate)
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify playlist hydrate failed: {e}") from e

        return Playlist(
            provider="spotify",
            playlist_id=entity_id,
            title=title,
            owner=owner,
            url=f"https://open.spotify.com/playlist/{entity_id}",
            tracks=tracks,
            total_tracks=full_length,
        )

    # ---- download --------------------------------------------------------

    def _make_progress_logger(
        self, track_id: str,
        on_stage: Optional[StageCallback] = None,
    ) -> Callable[[str, dict], Awaitable[None]]:
        """Build a progress callback that:
        1. emits a structured loguru line per download stage,
        2. forwards stage transitions to the bot's UI via `on_stage`.

        Returning an async callback (rather than the previous sync one)
        because we need to `await on_stage(...)` inside it. The internal
        downloader awaits the callback iff it's a coroutine."""
        tag = f"spotify:{track_id}"
        last_bucket: dict[str, int] = {"v": 0}  # buckets of 25% (1=25%, 2=50%, 3=75%)

        async def _set_stage(stage: str) -> None:
            if on_stage is None:
                return
            try:
                await on_stage(stage)
            except Exception:
                log.debug("on_stage(%s) failed", stage, exc_info=True)

        async def cb(event: str, info: dict) -> None:
            if event == "track_id":
                _log.debug("[{}] gid={}", tag, info.get("gid"))
            elif event == "metadata_fetch":
                _log.debug("[{}] fetching metadata via Mercury", tag)
            elif event == "metadata_ready":
                t = info.get("track")
                if t is not None:
                    artists = ", ".join(a.name for a in t.artists) or "?"
                    _log.info(
                        "[{}] metadata: {} - {} ({}) [{} files] ({:.2f}s)",
                        tag, artists, t.name, t.duration_str, len(t.files),
                        float(info.get("elapsed_s") or 0.0),
                    )
            elif event == "format_selected":
                _log.info("[{}] format: {}", tag, info.get("format_name"))
            elif event == "audio_key_request":
                _log.debug("[{}] requesting AES key", tag)
            elif event == "audio_key_ready":
                _log.debug(
                    "[{}] AES key ready ({:.2f}s)",
                    tag,
                    float(info.get("elapsed_s") or 0.0),
                )
            elif event == "cdn_resolve":
                _log.debug("[{}] resolving CDN URL", tag)
            elif event == "cdn_ready":
                cdn = info.get("cdn_url") or ""
                # Strip query string (signed URL — no need to log the token).
                _log.info(
                    "[{}] CDN: {} (resolve {:.2f}s)",
                    tag,
                    cdn.split("?", 1)[0],
                    float(info.get("elapsed_s") or 0.0),
                )
            elif event == "download_start":
                _log.info("[{}] downloading from CDN", tag)
                last_bucket["v"] = 0
                await _set_stage("downloading")
            elif event == "download_progress":
                total = info.get("total") or 0
                received = info.get("received") or 0
                if total > 0:
                    bucket = (received * 4) // total  # 0..4
                    if 1 <= bucket <= 3 and bucket > last_bucket["v"]:
                        _log.debug(
                            "[{}] download {}% ({:.2f}/{:.2f} MiB)",
                            tag, bucket * 25,
                            received / 1048576, total / 1048576,
                        )
                        last_bucket["v"] = bucket
            elif event == "download_done":
                size = info.get("size") or 0
                _log.info(
                    "[{}] downloaded {:.2f} MiB encrypted ({:.2f}s)",
                    tag, size / 1048576,
                    float(info.get("elapsed_s") or 0.0),
                )
                # The librespot pipeline immediately decrypts after the CDN
                # transfer completes, so flip the stage now to mirror that.
                await _set_stage("decrypting")
            elif event == "decrypt_done":
                size = info.get("size") or 0
                _log.info(
                    "[{}] decrypted {:.2f} MiB ({:.2f}s)",
                    tag,
                    size / 1048576,
                    float(info.get("elapsed_s") or 0.0),
                )
            elif event == "saved":
                _log.info(
                    "[{}] saved {} ({:.2f} MiB, write {:.2f}s, total {:.2f}s)",
                    tag,
                    Path(info.get("file_path", "")).name,
                    (info.get("size") or 0) / 1048576,
                    float(info.get("elapsed_s") or 0.0),
                    float(info.get("total_elapsed_s") or 0.0),
                )

        return cb

    async def download(
        self, track: Track, dest_dir: str,
        *, on_stage: Optional[StageCallback] = None,
    ) -> DownloadResult:
        _log.info(
            "[spotify:{}] starting download (priority={})",
            track.track_id, SPOTIFY_FORMAT_PRIORITY,
        )
        progress = self._make_progress_logger(track.track_id, on_stage=on_stage)

        try:
            async with self._decrypt_sem:
                result = await self._with_session(
                    lambda s: download_track_with_session(
                        s,
                        track.track_id,
                        output_dir=dest_dir,
                        preferred_formats=SPOTIFY_FORMAT_PRIORITY,
                        progress=progress,
                    )
                )
        except TrackUnavailableError as e:
            raise ProviderError(
                f"spotify track {track.track_id} unavailable: {e}",
                reason="unavailable",
            ) from e
        except SpotifyDownloaderError as e:
            raise ProviderError(f"spotify download failed: {e}") from e

        file_path = result.file_path
        format_name = result.selected_format
        size_bytes = result.output_size_bytes

        # Spotify gives us OGG_VORBIS, but Telegram's bot API frequently
        # mis-classifies bot-uploaded OGG as voice messages — voice file_ids
        # can't be swapped via edit_message_media (MEDIA_NEW_INVALID), which
        # breaks the inline article→audio upgrade. MP3 is unambiguous: it
        # always rides as audio. We transcode every Spotify download to MP3
        # at the source bitrate (lossy→lossy at 320 is inaudible for typical
        # material; 160/96 sources match their source rate to avoid wasting
        # space).
        from core.audio_convert import transcode_to_mp3, ffmpeg_available
        ext = file_path.rsplit(".", 1)[-1].lower()
        if ext == "ogg" and ffmpeg_available():
            target_kbps = 320
            if format_name.startswith("OGG_VORBIS_"):
                try:
                    target_kbps = int(format_name.rsplit("_", 1)[-1])
                except ValueError:
                    pass
            mp3_path = await transcode_to_mp3(
                file_path, bitrate_kbps=target_kbps,
            )
            if mp3_path is not None and mp3_path.is_file():
                file_path = str(mp3_path)
                size_bytes = mp3_path.stat().st_size
                format_name = f"MP3_{target_kbps}_FROM_{format_name}"
                ext = "mp3"

        mime = {
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
            "aac": "audio/aac",
            "m4a": "audio/mp4",
            "mp4": "audio/mp4",
            "flac": "audio/flac",
        }.get(ext, "application/octet-stream")

        # Re-hydrate metadata from the librespot-side Track. Internal artists
        # carry spotify_id, so we can resolve nice artist URLs.
        st = result.track
        sp_artists_all = [
            ArtistRef(
                name=a.name,
                artist_id=a.spotify_id,
                url=f"https://open.spotify.com/artist/{a.spotify_id}",
            )
            for a in st.artists
        ] or list(track.artists)
        # For downstream tags/scrobbling we want Spotify's primary artist only.
        sp_artists = sp_artists_all[:1] if sp_artists_all else []

        # Featured-artist names get stripped from track.artists (so the
        # display caption / artist button stays clean), but we still want
        # them in the file's ID3 tags so players can show "feat. X". Pipe
        # them through extra so the tagger can pick them up.
        featured_names = [a.name for a in st.featured_artists if a.name]
        new_extra = dict(track.extra or {})
        if featured_names:
            new_extra["featured_artists"] = featured_names
        if len(sp_artists_all) > 1:
            new_extra["all_artists"] = [a.name for a in sp_artists_all]

        merged = track.model_copy(update={
            "title": st.name or track.title,
            "artists": sp_artists,
            "album": (st.album.name if st.album else None) or track.album,
            "duration_seconds": (st.duration_ms // 1000) or track.duration_seconds,
            "artwork_url": (st.album.cover_url if st.album else None) or track.artwork_url,
            "isrc": st.isrc or track.isrc,
            "extra": new_extra,
        })

        _log.info(
            "[spotify:{}] download complete: {} ({}, {:.2f} MiB)",
            track.track_id, Path(file_path).name, format_name,
            size_bytes / 1048576,
        )
        return DownloadResult(
            track=merged,
            file_path=file_path,
            format_name=format_name,
            size_bytes=size_bytes,
            mime_type=mime,
        )
