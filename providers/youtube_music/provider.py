"""YouTubeMusicProvider — yt-dlp wrapper.

Defaults to the Android Music player client so most age-gated / region-locked
public Music tracks work without cookies (same trick cobalt.tools and
Invidious use). Cookies are opt-in for private library content.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import yt_dlp
from yt_dlp.cookies import YoutubeDLCookieJar

from core.exceptions import ProviderError, TrackNotFoundError
from core.filenames import safe_filename
from core.models import ArtistRef, DownloadResult, Playlist, Track

from ..base import Provider, StageCallback
from .browser_session import BrowserTokenMinter

log = logging.getLogger(__name__)


# yt-dlp player_client trick — bypasses age-gate / SABR throttling without cookies.
# Order matters: yt-dlp tries each in turn, first success wins.
#
# YouTube enforces a "Sign in to confirm you're not a bot" gate on most
# server-side IPs that don't carry a valid PO Token, and no public player
# client bypasses it any more. When we hit that gate, `browser_session`
# mints a token in a headless chromium and we retry once; age-gated videos
# are a different wall entirely and need YT_COOKIES_FILE (a signed-in adult
# account) — nothing else lifts those. We still try the modern client list
# first because authenticated / residential IPs often aren't gated at all.
DEFAULT_PLAYER_CLIENTS = [
    "tv",  # works with --no-cookies on some IPs
    "mweb",  # mobile web — newer, partial pot exemption
    "web_safari",  # safari UA bypass
    "android_music",  # legacy fallback
    "android",
    "web",
]


# yt-dlp reports every failure as an opaque DownloadError carrying the
# extractor's own English message. Map those onto our failure reasons so the
# bot can say what actually went wrong (and hand the song to the Spotify
# fallback) instead of showing a bare "couldn't download".
#
# Order matters. YouTube prefixes its rate-limit text with "Video
# unavailable.", and its age-gate text carries the same "use --cookies" hint
# as the bot check — so the narrower needles have to be tested first.
# Transport-level flakes ride the normal retry path, so they're tested
# first: "HTTP Error 503: Service Unavailable" would otherwise read as a
# dead video and get marked permanent.
_TRANSIENT_NEEDLES = (
    "http error 5",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "name resolution",
)
_RATE_LIMIT_NEEDLES = ("try again later", "rate-limit")
_AGE_GATE_NEEDLES = ("confirm your age", "inappropriate for some users")
_BOT_CHECK_NEEDLES = ("sign in to confirm", "please sign in", "use --cookies")
_UNAVAILABLE_NEEDLES = (
    "unavailable",  # "Video unavailable", "This video is unavailable"
    "not available",  # "...is not available", "not available in your country"
    "no longer available",
    "in your country",
    "has been removed",
    "removed by the uploader",
    "private video",
    "copyright",
    "account associated with this video has been terminated",
)

# Human-readable stems for the log line; the user-facing wording lives in
# `bot.status.STATUS_ALERTS` under `final_failed:<reason>`.
_REASON_LABELS = {
    "rate_limited": "YouTube is rate-limiting this host",
    "age_gated": "video is age-restricted (needs signed-in adult cookies)",
    "bot_check": (
        "YouTube is gating this IP behind 'Sign in to confirm you're not a "
        "bot' — set YT_COOKIES_FILE in config.py (export cookies from your "
        "browser) or expect YT Music links to fail"
    ),
    "unavailable": "video is unavailable (removed, private or region-locked)",
}


def _classify_ydl_error(msg: str) -> Optional[str]:
    """Failure reason for a yt-dlp error message, or None if unrecognised
    (transient — the caller retries those)."""
    m = msg.lower()
    if any(n in m for n in _TRANSIENT_NEEDLES):
        return None
    if any(n in m for n in _RATE_LIMIT_NEEDLES):
        return "rate_limited"
    if any(n in m for n in _AGE_GATE_NEEDLES):
        return "age_gated"
    if any(n in m for n in _BOT_CHECK_NEEDLES):
        return "bot_check"
    if any(n in m for n in _UNAVAILABLE_NEEDLES):
        return "unavailable"
    return None


def _cookies_stamp(path: Optional[str]) -> Optional[tuple[float, int]]:
    """(mtime, size) of the cookie file, or None when it's absent."""
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def _fmt_epoch(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _validated_cookies_file(path: Optional[str]) -> Optional[str]:
    """Load-test a configured cookie jar, at startup and on every change.

    yt-dlp aborts *every* extraction when `cookiefile` points at a missing or
    non-Netscape file ("does not look like a Netscape format cookies file"),
    so one bad export silently turns into "YouTube is broken" for every user.
    Check it up front and say exactly what's wrong; run without cookies
    rather than taking the whole provider down with it.

    A jar that loads but has expired is kept (some entries are session
    cookies with no expiry at all) and only logged — unusable format is
    fatal to the file, staleness is a warning."""
    if not path:
        return None
    try:
        jar = YoutubeDLCookieJar(path)
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        log.error(
            "YT_COOKIES_FILE %r is unusable (%s: %s) - continuing WITHOUT "
            "cookies, so age-gated and bot-checked videos will fail. "
            "Re-export it in Netscape format (first line must be "
            "'# Netscape HTTP Cookie File').",
            path,
            type(e).__name__,
            e,
        )
        return None
    expiries = [c.expires for c in jar if c.expires]
    if expiries and max(expiries) <= time.time():
        log.error(
            "YT_COOKIES_FILE %r has only EXPIRED cookies (newest expired %s) - "
            "re-export it; age-gated videos will fail until you do.",
            path,
            _fmt_epoch(max(expiries)),
        )
    else:
        log.info(
            "youtube cookies loaded from %s (%d cookies, newest expiry %s)",
            path,
            len(jar),
            _fmt_epoch(max(expiries)) if expiries else "session-only",
        )
    return path


def _ydl_opts(
    extra: Optional[dict] = None, *, cookies_file: Optional[str] = None
) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": False,
        # Playlist and search extraction keeps going past a dead entry.
        # Single-entity calls pass `ignoreerrors: False` instead — with it
        # on, yt-dlp swallows the DownloadError and hands back None, and
        # the reason for the failure (age gate, rate limit, removed video)
        # is lost before `_classify_ydl_error` can read it.
        "ignoreerrors": True,
        "extractor_args": {"youtube": {"player_client": DEFAULT_PLAYER_CLIENTS}},
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    if extra:
        # Merge extractor_args carefully — never clobber the player_client list.
        if "extractor_args" in extra:
            for k, v in extra["extractor_args"].items():
                opts["extractor_args"].setdefault(k, {}).update(v)
            extra = {k: v for k, v in extra.items() if k != "extractor_args"}
        opts.update(extra)
    return opts


def _artist_refs_from_entry(entry: dict) -> list[ArtistRef]:
    artists_field = entry.get("artists") or []
    out: list[ArtistRef] = []
    if isinstance(artists_field, list) and artists_field:
        for a in artists_field:
            name = a if isinstance(a, str) else (a or {}).get("name")
            if not name:
                continue
            aid = None if isinstance(a, str) else (a or {}).get("id")
            out.append(
                ArtistRef(
                    name=name,
                    artist_id=aid,
                    url=f"https://music.youtube.com/channel/{aid}" if aid else None,
                )
            )
        return out
    name = entry.get("artist") or entry.get("uploader") or entry.get("channel") or ""
    name = re.sub(r"\s*-\s*Topic$", "", name)
    if not name:
        return []
    cid = entry.get("channel_id") or entry.get("uploader_id")
    return [
        ArtistRef(
            name=name,
            artist_id=cid,
            url=f"https://music.youtube.com/channel/{cid}" if cid else None,
        )
    ]


def _entry_to_track(entry: dict) -> Optional[Track]:
    if not isinstance(entry, dict):
        return None
    vid = entry.get("id") or entry.get("video_id")
    if not vid:
        return None
    title = entry.get("track") or entry.get("title") or "<unknown>"
    duration = int(entry.get("duration") or 0)
    thumbs = entry.get("thumbnails") or []
    artwork = _pick_artwork_url(vid, thumbs, entry.get("thumbnail"))
    return Track(
        provider="youtube_music",
        track_id=vid,
        title=title,
        artists=_artist_refs_from_entry(entry),
        album=entry.get("album"),
        duration_seconds=duration,
        artwork_url=artwork,
        url=f"https://music.youtube.com/watch?v={vid}",
    )


def _pick_artwork_url(
    video_id: str, thumbs: list[dict], fallback: Optional[str]
) -> str:
    """Prefer stable ytimg URLs for Telegram inline thumbnail rendering.

    yt-dlp search entries often expose `vi_webp/.../maxresdefault.webp`,
    which may 404. Telegram fetches inline thumbnails itself, so return a
    robust JPEG candidate chain.
    """
    urls: list[str] = []
    for t in thumbs:
        u = (t or {}).get("url")
        if isinstance(u, str) and u:
            urls.append(u)
    if isinstance(fallback, str) and fallback:
        urls.append(fallback)

    def _score(u: str) -> tuple[int, int]:
        lu = u.lower()
        # Prefer non-webp + non-maxres entries first.
        webp_penalty = 1 if ".webp" in lu or "/vi_webp/" in lu else 0
        maxres_penalty = 1 if "maxresdefault" in lu else 0
        return (webp_penalty, maxres_penalty)

    if urls:
        urls = sorted(dict.fromkeys(urls), key=_score)
        best = urls[0]
        # Rewrite brittle webp maxres URL to known-stable jpg variant.
        if "/vi_webp/" in best.lower() or "maxresdefault.webp" in best.lower():
            return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        return best

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def _sanitize_filename(name: str) -> str:
    return safe_filename(name)


class YouTubeMusicProvider(Provider):
    name = "youtube_music"
    label = "YouTube Music"

    URL_PATTERNS = [
        ("track", re.compile(r"music\.youtube\.com/watch\?v=([A-Za-z0-9_\-]{11})")),
        # Covers music.youtube.com *and* plain youtube.com playlists — the
        # music player serves an ordinary `PL...` list under the same
        # `/playlist?list=` path, which is what `canonical_url` emits.
        (
            "playlist",
            re.compile(r"youtube\.com/playlist\?list=([A-Za-z0-9_\-]+)"),
        ),
        (
            "track",
            re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_\-]{11})"),
        ),
        # Shorts and (finished) livestreams carry ordinary video ids.
        ("track", re.compile(r"youtube\.com/(?:shorts|live)/([A-Za-z0-9_\-]{11})")),
    ]

    def __init__(
        self, cookies_file: Optional[str] = None, *, browser_potoken: bool = True
    ):
        # Minting is lazy: constructing this launches nothing. The browser
        # only starts the first time YouTube actually gates a download.
        self._minter = BrowserTokenMinter() if browser_potoken else None
        self._cookies_path = cookies_file or None
        self._cookies_lock = threading.Lock()
        self._cookies_stamp = _cookies_stamp(self._cookies_path)
        self._validated_cookies = _validated_cookies_file(self._cookies_path)

    def _cookies_file(self) -> Optional[str]:
        """The validated cookie jar, re-checked whenever the file changes.

        Jars go stale on their own — YouTube rotates account cookies — so
        the fix is always "drop a fresh export in place". Watching the
        file's size+mtime means a refresher script (or you, by hand) makes
        that take effect on the very next request instead of needing a bot
        restart. One stat() per yt-dlp call, off the event loop already."""
        if self._cookies_path is None:
            return None
        stamp = _cookies_stamp(self._cookies_path)
        with self._cookies_lock:
            if stamp != self._cookies_stamp:
                log.info("YT_COOKIES_FILE changed on disk - revalidating")
                self._cookies_stamp = stamp
                self._validated_cookies = _validated_cookies_file(
                    self._cookies_path
                )
            return self._validated_cookies

    async def close(self) -> None:
        if self._minter is not None:
            await self._minter.close()

    async def _potoken_args(self, video_id: str) -> Optional[dict]:
        """yt-dlp extractor args carrying a freshly minted PO token, or
        None when we couldn't mint one. Pins `player_client` to `web`
        because that's the client the token is minted for — handing a
        web token to the android client is just a slower failure."""
        if self._minter is None:
            return None
        tok = await self._minter.mint(video_id)
        if tok is None:
            return None
        return {
            "ignoreerrors": False,
            "extractor_args": {
                "youtube": {
                    "player_client": ["web"],
                    "po_token": [f"web.gvs+{tok.token}"],
                    "visitor_data": [tok.visitor_data],
                }
            },
        }

    def canonical_url(self, kind: str, entity_id: str) -> str:
        if kind == "playlist":
            return f"https://music.youtube.com/playlist?list={entity_id}"
        return f"https://music.youtube.com/watch?v={entity_id}"

    def artist_url(self, artist_id: str) -> Optional[str]:
        if not artist_id:
            return None
        return f"https://music.youtube.com/channel/{artist_id}"

    async def search(self, query: str, limit: int = 25) -> list[Track]:
        if not query.strip():
            return []
        url = f"ytsearch{min(limit, 50)}:{query}"
        info = await asyncio.to_thread(
            self._extract_info,
            url,
            extra={"extract_flat": "in_playlist"},
        )
        if not info:
            return []
        return [
            t for t in (_entry_to_track(e) for e in (info.get("entries") or [])) if t
        ]

    async def get_track(self, entity_id: str) -> Track:
        url = f"https://music.youtube.com/watch?v={entity_id}"
        try:
            info = await asyncio.to_thread(
                self._extract_info, url, {"ignoreerrors": False}
            )
        except ProviderError as e:
            # The bot wall is the one failure a browser can actually lift:
            # mint a token for this exact video and ask once more. Every
            # other reason (age gate, removed, rate limit) is untouched by
            # a token, so it propagates immediately.
            if e.reason != "bot_check":
                raise
            extra = await self._potoken_args(entity_id)
            if extra is None:
                raise
            info = await asyncio.to_thread(self._extract_info, url, extra)
        if not info:
            raise TrackNotFoundError(f"yt music {entity_id} not found")
        t = _entry_to_track(info)
        if t is None:
            raise TrackNotFoundError(f"yt music {entity_id} not parseable")
        return t

    async def get_playlist(
        self, entity_id: str, *, offset: int = 0, limit: Optional[int] = None
    ) -> Optional[Playlist]:
        url = f"https://music.youtube.com/playlist?list={entity_id}"
        info = await asyncio.to_thread(
            self._extract_info,
            url,
            extra={"extract_flat": "in_playlist"},
        )
        if not info:
            return None
        entries = info.get("entries") or []
        tracks = [t for t in (_entry_to_track(e) for e in entries) if t]
        total = len(tracks)
        if offset or limit is not None:
            tracks = tracks[offset : (offset + limit) if limit else None]
        return Playlist(
            provider="youtube_music",
            playlist_id=entity_id,
            title=info.get("title") or "<unknown>",
            owner=info.get("uploader"),
            url=f"https://music.youtube.com/playlist?list={entity_id}",
            tracks=tracks,
            total_tracks=total,
        )

    async def download(
        self,
        track: Track,
        dest_dir: str,
        *,
        on_stage: Optional[StageCallback] = None,
    ) -> DownloadResult:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        artist_part = ", ".join(a.name for a in track.artists) or "Unknown Artist"
        out_stem = _sanitize_filename(f"{artist_part} - {track.title}")
        out_template = str(Path(dest_dir) / f"{out_stem}.%(ext)s")

        if on_stage is not None:
            try:
                await on_stage("downloading")
            except Exception:
                log.debug("on_stage(downloading) failed", exc_info=True)

        try:
            result = await asyncio.to_thread(
                self._extract_audio,
                track.track_id,
                out_template,
            )
        except ProviderError as e:
            if e.reason != "bot_check":
                raise
            extra = await self._potoken_args(track.track_id)
            if extra is None:
                raise
            result = await asyncio.to_thread(
                self._extract_audio,
                track.track_id,
                out_template,
                extra,
            )
        if not result:
            raise ProviderError(f"yt-dlp failed to download {track.track_id}")

        path = Path(result["file_path"])
        if not path.is_file():
            raise ProviderError(f"yt-dlp reported success but no file at {path}")

        return DownloadResult(
            track=track,
            file_path=str(path),
            format_name=result.get("format_name") or "unknown",
            size_bytes=path.stat().st_size,
            mime_type=result.get("mime_type") or "audio/mpeg",
        )

    # ---- yt-dlp helpers (sync, run in to_thread) ------------------------

    def _extract_info(self, url: str, extra: Optional[dict] = None) -> Optional[dict]:
        opts = _ydl_opts(extra, cookies_file=self._cookies_file())
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            reason = _classify_ydl_error(str(e))
            if reason:
                # Typed reason so the bot layer shows the right explanation
                # and the cross-provider fallback can look for the same song
                # on Spotify instead of retrying a wall we can't get past.
                raise ProviderError(
                    f"yt-dlp: {_REASON_LABELS[reason]} ({url})", reason=reason
                ) from e
            log.warning("yt-dlp extract_info failed: %s", e)
            return None

    def _extract_audio(
        self,
        video_id: str,
        out_template: str,
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        opts = _ydl_opts(
            {
                "skip_download": False,
                "ignoreerrors": False,
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": out_template,
                "noplaylist": True,
                **(extra or {}),
            },
            cookies_file=self._cookies_file(),
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://music.youtube.com/watch?v={video_id}",
                    download=True,
                )
        except yt_dlp.utils.DownloadError as e:
            reason = _classify_ydl_error(str(e))
            if reason:
                # Permanent for this provider — don't burn the retry budget
                # on it; let the job runner explain it / fall back.
                raise ProviderError(
                    f"yt-dlp: {_REASON_LABELS[reason]} ({video_id})", reason=reason
                ) from e
            log.warning("yt-dlp download failed: %s", e)
            return None
        if not info:
            return None
        path = (
            (info.get("requested_downloads") or [{}])[0].get("filepath")
            or info.get("filepath")
            or info.get("_filename")
        )
        if not path:
            ext = info.get("ext") or "m4a"
            path = out_template.replace("%(ext)s", ext)
        ext = Path(path).suffix.lstrip(".")
        mime = {
            "m4a": "audio/mp4",
            "mp4": "audio/mp4",
            "webm": "audio/webm",
            "opus": "audio/ogg",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
        }.get(ext, "audio/mpeg")
        return {
            "file_path": path,
            "format_name": info.get("format") or info.get("format_id") or ext,
            "mime_type": mime,
        }
