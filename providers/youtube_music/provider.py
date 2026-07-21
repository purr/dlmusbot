"""YouTubeMusicProvider — yt-dlp wrapper.

Defaults to the Android Music player client so most age-gated / region-locked
public Music tracks work without cookies (same trick cobalt.tools and
Invidious use). Cookies are opt-in for private library content.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

import yt_dlp

from core.exceptions import ProviderError, TrackNotFoundError
from core.filenames import safe_filename
from core.models import ArtistRef, DownloadResult, Playlist, Track

from ..base import Provider, StageCallback

log = logging.getLogger(__name__)


# yt-dlp player_client trick — bypasses age-gate / SABR throttling without cookies.
# Order matters: yt-dlp tries each in turn, first success wins.
#
# As of mid-2025, YouTube enforces a "Sign in to confirm you're not a bot"
# gate on most server-side IPs that don't carry a valid PO Token. None of
# the public player clients bypass this any more — the only reliable
# fixes are:
#   1. supply YT_COOKIES_FILE in config.py (export from your browser),
#   2. run a local PO Token provider (e.g. bgutil-ytdlp-pot-provider).
# We still try the modern client list first because authenticated /
# residential IPs sometimes don't get gated.
DEFAULT_PLAYER_CLIENTS = [
    "tv",  # works with --no-cookies on some IPs
    "mweb",  # mobile web — newer, partial pot exemption
    "web_safari",  # safari UA bypass
    "android_music",  # legacy fallback
    "android",
    "web",
]


# yt-dlp's bot-check error message stem. Used to convert opaque DownloadError
# instances into a user-readable ProviderError up the stack.
_BOT_CHECK_NEEDLES = (
    "sign in to confirm you",
    "use --cookies",
)


def _ydl_opts(
    extra: Optional[dict] = None, *, cookies_file: Optional[str] = None
) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": False,
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
        (
            "playlist",
            re.compile(r"music\.youtube\.com/playlist\?list=([A-Za-z0-9_\-]+)"),
        ),
        (
            "track",
            re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_\-]{11})"),
        ),
        ("track", re.compile(r"youtube\.com/shorts/([A-Za-z0-9_\-]{11})")),
    ]

    def __init__(self, cookies_file: Optional[str] = None):
        self._cookies_file = cookies_file or None

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
        info = await asyncio.to_thread(self._extract_info, url)
        if not info:
            raise TrackNotFoundError(f"yt music {entity_id} not found")
        t = _entry_to_track(info)
        if t is None:
            raise TrackNotFoundError(f"yt music {entity_id} not parseable")
        return t

    async def get_playlist(self, entity_id: str) -> Optional[Playlist]:
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
        return Playlist(
            provider="youtube_music",
            playlist_id=entity_id,
            title=info.get("title") or "<unknown>",
            owner=info.get("uploader"),
            url=f"https://music.youtube.com/playlist?list={entity_id}",
            tracks=tracks,
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

        result = await asyncio.to_thread(
            self._extract_audio,
            track.track_id,
            out_template,
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
        opts = _ydl_opts(extra, cookies_file=self._cookies_file)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            msg = str(e).lower()
            if any(n in msg for n in _BOT_CHECK_NEEDLES):
                # Surface the auth wall as a typed error so the bot layer
                # can show a helpful message ("set YT_COOKIES_FILE") and
                # the SoundCloud fallback can kick in for download paths.
                raise ProviderError(
                    "YouTube is gating this IP behind 'Sign in to confirm you're not a bot'. "
                    "Set YT_COOKIES_FILE in config.py (export cookies from your browser) "
                    "or expect YT Music links to fail."
                ) from e
            log.warning("yt-dlp extract_info failed: %s", e)
            return None

    def _extract_audio(self, video_id: str, out_template: str) -> Optional[dict]:
        opts = _ydl_opts(
            {
                "skip_download": False,
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": out_template,
                "noplaylist": True,
            },
            cookies_file=self._cookies_file,
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://music.youtube.com/watch?v={video_id}",
                    download=True,
                )
        except yt_dlp.utils.DownloadError as e:
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
