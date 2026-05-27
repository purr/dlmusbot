"""Best-effort cover-art / metadata embedding via mutagen + thumbnail prep
for Telegram's `thumbnail` parameter.

Mutagen runs sync — wrapped in to_thread to keep the event loop free.
Failures are logged and swallowed; an untagged audio is still useful.
"""

from __future__ import annotations

import asyncio
import io
import subprocess
from pathlib import Path
from typing import Optional

import aiohttp
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, ID3NoHeaderError, TALB, TIT2, TPE1, TPE2, TXXX, WXXX, WOAS
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
from PIL import Image

from core.audio_convert import ffmpeg_available
from core.logging_setup import logger
from core.models import DownloadResult, Track


# Telegram's thumbnail constraints for sendAudio:
#   - JPEG only
#   - <= 200 KB
#   - width and height <= 320
TG_THUMB_MAX_DIM = 320
TG_THUMB_MAX_BYTES = 200 * 1024


async def fetch_cover(http: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Fetch cover art bytes. Returns None on any failure."""
    if not url:
        return None
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200:
                logger.error("cover fetch HTTP {} for {}", r.status, url)
                return None
            data = await r.read()
            return data if data else None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("cover fetch failed for {} ({}): {}", url, type(e).__name__, e)
        return None


def _prepare_telegram_thumbnail_sync(cover: bytes) -> Optional[bytes]:
    if not cover:
        return None
    try:
        img = Image.open(io.BytesIO(cover))
        img.load()
    except Exception as e:
        logger.error("thumbnail PIL decode failed ({}): {}", type(e).__name__, e)
        return None

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if w > TG_THUMB_MAX_DIM or h > TG_THUMB_MAX_DIM:
        scale = min(TG_THUMB_MAX_DIM / w, TG_THUMB_MAX_DIM / h)
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    # Try descending quality until we fit under TG_THUMB_MAX_BYTES.
    for q in (90, 80, 70, 60, 50, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= TG_THUMB_MAX_BYTES:
            return data
    return None


async def prepare_telegram_thumbnail(cover: bytes) -> Optional[bytes]:
    """Resize/encode a cover image into a Telegram-thumbnail-compatible JPEG.
    PIL decode + resize + JPEG encode are CPU-bound and were stalling the
    asyncio loop on bigger covers (~1 MB album artwork), so we hop to a
    worker thread."""
    return await asyncio.to_thread(_prepare_telegram_thumbnail_sync, cover)


async def embed_metadata(
    result: DownloadResult,
    http: aiohttp.ClientSession,
    *,
    original_spotify_url: Optional[str] = None,
) -> Optional[bytes]:
    """Embed title/artist/album + cover art into the file (in place).
    Returns the raw cover bytes (if fetched) so the caller can also build a
    Telegram thumbnail without re-downloading."""
    track = result.track
    cover_bytes: Optional[bytes] = None
    if track.artwork_url:
        cover_bytes = await fetch_cover(http, track.artwork_url)

    try:
        await asyncio.to_thread(
            _tag_sync,
            result.file_path,
            track,
            cover_bytes,
            original_spotify_url,
        )
    except Exception as e:
        name = type(e).__name__
        msg = f"tag embedding skipped for {result.file_path} ({name}: {e})"
        if name in {"OggVorbisHeaderError", "OggOpusHeaderError"}:
            logger.info(msg)
        else:
            logger.warning(msg)

    return cover_bytes


def _tag_sync(
    path: str,
    track: Track,
    cover: Optional[bytes],
    original_spotify_url: Optional[str],
) -> None:
    urls = _collect_urls(track, original_spotify_url=original_spotify_url)
    extras = _collect_extras(track)
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "mp3":
        _tag_id3(path, track.title, track.artists_str, track.album, cover, urls, extras)
    elif ext == "flac":
        _tag_flac(path, track.title, track.artists_str, track.album, cover, urls, extras)
    elif ext in ("m4a", "mp4"):
        _tag_mp4(path, track.title, track.artists_str, track.album, cover, urls, extras)
    elif ext == "ogg":
        _tag_ogg(path, track.title, track.artists_str, track.album, cover, OggVorbis, urls, extras)
    elif ext == "opus":
        _tag_ogg(path, track.title, track.artists_str, track.album, cover, OggOpus, urls, extras)


def _collect_extras(track: Track) -> dict:
    """Pull provider-supplied extra metadata (featured artists, full artist
    list, etc.) out of `track.extra` for tagging. Spotify strips featured
    artists from `track.artists` so the inline UI / artist button stays
    clean — but they should still land in the file's tags."""
    out: dict = {}
    feat = track.extra.get("featured_artists") if track.extra else None
    if isinstance(feat, list):
        out["featured_artists"] = [str(x) for x in feat if x]
    all_artists = track.extra.get("all_artists") if track.extra else None
    if isinstance(all_artists, list):
        out["all_artists"] = [str(x) for x in all_artists if x]
    isrc = getattr(track, "isrc", None)
    if isrc:
        out["isrc"] = str(isrc)
    return out


def _collect_urls(track: Track, *, original_spotify_url: Optional[str]) -> dict:
    artist_urls = [a.url for a in track.artists if a.url]
    # Keep insertion order and dedupe.
    artist_urls = list(dict.fromkeys(artist_urls))
    album_url = ""
    raw_album_url = track.extra.get("album_url")
    if isinstance(raw_album_url, str) and raw_album_url:
        album_url = raw_album_url
    elif track.provider == "spotify":
        album_id = track.extra.get("album_id")
        if isinstance(album_id, str) and album_id:
            album_url = f"https://open.spotify.com/album/{album_id}"

    return {
        "track_url": track.url or "",
        "album_url": album_url,
        "artist_urls": artist_urls,
        "source_url": original_spotify_url or track.url or "",
        "spotify_url": original_spotify_url or "",
        "permalink_url": (
            track.extra.get("permalink_url")
            if isinstance(track.extra.get("permalink_url"), str)
            else ""
        ),
    }


def _tag_id3(path, title, artist, album, cover, urls: dict, extras: dict):
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags["TIT2"] = TIT2(encoding=3, text=title)
    tags["TPE1"] = TPE1(encoding=3, text=artist)
    if album:
        tags["TALB"] = TALB(encoding=3, text=album)
    if cover:
        tags["APIC"] = APIC(
            encoding=3, mime="image/jpeg", type=3,
            desc="Cover", data=cover,
        )
    track_url = urls.get("track_url") or ""
    if track_url:
        tags["WOAS"] = WOAS(url=track_url)
    _set_wxxx(tags, "SOURCE_URL", urls.get("source_url"))
    _set_wxxx(tags, "SPOTIFY_URL", urls.get("spotify_url"))
    _set_wxxx(tags, "ALBUM_URL", urls.get("album_url"))
    _set_wxxx(tags, "PERMALINK_URL", urls.get("permalink_url"))
    for i, url in enumerate(urls.get("artist_urls") or [], start=1):
        _set_wxxx(tags, f"ARTIST_URL_{i}", url)
    # TPE2 is conventionally the album artist; default to the lead
    # performer (TPE1) so libraries group correctly.
    tags["TPE2"] = TPE2(encoding=3, text=artist)
    feat = extras.get("featured_artists") or []
    if feat:
        _set_txxx(tags, "FEATURED_ARTISTS", ", ".join(feat))
    all_artists = extras.get("all_artists") or []
    if all_artists:
        _set_txxx(tags, "ALL_ARTISTS", ", ".join(all_artists))
    isrc = extras.get("isrc")
    if isrc:
        _set_txxx(tags, "ISRC", isrc)
    tags.save(path, v2_version=3)


def _tag_flac(path, title, artist, album, cover, urls: dict, extras: dict):
    f = FLAC(path)
    f["title"] = title
    f["artist"] = artist
    if album:
        f["album"] = album
    _set_vorbis_urls(f, urls)
    _set_vorbis_extras(f, extras)
    f.clear_pictures()
    if cover:
        pic = Picture()
        pic.data = cover
        pic.type = 3
        pic.mime = "image/jpeg"
        f.add_picture(pic)
    f.save()


def _tag_mp4(path, title, artist, album, cover, urls: dict, extras: dict):
    f = MP4(path)
    f["\xa9nam"] = title
    f["\xa9ART"] = artist
    if album:
        f["\xa9alb"] = album
    _set_mp4_urls(f, urls)
    f["aART"] = artist  # album artist
    feat = extras.get("featured_artists") or []
    if feat:
        f["----:com.apple.iTunes:FEATURED_ARTISTS"] = [", ".join(feat).encode("utf-8")]
    if extras.get("all_artists"):
        f["----:com.apple.iTunes:ALL_ARTISTS"] = [
            ", ".join(extras["all_artists"]).encode("utf-8")
        ]
    if extras.get("isrc"):
        f["----:com.apple.iTunes:ISRC"] = [extras["isrc"].encode("utf-8")]
    if cover:
        f["covr"] = [MP4Cover(cover, imageformat=MP4Cover.FORMAT_JPEG)]
    f.save()


def _tag_ogg(path, title, artist, album, cover, klass, urls: dict, extras: dict):
    try:
        _apply_ogg_tags(path, title, artist, album, cover, klass, urls, extras)
        return
    except Exception:
        if not ffmpeg_available():
            raise
        _repair_ogg_container(path)
        _apply_ogg_tags(path, title, artist, album, cover, klass, urls, extras)


def _apply_ogg_tags(path, title, artist, album, cover, klass, urls: dict, extras: dict):
    f = klass(path)
    f["title"] = title
    f["artist"] = artist
    if album:
        f["album"] = album
    _set_vorbis_urls(f, urls)
    _set_vorbis_extras(f, extras)
    if cover:
        import base64
        pic = Picture()
        pic.data = cover
        pic.type = 3
        pic.mime = "image/jpeg"
        f["metadata_block_picture"] = [base64.b64encode(pic.write()).decode("ascii")]
    f.save()


def _repair_ogg_container(path: str) -> None:
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"missing ogg file: {src}")
    repaired = src.with_name(f"{src.stem}.repaired{src.suffix}")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(repaired),
        ],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0 or not repaired.is_file():
        stderr = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"ffmpeg ogg repair failed: {stderr.strip()}")
    repaired.replace(src)


def _set_wxxx(tags: ID3, desc: str, url: Optional[str]) -> None:
    if not url:
        return
    key = f"WXXX:{desc}"
    tags.delall(key)
    tags.add(WXXX(encoding=3, desc=desc, url=url))


def _set_txxx(tags: ID3, desc: str, value: Optional[str]) -> None:
    if not value:
        return
    key = f"TXXX:{desc}"
    tags.delall(key)
    tags.add(TXXX(encoding=3, desc=desc, text=value))


def _set_vorbis_extras(f, extras: dict) -> None:
    feat = extras.get("featured_artists") or []
    if feat:
        f["featured_artists"] = feat
    if extras.get("all_artists"):
        f["all_artists"] = extras["all_artists"]
    if extras.get("isrc"):
        f["isrc"] = extras["isrc"]
    # Default album-artist to the existing artist tag so libraries group.
    artist_val = f.get("artist")
    if artist_val and not f.get("albumartist"):
        f["albumartist"] = artist_val


def _set_vorbis_urls(f, urls: dict) -> None:
    if urls.get("track_url"):
        f["track_url"] = urls["track_url"]
        f["website"] = urls["track_url"]
    if urls.get("source_url"):
        f["source_url"] = urls["source_url"]
    if urls.get("spotify_url"):
        f["spotify_url"] = urls["spotify_url"]
    if urls.get("album_url"):
        f["album_url"] = urls["album_url"]
    if urls.get("permalink_url"):
        f["permalink_url"] = urls["permalink_url"]
    artist_urls = urls.get("artist_urls") or []
    if artist_urls:
        f["artist_url"] = artist_urls


def _set_mp4_urls(f: MP4, urls: dict) -> None:
    def put(key: str, value: str) -> None:
        if value:
            f[f"----:com.apple.iTunes:{key}"] = [value.encode("utf-8")]

    put("TRACK_URL", urls.get("track_url") or "")
    put("SOURCE_URL", urls.get("source_url") or "")
    put("SPOTIFY_URL", urls.get("spotify_url") or "")
    put("ALBUM_URL", urls.get("album_url") or "")
    put("PERMALINK_URL", urls.get("permalink_url") or "")
    for i, url in enumerate(urls.get("artist_urls") or [], start=1):
        put(f"ARTIST_URL_{i}", url)
