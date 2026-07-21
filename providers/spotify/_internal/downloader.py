"""High-level async download pipeline."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiohttp

from core.filenames import safe_filename

from . import aes
from . import api as api_mod
from .auth import DEFAULT_HEADERS, HTTP_TIMEOUT, get_access_token
from .exceptions import DownloadError
from .ids import base62_to_gid, parse_track_id
from .librespot import Session
from .models import DownloadResult, Track


ProgressCallback = Callable[[str, dict], Awaitable[None] | None]


async def _emit(cb: Optional[ProgressCallback], event: str, **info) -> None:
    if cb is None:
        return
    r = cb(event, info)
    if asyncio.iscoroutine(r):
        await r


def _safe_filename(name: str) -> str:
    return safe_filename(name)


async def fetch_track_info(sp_dc: str, track_input: str) -> Track:
    """Just metadata. No download. Useful for previewing."""
    track_id = parse_track_id(track_input)
    gid_hex = base62_to_gid(track_id).hex()
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT) as http:
        token = (await get_access_token(sp_dc, session=http))["accessToken"]
        async with Session(token, http=http) as sess:
            return await api_mod.fetch_track(sess, gid_hex)


async def _download_with_session(
    sess: Session,
    gid_hex: str,
    *,
    output_dir: str | os.PathLike,
    output_path: str | os.PathLike | None,
    preferred_formats: Optional[list[str]],
    progress: Optional[ProgressCallback],
) -> DownloadResult:
    """Run the metadata -> key -> CDN -> decrypt -> write pipeline against an
    already-connected Session. Used both by the standalone CLI flow (which
    wraps its own Session) and the long-lived bot provider (which caches a
    Session and reuses it across requests)."""
    http = sess.http
    if http is None:
        raise DownloadError("session has no HTTP client attached")
    access_token = sess.access_token

    t0 = time.perf_counter()
    await _emit(progress, "metadata_fetch")
    track = await api_mod.fetch_track(sess, gid_hex)
    await _emit(progress, "metadata_ready", track=track, elapsed_s=time.perf_counter() - t0)

    t_format = time.perf_counter()
    audio_file, file_ext = api_mod.select_best_file(track, preferred_formats)
    await _emit(progress, "format_selected",
                format_name=audio_file.format_name, file_ext=file_ext, elapsed_s=time.perf_counter() - t_format)

    t_key = time.perf_counter()
    await _emit(progress, "audio_key_request")
    file_id = bytes.fromhex(audio_file.file_id_hex)
    # Use the resolved track gid (may differ from input if linked).
    track_gid = bytes.fromhex(track.gid_hex)
    aes_key = await sess.request_audio_key(file_id, track_gid)
    await _emit(progress, "audio_key_ready", aes_key_hex=aes_key.hex(), elapsed_s=time.perf_counter() - t_key)

    t_cdn = time.perf_counter()
    await _emit(progress, "cdn_resolve")
    cdn_url = await api_mod.resolve_cdn_url(http, audio_file.file_id_hex, access_token)
    await _emit(progress, "cdn_ready", cdn_url=cdn_url, elapsed_s=time.perf_counter() - t_cdn)

    t_dl = time.perf_counter()
    await _emit(progress, "download_start")
    try:
        async with http.get(
            cdn_url, timeout=aiohttp.ClientTimeout(total=300),
        ) as r:
            if r.status != 200:
                raise DownloadError(f"CDN GET {r.status}")
            total = int(r.headers.get("Content-Length") or 0)
            chunks: list[bytes] = []
            received = 0
            async for chunk in r.content.iter_chunked(64 * 1024):
                chunks.append(chunk)
                received += len(chunk)
                await _emit(progress, "download_progress",
                            received=received, total=total)
    except aiohttp.ClientError as e:
        raise DownloadError(f"CDN download failed: {e}") from e
    encrypted = b"".join(chunks)
    await _emit(
        progress, "download_done",
        size=len(encrypted),
        elapsed_s=time.perf_counter() - t_dl,
    )

    t_dec = time.perf_counter()
    await _emit(progress, "decrypt_start")
    # AES-CTR over multi-megabyte payloads in pure Python is heavily CPU
    # bound (~50s for a 10 MiB track on a typical box) and was previously
    # running on the asyncio loop — that froze the bot for the whole
    # decrypt window. Push it to a worker thread so the loop keeps
    # servicing inline queries / messages / heartbeats while one track
    # crunches through.
    decrypted = await asyncio.to_thread(
        aes.ctr_xor, aes_key, aes.SPOTIFY_AUDIO_IV, encrypted,
    )
    skip = 167 if file_ext == "ogg" else 0
    output_bytes = decrypted[skip:]
    await _emit(
        progress, "decrypt_done",
        size=len(output_bytes),
        elapsed_s=time.perf_counter() - t_dec,
    )

    t_write = time.perf_counter()
    if output_path is None:
        artist_part = ", ".join(track.all_artist_names) or "Unknown Artist"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = _safe_filename(f"{artist_part} - {track.name}") + f".{file_ext}"
        final_path = out_dir / fname
    else:
        final_path = Path(output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

    # write_bytes on a multi-MB payload can stall the loop on slow disks;
    # off-thread keeps the event loop responsive.
    await asyncio.to_thread(final_path.write_bytes, output_bytes)
    await _emit(
        progress, "saved",
        file_path=str(final_path),
        size=len(output_bytes),
        elapsed_s=time.perf_counter() - t_write,
        total_elapsed_s=time.perf_counter() - t0,
    )

    return DownloadResult(
        track=track,
        selected_format=audio_file.format_name,
        file_path=str(final_path),
        output_size_bytes=len(output_bytes),
        encrypted_size_bytes=len(encrypted),
        cdn_url=cdn_url,
        aes_key_hex=aes_key.hex(),
    )


async def download_track_with_session(
    session: Session,
    track_input: str,
    *,
    output_dir: str | os.PathLike = "downloads",
    output_path: str | os.PathLike | None = None,
    preferred_formats: Optional[list[str]] = None,
    progress: Optional[ProgressCallback] = None,
) -> DownloadResult:
    """Download using a caller-managed Session. The session must already be
    connected and have an HTTP client + a fresh access_token. Caller owns the
    session lifecycle (we do not close it)."""
    track_id = parse_track_id(track_input)
    gid_hex = base62_to_gid(track_id).hex()
    await _emit(progress, "track_id", track_id=track_id, gid=gid_hex)
    return await _download_with_session(
        session, gid_hex,
        output_dir=output_dir, output_path=output_path,
        preferred_formats=preferred_formats, progress=progress,
    )


async def download_track(
    sp_dc: str,
    track_input: str,
    *,
    output_dir: str | os.PathLike = "downloads",
    output_path: str | os.PathLike | None = None,
    preferred_formats: Optional[list[str]] = None,
    progress: Optional[ProgressCallback] = None,
) -> DownloadResult:
    """Full pipeline: auth -> librespot -> metadata -> AES key -> CDN -> decrypt -> write."""
    track_id = parse_track_id(track_input)
    gid_hex = base62_to_gid(track_id).hex()
    await _emit(progress, "track_id", track_id=track_id, gid=gid_hex)

    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT) as http:
        await _emit(progress, "auth_start")
        token_info = await get_access_token(sp_dc, session=http)
        access_token = token_info["accessToken"]
        await _emit(
            progress, "auth_done",
            client_id=token_info.get("clientId"),
            anonymous=token_info.get("isAnonymous"),
            expires_ms=token_info.get("accessTokenExpirationTimestampMs"),
        )

        await _emit(progress, "session_connect")
        async with Session(access_token, http=http) as sess:
            await _emit(progress, "session_ready", device_id=sess.device_id)
            return await _download_with_session(
                sess, gid_hex,
                output_dir=output_dir, output_path=output_path,
                preferred_formats=preferred_formats, progress=progress,
            )
