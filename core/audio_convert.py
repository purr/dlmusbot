"""ffmpeg-based audio transcoding.

Spotify only hands us OGG_VORBIS for non-HiFi accounts (FLAC is DRM-locked).
OGG carries Vorbis comments natively but lots of media players + Telegram
clients render ID3 metadata more reliably, so we transcode to MP3 320 CBR
before tagging. Lossy → lossy adds one re-encode generation; the difference
is inaudible at 320 CBR for typical material.

This module also handles the **fit-to-cap** path: when a downloaded file
overshoots Telegram's size limit, we re-encode at a lower bitrate that
keeps the audio listenable while shrinking the payload. The threshold for
"still listenable" is set by `MIN_LISTENABLE_KBPS` — anything below that
makes it not worth shipping.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from loguru import logger


# Lower bound for "still listenable" MP3 CBR. Below this, re-encoding to
# fit a size cap produces audibly compromised audio (bubbly highs, slurred
# transients) and we'd rather fail fast than ship junk.
MIN_LISTENABLE_KBPS = 128

# Conservative source-bitrate assumption used for upfront duration → size
# estimates before we've actually downloaded anything. Spotify hands us
# OGG_VORBIS_320 for free / standard accounts; SoundCloud / YT Music are
# usually lower. Using 320 as the worst case means we only fail-fast when
# the track genuinely can't fit at any sane quality.
ASSUMED_SOURCE_KBPS = 320


def estimate_size_mb(duration_seconds: int, bitrate_kbps: int) -> float:
    """Bitrate-based size estimate. Slight safety overhead added for
    container framing / ID3 / cover art (typically ~1–2% of payload)."""
    if duration_seconds <= 0 or bitrate_kbps <= 0:
        return 0.0
    raw = (duration_seconds * bitrate_kbps) / 8 / 1024
    return raw * 1.02


def target_bitrate_for_size(
    duration_seconds: int, target_mb: float, *, safety_margin: float = 0.95,
) -> int:
    """Compute the highest CBR bitrate that fits `target_mb` for a track
    of `duration_seconds`. Includes a safety margin so encoded files come
    in under (not exactly at) the target. Returns 0 if even the lowest
    listenable rate (`MIN_LISTENABLE_KBPS`) wouldn't fit."""
    if duration_seconds <= 0:
        return 0
    raw_kbps = int((target_mb * safety_margin * 8 * 1024) / duration_seconds)
    # Round down to a "clean" bitrate (multiples of 16 kbps).
    raw_kbps -= raw_kbps % 16
    if raw_kbps < MIN_LISTENABLE_KBPS:
        return 0
    # Don't bother encoding *higher* than the source.
    return min(raw_kbps, 320)


class TranscodeError(RuntimeError):
    """ffmpeg not available or returned non-zero."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: list[str], timeout: float) -> None:
    try:
        proc = subprocess.run(
            args, capture_output=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        raise TranscodeError(f"ffmpeg invocation failed: {e}") from e
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise TranscodeError(f"ffmpeg exit {proc.returncode}: {stderr.strip()}")


def _run_capture(args: list[str], timeout: float) -> str:
    try:
        proc = subprocess.run(
            args, capture_output=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        raise TranscodeError(f"tool invocation failed: {e}") from e
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise TranscodeError(f"command exit {proc.returncode}: {stderr.strip()}")
    return (proc.stdout or b"").decode("utf-8", "replace")


def _probe_duration_sync(src: Path, timeout: float) -> float:
    out = _run_capture(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(src),
        ],
        timeout=timeout,
    ).strip()
    try:
        return float(out)
    except ValueError as e:
        raise TranscodeError(f"ffprobe returned non-float duration: {out!r}") from e


def _detect_silence_events_sync(
    src: Path, *, silence_db: float, min_detect_seconds: float, timeout: float,
) -> list[tuple[float, float, float]]:
    """Run ffmpeg silencedetect and return (start, end, duration) tuples."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v", "info",
                "-nostdin",
                "-i", str(src),
                "-af", f"silencedetect=noise={silence_db}dB:d={min_detect_seconds}",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        raise TranscodeError(f"ffmpeg silencedetect failed: {e}") from e
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise TranscodeError(f"silencedetect exit {proc.returncode}: {stderr.strip()}")

    stderr_text = (proc.stderr or b"").decode("utf-8", "replace")
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([0-9.]+)", stderr_text)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([0-9.]+)", stderr_text)]
    durs = [float(m.group(1)) for m in re.finditer(r"silence_duration:\s*([0-9.]+)", stderr_text)]

    n = min(len(starts), len(ends), len(durs))
    return [(starts[i], ends[i], durs[i]) for i in range(n)]


def _trim_edges_sync(
    src: Path, *, start_at: float, end_at: float, timeout: float,
) -> Path:
    dst = src.with_name(f"{src.stem}.trim{src.suffix}")
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            "-ss", f"{start_at:.3f}",
            "-to", f"{end_at:.3f}",
            "-i", str(src),
            "-vn",
            "-c:a", "copy",
            str(dst),
        ],
        timeout=timeout,
    )
    if not dst.is_file():
        raise TranscodeError(f"silence trim produced no output: {dst}")
    return dst


def _transcode_to_mp3_sync(
    src: Path, dst: Path, bitrate_kbps: int, timeout: float,
) -> None:
    args = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", str(src),
        "-vn",                              # drop any embedded image stream
        "-c:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        "-id3v2_version", "3",
        str(dst),
    ]
    _run_ffmpeg(args, timeout=timeout)


async def transcode_to_mp3(
    src_path: str | Path,
    *,
    bitrate_kbps: int = 320,
    timeout: float = 90.0,
    delete_src: bool = True,
) -> Optional[Path]:
    """Transcode any ffmpeg-readable audio file to MP3 CBR. Returns the new
    path on success, None if ffmpeg is unavailable. Raises TranscodeError on
    failure. Replaces the source file if `delete_src` is True."""
    if not ffmpeg_available():
        logger.warning("ffmpeg not on PATH — skipping mp3 transcode")
        return None

    src = Path(src_path)
    if not src.is_file():
        raise TranscodeError(f"source missing: {src}")
    dst = src.with_suffix(".mp3")
    if dst == src:
        # Already mp3; nothing to do.
        return src

    t0 = time.perf_counter()
    src_size = src.stat().st_size
    await asyncio.to_thread(
        _transcode_to_mp3_sync, src, dst, bitrate_kbps, timeout,
    )
    dst_size = dst.stat().st_size
    elapsed = time.perf_counter() - t0
    logger.info(
        "transcoded {} -> mp3 {}k in {:.1f}s ({:.2f} -> {:.2f} MiB)",
        src.suffix.lstrip("."), bitrate_kbps, elapsed,
        src_size / 1024 / 1024, dst_size / 1024 / 1024,
    )

    if delete_src:
        try:
            src.unlink()
        except OSError:
            pass

    return dst


async def trim_long_edge_silence(
    src_path: str | Path,
    *,
    min_edge_seconds: float = 15.0,
    detect_window_seconds: float = 1.0,
    silence_db: float = -58.0,
    detect_timeout: float = 120.0,
    trim_timeout: float = 60.0,
    delete_src: bool = True,
    on_trim_started: Optional[Callable[[], Awaitable[None]]] = None,
) -> tuple[Path, float]:
    """Trim only *long* leading/trailing pure silence from a file.

    Rules:
    - Ignore short silence entirely (anything below `min_edge_seconds`).
    - Only trim edge silence (start/end), never middle gaps.
    - Keep audio untouched if ffmpeg/ffprobe is unavailable or trim fails.

    Returns `(path, seconds_removed)`.
    """
    src = Path(src_path)
    if not src.is_file() or not ffmpeg_available():
        return src, 0.0

    try:
        duration = await asyncio.to_thread(_probe_duration_sync, src, detect_timeout)
        if duration <= 0:
            return src, 0.0

        events = await asyncio.to_thread(
            _detect_silence_events_sync,
            src,
            silence_db=silence_db,
            min_detect_seconds=detect_window_seconds,
            timeout=detect_timeout,
        )
        if not events:
            return src, 0.0

        lead = 0.0
        trail = 0.0
        edge_eps = 0.25

        first_start, _, first_dur = events[0]
        if first_start <= edge_eps and first_dur >= min_edge_seconds:
            lead = first_dur

        _, last_end, last_dur = events[-1]
        if (duration - last_end) <= edge_eps and last_dur >= min_edge_seconds:
            trail = last_dur

        if lead <= 0 and trail <= 0:
            return src, 0.0

        start_at = lead
        end_at = max(start_at + 1.0, duration - trail)
        if end_at - start_at < 1.0:
            return src, 0.0

        if on_trim_started is not None:
            await on_trim_started()

        dst = await asyncio.to_thread(
            _trim_edges_sync, src, start_at=start_at, end_at=end_at, timeout=trim_timeout,
        )
        removed = lead + trail
        logger.info(
            "trimmed edge silence: lead={:.1f}s trail={:.1f}s removed={:.1f}s",
            lead, trail, removed,
        )
        if delete_src:
            try:
                src.unlink()
            except OSError:
                pass
        return dst, removed
    except TranscodeError as e:
        # Timeout/ffmpeg issues are non-fatal. Keep original file and continue.
        logger.info("edge-silence trim skipped: {}", e)
        return src, 0.0
