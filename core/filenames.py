"""Filesystem-safe track filenames, shared by every provider.

Strips characters that are illegal on Windows/Linux and byte-caps the
stem: Linux limits a filename component to 255 **bytes**, and multi-byte
titles (Arabic, Cyrillic, kana + many artists) blew past it — every
write then failed with `[Errno 36] File name too long`. 180 bytes leaves
room for suffixes the pipeline appends (".mp4.part", ".trim.ogg", ".mp3").
"""

from __future__ import annotations

import re

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

MAX_STEM_BYTES = 180


def safe_filename(name: str, max_bytes: int = MAX_STEM_BYTES) -> str:
    cleaned = _ILLEGAL.sub("_", name).strip().rstrip(".")
    raw = cleaned.encode("utf-8")
    if len(raw) > max_bytes:
        # Cut at the byte cap, then drop any half-truncated final rune.
        cleaned = raw[:max_bytes].decode("utf-8", "ignore").strip().rstrip(".")
    return cleaned or "track"
