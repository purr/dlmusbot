"""Atomic file write: fsync the temp file, then os.replace with a short
Windows-lock backoff.

Same pattern core/stats.py uses inline; factored out so the backup
subsystem (state file + restore installs) and core/cache.py share one
crash-safe writer instead of hand-rolling three copies.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path


def atomic_write_bytes(path: str | os.PathLike, body: bytes) -> None:
    """Write `body` to `path` atomically. fsync guarantees the bytes hit
    disk before the rename; the retry loop absorbs the transient
    PermissionError (WinError 32) Windows raises when an AV scanner or
    indexer momentarily holds the freshly written temp file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        last_err: OSError | None = None
        for delay in (0, 0.05, 0.15, 0.4):
            if delay:
                time.sleep(delay)
            try:
                os.replace(tmp, p)
                return
            except PermissionError as exc:
                last_err = exc
                continue
        assert last_err is not None
        raise last_err
    except OSError:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()
        raise
