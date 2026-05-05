"""Persistent (provider, track_id, format) -> Telegram file_id cache.

Stored as JSON on disk so the bot survives restarts without re-downloading.
Atomic writes via os.replace; in-memory dict guarded by an asyncio.Lock so
concurrent download workers can't corrupt the file.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class CachedAudio(BaseModel):
    file_id: str
    file_unique_id: str = ""
    title: str = ""
    performer: str = ""
    duration: int = 0
    mime_type: str = ""
    reencoded: bool = False
    # Target bitrate the file was shrunk to during the fit-to-cap path.
    # Persisted so cache hits surface the same "Re-encoded to N kbps MP3"
    # warning every time, not just on the first delivery.
    reencoded_kbps: int = 0
    cached_at: int = Field(default_factory=lambda: int(time.time()))


def cache_key(provider: str, track_id: str, format_name: str = "default") -> str:
    return f"{provider}:{track_id}:{format_name}"


class FileIdCache:
    """Thread-safe (asyncio) JSON-backed file_id store."""

    def __init__(self, path: str | os.PathLike):
        self._path = Path(path)
        self._data: dict[str, CachedAudio] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    def _load_sync(self) -> None:
        if not self._path.is_file():
            self._data = {}
            self._loaded = True
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        self._data = {
            k: CachedAudio.model_validate(v) for k, v in raw.items()
            if isinstance(v, dict)
        }
        self._loaded = True

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if not self._loaded:
                self._load_sync()

    async def get(
        self, provider: str, track_id: str, format_name: str = "default"
    ) -> Optional[CachedAudio]:
        await self._ensure_loaded()
        return self._data.get(cache_key(provider, track_id, format_name))

    async def put(
        self,
        provider: str,
        track_id: str,
        entry: CachedAudio,
        format_name: str = "default",
    ) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._data[cache_key(provider, track_id, format_name)] = entry
            await asyncio.to_thread(self._flush)

    async def remove(
        self, provider: str, track_id: str, format_name: str = "default"
    ) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._data.pop(cache_key(provider, track_id, format_name), None)
            await asyncio.to_thread(self._flush)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {k: v.model_dump() for k, v in self._data.items()}
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    def __len__(self) -> int:
        return len(self._data)
