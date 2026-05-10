"""Persistent rolling-window delivery stats.

Records one event per successfully delivered track (fresh download or
cache hit). `/stats` reports unique tracks + unique users over rolling
windows from *now*: 24h, 7d, 30d, 365d — only windows the bot has
actually existed for are shown. All-time totals always shown.

Storage: a single JSON file (`STATS_FILE`, default `data/bot_stats.json`),
atomically written via `os.replace`. Events older than 366 days are
pruned on every write so the file stays bounded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


_SCHEMA = 1
_DAY_SEC = 86400
_WEEK_SEC = 7 * _DAY_SEC
_MONTH_SEC = 30 * _DAY_SEC
_YEAR_SEC = 365 * _DAY_SEC
# Keep one extra day so the 365d window is always fully covered.
_PRUNE_AGE_SEC = 366 * _DAY_SEC
# Hard ceiling — beyond this the file gets pruned to most-recent N.
# At ~one event per track delivery, 500k covers years for any sane bot.
_MAX_EVENTS = 500_000


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, body: bytes) -> None:
    # Windows: `os.replace` over a target/tmp that the AV scanner (or a
    # prior write barely released) still has a handle on raises WinError
    # 32 ("being used by another process"). Brief backoff resolves it.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
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
                os.replace(str(tmp), str(path))
                return
            except PermissionError as exc:
                last_err = exc
                continue
        assert last_err is not None
        raise last_err
    except OSError as exc:
        logger.error("stats: atomic write failed | {}", exc)
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        raise


def _prune(events: list[dict[str, Any]], now_ts: float) -> None:
    cutoff = now_ts - _PRUNE_AGE_SEC
    kept = [e for e in events if float(e.get("ts") or 0) >= cutoff]
    if len(kept) > _MAX_EVENTS:
        kept = kept[-_MAX_EVENTS:]
    events.clear()
    events.extend(kept)


def _aggregate(
    events: list[dict[str, Any]], window_sec: float, now_ts: float,
) -> tuple[set[str], set[int]]:
    """Return (unique track keys, unique user_ids) for events in window."""
    lo = now_ts - window_sec
    tracks: set[str] = set()
    users: set[int] = set()
    for e in events:
        ts = float(e.get("ts") or 0)
        if ts < lo:
            continue
        t = e.get("t")
        if isinstance(t, str) and t:
            tracks.add(t)
        u = e.get("u")
        if isinstance(u, int) and u > 0:
            users.add(u)
    return tracks, users


@dataclass
class _Store:
    path: Path = field(default_factory=lambda: Path("data/bot_stats.json"))
    created_at: datetime = field(default_factory=_now_dt)
    last_activity: datetime = field(default_factory=_now_dt)
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _persist_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _persist_task: asyncio.Task[None] | None = None
    _persist_pending: bool = False
    _shutdown: bool = False

    def _serialize(self) -> dict[str, Any]:
        _prune(self.events, _now_ts())
        return {
            "schema_version": _SCHEMA,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "events": self.events,
        }

    async def load(self) -> None:
        if not self.path.exists():
            async with self._lock:
                self.created_at = _now_dt()
            return
        try:
            raw_text = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
            data = json.loads(raw_text)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.error("stats: failed to load {} | {}", self.path, exc)
            return
        if not isinstance(data, dict):
            return
        async with self._lock:
            ca = data.get("created_at")
            if isinstance(ca, str) and ca.strip():
                try:
                    dt = datetime.fromisoformat(ca.replace("Z", "+00:00"))
                    self.created_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    self.created_at = _now_dt()
            la = data.get("last_activity")
            if isinstance(la, str):
                try:
                    dt = datetime.fromisoformat(la.replace("Z", "+00:00"))
                    self.last_activity = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            ev_raw = data.get("events")
            if isinstance(ev_raw, list):
                self.events = [
                    e for e in ev_raw
                    if isinstance(e, dict)
                    and isinstance(e.get("ts"), (int, float))
                    and isinstance(e.get("t"), str)
                ]
            _prune(self.events, _now_ts())

    async def record(self, provider: str, track_id: str, user_id: int) -> None:
        provider = (provider or "").strip()
        track_id = (track_id or "").strip()
        if not provider or not track_id:
            return
        key = f"{provider}:{track_id}"
        ts = _now_ts()
        now = _now_dt()
        async with self._lock:
            ev: dict[str, Any] = {"ts": ts, "t": key}
            if user_id and user_id > 0:
                ev["u"] = int(user_id)
            self.events.append(ev)
            self.last_activity = now
            _prune(self.events, ts)
        self._schedule_persist()

    def _schedule_persist(self) -> None:
        if self._shutdown:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _done(t: asyncio.Task[None]) -> None:
            if self._persist_task is t:
                self._persist_task = None
            if t.cancelled():
                if self._persist_pending:
                    self._persist_pending = False
                    self._schedule_persist()
                return
            exc = t.exception()
            if exc is not None:
                logger.error("stats: persist task failed | {}", exc)
            if self._persist_pending:
                self._persist_pending = False
                self._schedule_persist()

        if self._persist_task is not None and not self._persist_task.done():
            self._persist_pending = True
            return
        self._persist_task = loop.create_task(self._persist(), name="stats_persist")
        self._persist_task.add_done_callback(_done)

    async def _persist(self) -> None:
        if self._shutdown:
            return
        async with self._persist_lock:
            async with self._lock:
                payload = self._serialize()
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            await asyncio.to_thread(_atomic_write, self.path, body)

    async def shutdown_flush(self) -> None:
        self._shutdown = True
        t = self._persist_task
        if t is not None and not t.done():
            try:
                await asyncio.wait_for(t, timeout=20.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("stats: shutdown wait timed out")
        async with self._persist_lock:
            async with self._lock:
                payload = self._serialize()
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            await asyncio.to_thread(_atomic_write, self.path, body)

    async def snapshot(self) -> tuple[
        datetime, datetime, list[dict[str, Any]],
    ]:
        async with self._lock:
            return self.created_at, self.last_activity, list(self.events)


_store = _Store()


def configure(path: str | os.PathLike[str]) -> None:
    """Point the store at a non-default file. Call before `load()`."""
    _store.path = Path(path)


async def load() -> None:
    await _store.load()


async def shutdown_flush() -> None:
    await _store.shutdown_flush()


def schedule_record(provider: str, track_id: str, user_id: int) -> None:
    """Fire-and-forget recorder safe to call from sync contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        try:
            await _store.record(provider, track_id, user_id)
        except Exception as exc:
            logger.debug("stats: record failed | {}", exc)

    asyncio.create_task(_run(), name="stats_record")


async def format_stats_message() -> str:
    created, last, events = await _store.snapshot()
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    now = _now_dt()
    now_ts = now.timestamp()
    age_sec = max(0.0, (now - created).total_seconds())

    windows: list[tuple[str, int]] = [("Last 24h", _DAY_SEC)]
    if age_sec >= _WEEK_SEC:
        windows.append(("Last 7d", _WEEK_SEC))
    if age_sec >= _MONTH_SEC:
        windows.append(("Last 30d", _MONTH_SEC))
    if age_sec >= _YEAR_SEC:
        windows.append(("Last 365d", _YEAR_SEC))

    rows: list[tuple[str, int, int]] = []
    for label, w in windows:
        tracks, users = _aggregate(events, w, now_ts)
        rows.append((label, len(tracks), len(users)))

    all_tracks: set[str] = set()
    all_users: set[int] = set()
    for e in events:
        t = e.get("t")
        if isinstance(t, str) and t:
            all_tracks.add(t)
        u = e.get("u")
        if isinstance(u, int) and u > 0:
            all_users.add(u)
    rows.append(("All-time", len(all_tracks), len(all_users)))

    label_w = max(len(r[0]) for r in rows)
    tracks_w = max(max(len(f"{r[1]:,}") for r in rows), len("tracks"))
    users_w = max(max(len(f"{r[2]:,}") for r in rows), len("users"))

    header = f"{'window':<{label_w}}  {'tracks':>{tracks_w}}  {'users':>{users_w}}"
    sep = f"{'-' * label_w}  {'-' * tracks_w}  {'-' * users_w}"
    body_lines = [
        f"{lbl:<{label_w}}  {t:>{tracks_w},}  {u:>{users_w},}"
        for lbl, t, u in rows
    ]

    table = "\n".join([header, sep, *body_lines])
    return (
        "📊 <b>Stats</b>\n"
        f"Since {created.strftime('%Y-%m-%d %H:%M')} UTC · "
        f"last {last.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"<pre>{table}</pre>"
    )
