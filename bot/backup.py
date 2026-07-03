"""Durable backup of the file_id cache + delivery stats to a private
Telegram channel, with auto-restore on a fresh/empty server.

Why: cache.json (every Telegram file_id the bot ever produced) and
bot_stats.json live only on the server's disk. A server switch or a wiped
volume loses them. This subsystem keeps a copy in a private channel and
pulls it back on boot, so the data follows the bot.

Design is deliberately paranoid (see the module's hardening notes):

- The pinned message is the single source of truth. A bot cannot list
  channel history via the Bot API, but it can always read getChat().
  pinned_message — so the newest backup is pinned and restore reads it.
- NEVER delete the last good backup. Old backups are pruned to the most
  recent K, and only *after* a new one is uploaded + pinned + verified.
- Local files are authoritative when healthy. Restore only replaces a
  file that is ABSENT, CORRUPT, or has catastrophically shrunk vs the
  backup — judged by PARSING, never by byte size.
- The backup task never opens the live JSON files; it snapshots them from
  memory (lock-consistent) so it can't race the delivery-path os.replace.
- Only cache.json + bot_stats.json + manifest.json are ever archived —
  never secrets (TOTP seeds, cookies) that share the data/ directory.
- Binding size ceiling is the 20 MB getFile *download* cap, not the 50 MB
  upload cap: an archive between them uploads fine but is un-restorable.
- All of it is best-effort: any failure is logged and swallowed, never
  crashing the bot.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

from aiogram import Bot
from aiogram.types import BufferedInputFile

from core import stats
from core.atomic_io import atomic_write_bytes

log = logging.getLogger(__name__)

ARCHIVE_NAME = "dlmus_backup.tar.gz"
MANIFEST_NAME = "manifest.json"
CACHE_MEMBER = "cache.json"
STATS_MEMBER = "bot_stats.json"
MAGIC = "dlmus-backup-v1"

_MiB = 1024 * 1024
# Binding ceiling is the 20 MB getFile *download* cap. Refuse well below it
# so we never publish an archive that can be uploaded but not restored.
DOWNLOAD_CAP = 20 * _MiB
HARD_BYTES = 18 * _MiB   # refuse to upload at/above this (un-restorable soon)
WARN_BYTES = 15 * _MiB   # warn loudly approaching the wall
# A cache that suddenly holds <50% of its last-known size signals corruption
# / empty-load; refuse to propagate that into the durable copy.
SHRINK_RATIO = 0.5
SHRINK_MIN_BASELINE = 25  # don't shrink-guard a tiny/new cache

# Never let these land in an archive even if the member-set assert is ever
# loosened. Second gate behind "only add the three named in-memory blobs".
_SECRET_DENYLIST = frozenset({
    "spotify_totp_secrets.json",
    "cookies.youtube.txt",
    "soundcloud_client_id.json",
    "backup_state.json",
    "config.py",
})


# ---- small helpers --------------------------------------------------------

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_channel_id(raw: Any) -> Optional[Any]:
    """Return a Bot-API chat_id (int for numeric channels, str for
    @usernames), or None. Mirrors JobRunner._normalize_channel_id and also
    coerces a numeric string to int so aiogram routes it as a chat_id."""
    s = str(raw or "").strip()
    if not s:
        return None
    if s.startswith("@"):
        return s
    if s.isdigit() and s.startswith("100") and len(s) >= 12:
        return int("-" + s)
    try:
        return int(s)
    except ValueError:
        return s


def _classify(path: Path, kind: str) -> tuple[str, int]:
    """Judge a local JSON file by PARSING, never by size.

    Returns (status, count) where status is ABSENT | CORRUPT | HEALTHY.
    An empty cache ('{}') or a zero-event stats file is HEALTHY, not corrupt.
    """
    p = Path(path)
    try:
        if not p.exists():
            return ("ABSENT", 0)
        raw = p.read_bytes()
    except OSError:
        return ("CORRUPT", 0)
    if not raw.strip():
        return ("ABSENT", 0)
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return ("CORRUPT", 0)
    if kind == "cache":
        if not isinstance(data, dict):
            return ("CORRUPT", 0)
        return ("HEALTHY", len(data))
    # stats
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        return ("CORRUPT", 0)
    return ("HEALTHY", len(data["events"]))


def _stats_event_count(stats_bytes: bytes) -> int:
    try:
        data = json.loads(stats_bytes)
        ev = data.get("events")
        return len(ev) if isinstance(ev, list) else 0
    except (ValueError, AttributeError):
        return 0


def _build_archive(members: dict[str, bytes]) -> bytes:
    """gzip tar of exactly the given named blobs. Members come only from
    in-memory snapshots — a directory is NEVER walked, so secrets can't leak.
    A denylist assert is the second gate."""
    bad = set(members) & _SECRET_DENYLIST
    assert not bad, f"refusing to archive denied files: {bad}"
    assert set(members) == {CACHE_MEMBER, STATS_MEMBER, MANIFEST_NAME}, (
        f"unexpected archive members: {set(members)}"
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0  # deterministic tar entries
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _unpack(archive: bytes) -> Optional[dict]:
    """Return {manifest, cache, stats} from a validated backup archive, or
    None if it isn't a genuine dlmus backup. Per-file sha256 is checked
    against the manifest; a mismatching file is dropped (set None)."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            names = set(tar.getnames())
            got: dict[str, Optional[bytes]] = {}
            for name in (MANIFEST_NAME, CACHE_MEMBER, STATS_MEMBER):
                if name in names:
                    m = tar.extractfile(name)
                    got[name] = m.read() if m else None
                else:
                    got[name] = None
    except (tarfile.TarError, OSError, EOFError, ValueError):
        return None
    if not got.get(MANIFEST_NAME):
        return None
    try:
        manifest = json.loads(got[MANIFEST_NAME])
    except ValueError:
        return None
    if not isinstance(manifest, dict) or manifest.get("magic") != MAGIC:
        return None
    cache_b = got.get(CACHE_MEMBER)
    stats_b = got.get(STATS_MEMBER)
    if cache_b is not None and _sha(cache_b) != manifest.get("sha_cache"):
        cache_b = None
    if stats_b is not None and _sha(stats_b) != manifest.get("sha_stats"):
        stats_b = None
    return {"manifest": manifest, "cache": cache_b, "stats": stats_b}


async def _fetch_remote(
    bot: Bot, chat_id: Any
) -> tuple[bool, Optional[tuple[int, dict]]]:
    """Fetch + unpack the pinned backup.

    Returns (reached, payload) where `reached` distinguishes 'Telegram
    unreachable' (suspend backups) from 'reached, no valid backup' (safe to
    start fresh). payload = (message_id, unpacked-dict) or None.
    """
    try:
        chat = await bot.get_chat(chat_id)
    except Exception as e:  # network / bad channel — treat as unreachable-ish
        log.warning("[backup] get_chat failed: %s", e)
        return (False, None)
    msg = getattr(chat, "pinned_message", None)
    doc = getattr(msg, "document", None) if msg else None
    if not msg or not doc:
        return (True, None)
    if getattr(doc, "file_name", None) != ARCHIVE_NAME:
        log.info("[backup] pinned message is not a dlmus backup; ignoring")
        return (True, None)
    size = getattr(doc, "file_size", None) or 0
    if size > DOWNLOAD_CAP:
        log.critical(
            "[backup] pinned backup is %.1f MB > 20 MB download cap — "
            "CANNOT restore it. Bound the cache or use a local Bot API server.",
            size / _MiB,
        )
        return (True, None)
    try:
        f = await bot.get_file(doc.file_id)
        buf = await bot.download_file(f.file_path)
        archive = buf.read() if buf else None
    except Exception as e:
        log.warning("[backup] download of pinned backup failed: %s", e)
        return (True, None)
    if not archive:
        return (True, None)
    unpacked = _unpack(archive)
    if unpacked is None:
        log.warning("[backup] pinned backup failed validation; ignoring")
        return (True, None)
    return (True, (msg.message_id, unpacked))


def _install(path: Path, data: bytes, prev_status: str) -> None:
    """Atomically write `data` to `path`. If the pre-existing file was
    CORRUPT, quarantine it first (renamed with a timestamp) for forensics —
    a HEALTHY file is just overwritten (the merged result is a superset)."""
    p = Path(path)
    if prev_status == "CORRUPT" and p.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            p.rename(p.with_name(f"{p.name}.corrupt-{stamp}"))
        except OSError as e:
            log.warning("[backup] could not quarantine %s: %s", p, e)
    atomic_write_bytes(p, data)


def _merge_cache(local: Optional[bytes], remote: bytes) -> tuple[bytes, int]:
    """Union of the local + remote file_id caches. Both are valid for the
    same bot, so a union only ever gains entries — it can never lose a
    file_id. On a key collision keep the newer cached_at. Returns
    (merged_json_bytes, entry_count)."""
    def _obj(b: Optional[bytes]) -> dict:
        try:
            d = json.loads(b) if b else {}
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}
    merged = dict(_obj(remote))
    for k, v in _obj(local).items():
        cur = merged.get(k)
        if cur is None or (
            isinstance(v, dict) and isinstance(cur, dict)
            and (v.get("cached_at") or 0) >= (cur.get("cached_at") or 0)
        ):
            merged[k] = v
    body = json.dumps(
        merged, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return body, len(merged)


def _merge_stats(local: Optional[bytes], remote: bytes) -> tuple[bytes, int]:
    """Union of two stats files: events deduped by (ts, t, u), earliest
    created_at, latest last_activity. `/stats` reads events as sets, so a
    union never distorts the reported numbers."""
    def _obj(b: Optional[bytes]) -> dict:
        try:
            d = json.loads(b) if b else {}
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}
    lo, ro = _obj(local), _obj(remote)
    seen: set = set()
    events: list = []
    for e in (list(ro.get("events") or []) + list(lo.get("events") or [])):
        if not isinstance(e, dict):
            continue
        key = (e.get("ts"), e.get("t"), e.get("u"))
        if key in seen:
            continue
        seen.add(key)
        events.append(e)
    events.sort(key=lambda e: float(e.get("ts") or 0))
    created = min(
        [x for x in (lo.get("created_at"), ro.get("created_at")) if x],
        default=None,
    )
    last = max(
        [x for x in (lo.get("last_activity"), ro.get("last_activity")) if x],
        default=None,
    )
    out = {
        "schema_version": lo.get("schema_version") or ro.get("schema_version") or 1,
        "created_at": created,
        "last_activity": last,
        "events": events,
    }
    body = json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")
    return body, len(events)


# ---- state file -----------------------------------------------------------

def _load_state(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    try:
        atomic_write_bytes(
            path, json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
        )
    except OSError as e:
        log.warning("[backup] could not persist state: %s", e)


# ---- boot-time restore ----------------------------------------------------

async def merge_on_boot(
    bot: Bot,
    *,
    channel_id: Any,
    cache_path: str,
    stats_path: str,
    me_id: int,
) -> dict:
    """Run BEFORE FileIdCache/stats are constructed. MERGE the pinned backup
    into the local files (union — can never lose an entry), so a fresh or
    switched server self-heals and a partially-populated one gets topped up.
    A corrupt local file is quarantined and replaced by the backup. Returns a
    dict the BackupManager uses to seed its state:
        {backup_ready, message_id, remote_manifest, baseline_cache}.
    """
    out: dict[str, Any] = {
        "backup_ready": False,
        "message_id": None,
        "remote_manifest": None,
        "baseline_cache": None,
    }
    chat_id = normalize_channel_id(channel_id)
    if not chat_id:
        return out

    cstat, ccount = _classify(Path(cache_path), "cache")
    sstat, scount = _classify(Path(stats_path), "stats")
    log.info(
        "[backup] boot check: cache=%s(%d) stats=%s(%d)",
        cstat, ccount, sstat, scount,
    )

    reached, payload = await _fetch_remote(bot, chat_id)
    if not reached:
        log.warning(
            "[backup] Telegram unreachable at boot; keeping local data, "
            "backups suspended until a remote read succeeds"
        )
        out["baseline_cache"] = ccount
        return out
    out["backup_ready"] = True

    if not payload:
        log.info("[backup] no remote backup yet; starting from local")
        out["baseline_cache"] = ccount
        return out

    message_id, unpacked = payload
    manifest = unpacked["manifest"]
    out["message_id"] = message_id
    out["remote_manifest"] = manifest
    m_bot = manifest.get("bot_id")

    # cache: file_ids are bot-scoped — ignore a cache from a different bot.
    remote_cache = unpacked["cache"]
    if remote_cache is not None and m_bot not in (None, me_id):
        log.warning(
            "[backup] remote cache is from bot %s (we are %s); ignoring it "
            "(those file_ids would be invalid for us)", m_bot, me_id,
        )
        remote_cache = None
    if remote_cache is not None:
        local_bytes = None
        if cstat == "HEALTHY":
            try:
                local_bytes = Path(cache_path).read_bytes()
            except OSError:
                local_bytes = None
        merged, n = _merge_cache(local_bytes, remote_cache)
        _install(Path(cache_path), merged, cstat)
        out["baseline_cache"] = n
        log.info("[backup] merged cache: local=%d + backup=%s -> %d entries",
                 ccount, manifest.get("count_cache"), n)
    else:
        out["baseline_cache"] = ccount

    # stats: bot-agnostic — always merge.
    remote_stats = unpacked["stats"]
    if remote_stats is not None:
        local_bytes = None
        if sstat == "HEALTHY":
            try:
                local_bytes = Path(stats_path).read_bytes()
            except OSError:
                local_bytes = None
        merged, n = _merge_stats(local_bytes, remote_stats)
        _install(Path(stats_path), merged, sstat)
        log.info("[backup] merged stats: local=%d + backup=%s -> %d events",
                 scount, manifest.get("count_stats"), n)
    return out


# ---- periodic manager -----------------------------------------------------

class BackupManager:
    def __init__(self, *, bot: Bot, cache: Any, cfg: Any, me_id: int, boot: dict):
        self._bot = bot
        self._cache = cache
        self._me_id = me_id
        self._chat_id = normalize_channel_id(
            getattr(cfg, "BACKUP_CHANNEL_ID", "")
            or getattr(cfg, "FORWARD_LOG_CHANNEL_ID", "")
        )
        self._interval = max(60, int(getattr(cfg, "BACKUP_INTERVAL_MINUTES", 10)) * 60)
        self._state_path = Path(
            getattr(cfg, "BACKUP_STATE_FILE", "") or "data/backup_state.json"
        )
        self._keep = max(1, int(getattr(cfg, "BACKUP_KEEP", 3)))
        # Auto-on: a resolvable channel is the only switch. validate()
        # self-disables if the bot lacks admin/Edit-Messages there.
        self._enabled = bool(self._chat_id)
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._valid = False

        self._state = _load_state(self._state_path)
        self._backup_ready = bool(boot.get("backup_ready"))
        self._baseline_cache = boot.get("baseline_cache")
        if boot.get("message_id") and not self._state.get("message_id"):
            self._state["message_id"] = boot["message_id"]
        rm = boot.get("remote_manifest") or {}
        if rm and not self._state.get("sha_cache"):
            self._state["sha_cache"] = rm.get("sha_cache")
            self._state["sha_stats"] = rm.get("sha_stats")

    # -- lifecycle --
    async def validate(self) -> bool:
        if not self._enabled:
            log.info("[backup] disabled (no backup channel configured)")
            return False
        try:
            chat = await self._bot.get_chat(self._chat_id)
        except Exception as e:
            log.error("[backup] cannot access channel %s: %s; disabling",
                      self._chat_id, e)
            return False
        if getattr(chat, "type", None) != "channel":
            log.error("[backup] BACKUP_CHANNEL_ID must be a channel (got %s); "
                      "disabling", getattr(chat, "type", None))
            return False
        if getattr(chat, "username", None):
            log.error("[backup] refusing PUBLIC channel @%s — backup holds "
                      "user data; use a private channel; disabling",
                      chat.username)
            return False
        try:
            mem = await self._bot.get_chat_member(self._chat_id, self._me_id)
        except Exception as e:
            log.error("[backup] cannot read bot membership: %s; disabling", e)
            return False
        status = getattr(mem, "status", None)
        can_post = bool(getattr(mem, "can_post_messages", False))
        can_edit = bool(getattr(mem, "can_edit_messages", False))
        if status != "administrator" or not can_post or not can_edit:
            log.error(
                "[backup] bot needs to be a channel admin with 'Post Messages' "
                "AND 'Edit Messages' rights (status=%s post=%s edit=%s); "
                "disabling backup", status, can_post, can_edit,
            )
            return False
        self._valid = True
        return True

    async def start(self) -> None:
        if not self._enabled:
            return
        if not await self.validate():
            return
        self._task = asyncio.create_task(self._loop(), name="backup_loop")
        log.info("[backup] enabled: channel=%s every %ds (keep last %d)",
                 self._chat_id, self._interval, self._keep)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._enabled and self._valid:
            try:
                await self.do_backup()  # final flush
            except Exception:
                log.debug("[backup] final backup failed", exc_info=True)

    async def _loop(self) -> None:
        while True:
            try:
                await self.do_backup()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("[backup] tick failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                raise

    # -- the tick --
    async def do_backup(self, *, force: bool = False) -> str:
        async with self._lock:
            return await self._do_backup_locked(force=force)

    async def _do_backup_locked(self, *, force: bool) -> str:
        # 1. boot gate: never overwrite a good remote before we've read it.
        if not self._backup_ready:
            reached, payload = await _fetch_remote(self._bot, self._chat_id)
            if not reached:
                log.warning("[backup] remote still unreachable; skipping tick")
                return "suspended"
            self._backup_ready = True
            if payload:
                mid, unpacked = payload
                self._state.setdefault("message_id", mid)
                self._baseline_cache = max(
                    self._baseline_cache or 0,
                    int(unpacked["manifest"].get("count_cache") or 0),
                )

        # 2. pin-health (heals a manual unpin even when data is unchanged).
        await self._ensure_pinned()

        # 3. lock-consistent in-memory snapshots (no disk reads).
        cache_bytes = await self._cache.serialize_bytes()
        stats_bytes = await stats.serialize_snapshot_bytes()
        sha_c, sha_s = _sha(cache_bytes), _sha(stats_bytes)

        # 4. diff-aware skip.
        if (not force and sha_c == self._state.get("sha_cache")
                and sha_s == self._state.get("sha_stats")):
            return "unchanged"

        cache_count = len(self._cache)
        stats_count = _stats_event_count(stats_bytes)

        # 5. shrink-guard (cache is append-only + irreplaceable).
        base = self._baseline_cache
        if (base and base >= SHRINK_MIN_BASELINE
                and cache_count < base * SHRINK_RATIO):
            log.error(
                "[backup] cache shrank %d -> %d (>50%%); REFUSING to back up "
                "(keeping the good remote copy)", base, cache_count,
            )
            return "refused-shrink"

        # 6. build archive from the three named in-memory blobs.
        manifest = {
            "magic": MAGIC,
            "bot_id": self._me_id,
            "ts": time.time(),
            "count_cache": cache_count,
            "count_stats": stats_count,
            "sha_cache": sha_c,
            "sha_stats": sha_s,
            "raw_cache": len(cache_bytes),
            "raw_stats": len(stats_bytes),
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        archive = _build_archive({
            CACHE_MEMBER: cache_bytes,
            STATS_MEMBER: stats_bytes,
            MANIFEST_NAME: manifest_bytes,
        })
        gz = len(archive)

        # 7. size-guard against the 20 MB download wall.
        if gz >= HARD_BYTES:
            log.critical(
                "[backup] archive is %.1f MB (>= %d MB) — NOT uploading; it "
                "would be un-restorable past the 20 MB download cap. Bound the "
                "cache (LRU) or run a local Bot API server.",
                gz / _MiB, HARD_BYTES // _MiB,
            )
            return "refused-size"
        if gz >= WARN_BYTES:
            log.warning("[backup] archive %.1f MB — approaching the 20 MB "
                        "download wall", gz / _MiB)

        # 8. upload.
        msg = await self._bot.send_document(
            self._chat_id,
            BufferedInputFile(archive, filename=ARCHIVE_NAME),
            caption=(f"dlmus backup\ncache {cache_count} entries\n"
                     f"stats {stats_count} events"),
            disable_notification=True,
        )

        # 9. pin + verify (the commit point). Do NOT advance state on failure.
        try:
            await self._bot.pin_chat_message(
                self._chat_id, msg.message_id, disable_notification=True
            )
            chat = await self._bot.get_chat(self._chat_id)
            pinned = getattr(chat, "pinned_message", None)
            if not pinned or pinned.message_id != msg.message_id:
                raise RuntimeError("pin verify mismatch")
        except Exception as e:
            log.error("[backup] pin/verify failed (%s); leaving previous "
                      "backup pinned, not advancing state", e)
            return "pin-failed"

        # 10. advance state atomically BEFORE any destructive delete.
        old_id = self._state.get("message_id")
        prev = list(self._state.get("prev_ids") or [])
        if old_id and old_id != msg.message_id:
            prev.append(old_id)
        self._state = {
            "message_id": msg.message_id,
            "sha_cache": sha_c, "sha_stats": sha_s,
            "count_cache": cache_count, "count_stats": stats_count,
            "prev_ids": prev, "ts": manifest["ts"],
        }
        _save_state(self._state_path, self._state)
        self._baseline_cache = max(self._baseline_cache or 0, cache_count)

        # 11. prune to keep-last-K (best-effort, only after state is durable).
        await self._prune_old()
        log.info("[backup] uploaded (cache=%d, stats=%d, %.2f MB gz)",
                 cache_count, stats_count, gz / _MiB)
        return "uploaded"

    async def _ensure_pinned(self) -> None:
        mid = self._state.get("message_id")
        if not mid:
            return
        try:
            chat = await self._bot.get_chat(self._chat_id)
            pinned = getattr(chat, "pinned_message", None)
            if pinned and pinned.message_id == mid:
                return
            await self._bot.pin_chat_message(
                self._chat_id, mid, disable_notification=True
            )
            log.info("[backup] re-pinned drifted backup message %s", mid)
        except Exception as e:
            log.debug("[backup] pin-health check failed: %s", e)

    async def _prune_old(self) -> None:
        prev = list(self._state.get("prev_ids") or [])
        keep_prev = self._keep - 1
        excess = prev[:-keep_prev] if keep_prev > 0 else prev
        remaining = prev[-keep_prev:] if keep_prev > 0 else []
        for mid in excess:
            try:
                await self._bot.delete_message(self._chat_id, mid)
            except Exception as e:
                # 48h delete window / already gone — drop it, never retry
                # forever. Restore is unaffected (uses the pinned message).
                log.debug("[backup] prune delete %s failed: %s", mid, e)
        self._state["prev_ids"] = remaining
        _save_state(self._state_path, self._state)
