"""Tiny JSON-backed disk cache with TTL + stale-fallback.

Shared by `providers/spotify/_internal/auth.py` (TOTP secrets) and
`providers/soundcloud/api.py` (client_id). Both had the same hand-rolled
pattern: load JSON, check `ts` field, return `(value, is_fresh)`; save
JSON with current timestamp on success. Pulled out so a third caller
doesn't tempt a third copy.

Stale values are returned alongside `is_fresh=False` so callers can
fall back to them when a remote refresh fails — strongly preferred
over crashing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger(__name__)


def load(
    path: Path,
    *,
    value_key: str,
    ttl_seconds: float,
    validator: Optional[Callable[[Any], bool]] = None,
) -> Tuple[Optional[Any], bool]:
    """Load `(value, is_fresh)` from a JSON cache file at `path`.

    Returns `(None, False)` when the file doesn't exist, can't be
    parsed, is missing the expected `value_key`, or fails `validator`.
    Otherwise returns the stored value plus whether its age is within
    `ttl_seconds` of now.

    Stale values are still returned (with `is_fresh=False`) so callers
    can use them as a fallback when a remote refresh fails.
    """
    try:
        if not path.exists():
            return None, False
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get(value_key)
        ts = data.get("ts", 0)
        if value is None:
            return None, False
        if validator is not None and not validator(value):
            log.warning(
                "%s cache present but failed validation; ignoring", path.name,
            )
            return None, False
        return value, (time.time() - float(ts)) <= ttl_seconds
    except (OSError, ValueError, TypeError) as e:
        log.debug("%s cache load failed (%s); ignoring", path.name, e)
        return None, False


def save(path: Path, *, value_key: str, value: Any) -> None:
    """Write `{value_key: value, "ts": time.time()}` JSON to `path`.

    Creates parent dirs if needed. Swallows all expected error shapes —
    caching is a performance opt, never a correctness requirement.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({value_key: value, "ts": time.time()}),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as e:
        log.warning("could not persist %s cache: %s", path.name, e)
