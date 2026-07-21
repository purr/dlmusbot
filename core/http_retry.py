"""Async HTTP GET with exponential backoff for 429 + 5xx.

Intended for any provider's REST layer. Honours `Retry-After` (seconds OR
HTTP-date) when present, otherwise uses exponential backoff with jitter.

Returns the aiohttp.ClientResponse from the *last* attempt — caller is
responsible for closing it (use `async with`).
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from typing import Optional

import aiohttp

from .logging_setup import logger


_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _retry_after_seconds(headers, default: float) -> float:
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return default
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    parsed = email.utils.parsedate_to_datetime(raw)
    if parsed is None:
        return default
    return max(0.0, parsed.timestamp() - time.time())


async def get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    timeout: Optional[aiohttp.ClientTimeout] = None,
) -> aiohttp.ClientResponse:
    """Retry GET on 408/425/429/5xx with exponential backoff.

    `Retry-After` is honoured but **clamped to `max_delay`** — Spotify in
    particular sometimes returns Retry-After: 86400 (24h) under sustained
    pressure, which is useless to wait out. We cap and re-attempt sooner; if
    the server is still angry we propagate the last response upstream.

    `max_attempts=1` disables retries entirely (fail-fast)."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if timeout is not None:
                r = await session.get(
                    url, params=params, headers=headers, timeout=timeout,
                )
            else:
                r = await session.get(url, params=params, headers=headers)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # ClientTimeout expiry raises asyncio.TimeoutError, not a
            # ClientError — the most common transient, so it must retry.
            last_exc = e
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(min(max_delay, base_delay * 2 ** (attempt - 1)))
            continue

        if r.status not in _RETRYABLE_STATUSES or attempt >= max_attempts:
            return r

        backoff = min(max_delay, base_delay * 2 ** (attempt - 1))
        # Cap Retry-After at max_delay — see docstring above.
        wait = min(
            max_delay,
            _retry_after_seconds(r.headers, backoff),
        ) + random.uniform(0, 0.25)
        logger.warning(
            "{} -> {} (attempt {}/{}), retrying in {:.1f}s",
            url, r.status, attempt, max_attempts, wait,
        )
        r.release()
        await asyncio.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("get_with_retry: unreachable")
