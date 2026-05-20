"""Telegram flood-control middleware.

Telegram throttles bots per-chat and globally. When it does, the API
raises `TelegramRetryAfter` carrying the exact number of seconds to back
off. Unhandled, that exception propagates out of whatever bot call hit
the limit — crashing the download job mid-pipeline (a stage-button edit,
the audio upload, even the failure notice).

This session middleware sits below every bot API call: on a flood error
it sleeps the requested cooldown and retries the same request, so callers
never see `TelegramRetryAfter`. A worker simply waits the throttle out
and the queue keeps draining smoothly. A bounded retry count stops a
permanently throttled chat from pinning a worker forever.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter

log = logging.getLogger(__name__)

# Telegram's retry_after values are usually small (a few to ~30s). Cap the
# attempts so a hard-throttled chat eventually fails instead of looping.
MAX_FLOOD_RETRIES = 5
# retry_after is the *minimum* wait; a small pad avoids landing right on
# the boundary and getting throttled again immediately.
FLOOD_RETRY_PAD_S = 1.0


async def flood_control_middleware(make_request, bot: Bot, method):
    """Retry a bot API call across `TelegramRetryAfter`, waiting exactly
    the cooldown Telegram asks for each time."""
    for attempt in range(1, MAX_FLOOD_RETRIES + 1):
        try:
            return await make_request(bot, method)
        except TelegramRetryAfter as e:
            if attempt >= MAX_FLOOD_RETRIES:
                log.error(
                    "flood control on %s: gave up after %d attempts",
                    type(method).__name__, attempt,
                )
                raise
            wait = e.retry_after + FLOOD_RETRY_PAD_S
            log.warning(
                "flood control on %s: waiting %.1fs then retry (%d/%d)",
                type(method).__name__, wait, attempt, MAX_FLOOD_RETRIES,
            )
            await asyncio.sleep(wait)
    # Unreachable: the loop always returns or raises.
    raise RuntimeError("flood_control_middleware exhausted without result")
