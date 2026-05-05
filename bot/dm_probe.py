"""Detect whether the bot can DM a given user.

Telegram doesn't expose this directly. The reliable probe is to call
`sendChatAction` for the user — it succeeds with a transient typing
indicator if the user has started the bot, returns 403/400 otherwise.

Result is cached in-memory (positives only) since DM permission rarely
flips back to "no" once granted.
"""

from __future__ import annotations

import logging
from typing import Optional

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramAPIError,
)

log = logging.getLogger(__name__)


class DMProbe:
    def __init__(self) -> None:
        # user_id -> True (we know they have an open DM with the bot)
        self._known_open: set[int] = set()

    def mark_open(self, user_id: int) -> None:
        """Called when we observe the user actually messaging the bot
        (e.g. /start). Skips probe next time."""
        self._known_open.add(user_id)

    def drop_open(self, user_id: Optional[int]) -> None:
        """Forget that this user has an open DM. Called when an actual
        delivery attempt produced TelegramForbidden — the cached "open"
        state was stale (user blocked the bot, or `mark_open` fired without
        a real DM ever existing)."""
        if user_id is not None:
            self._known_open.discard(user_id)

    async def can_dm(self, bot: Bot, user_id: Optional[int]) -> bool:
        if user_id is None:
            return False
        if user_id in self._known_open:
            return True
        try:
            await bot.send_chat_action(chat_id=user_id, action="typing")
        except TelegramForbiddenError:
            return False
        except TelegramBadRequest as e:
            # "chat not found" / "bot can't initiate conversation with a user"
            log.debug("dm probe denied for %s: %s", user_id, e)
            return False
        except TelegramAPIError as e:
            log.warning("dm probe error for %s: %s", user_id, e)
            return False
        self._known_open.add(user_id)
        return True
