"""Live chat-header activity indicator ("sending audio…").

Telegram only shows what a bot is doing while a chat action is *fresh*:
`sendChatAction` sets the status "for 5 seconds or less (when a message
arrives from your bot, Telegram clients clear its typing status)". A
single call at job start is long gone by the time a 40-second Spotify
download finishes, so the action has to be re-sent on a timer — and the
action itself has to change as the pipeline moves.

`ChatActionHub` owns that timer. It runs ONE refresh loop per chat and
hands out refcounted handles, so ten links pasted into one DM cost one
`sendChatAction` per `REFRESH_S` — not ten (the bot would otherwise
throttle itself on a batch paste and burn the flood middleware's retry
budget on a cosmetic call). Every live handle carries its own stage; the
loop sends the highest-priority action among them, so a chat with one
track uploading and another still downloading shows the upload.

Action choice — the Bot API offers exactly these: typing, upload_photo,
record_video, upload_video, record_voice, upload_voice, upload_document,
choose_sticker, find_location, record_video_note, upload_video_note.

There is no audio or music entry. `upload_audio` existed before Bot API
4.0 and was renamed to `upload_voice`, which clients render as "sending
voice message" — a lie about a music track, and a worse one than the
generic file status, so the whole delivery stays on `upload_document`:

    resolving, queued        typing           next thing sent is text
    downloading … uploading  upload_document  a file is on its way

The client-side wording belongs to Telegram; a bot cannot supply its own
text, so "sending audio…" is not reachable from this API at all.
Changing the mapping is a one-line edit in `STAGE_ACTIONS`.

Not `aiogram.utils.chat_action.ChatActionSender`: it repeats one fixed
action for the length of a `with` block and has no cross-job sharing. A
single download changes stage five times inside that block, and a batch
paste puts several of those blocks on the same chat at once.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
)

from core.logging_setup import logger

# Telegram clears the status after 5s — refresh just under that so the
# header never blinks between two sends.
REFRESH_S = 4.0

# A wedged request must not park the refresh loop for the session's full
# default timeout; the indicator is cosmetic and the next tick is 4s away.
SEND_TIMEOUT_S = 10

# Consecutive send failures (network blips, or a throttle that outlived
# the flood middleware's retries) after which the loop gives up on this
# chat. Nothing about the indicator is worth retrying into a wall.
MAX_CONSECUTIVE_ERRORS = 3

# Every stage that ends in a file reaching the user. Not `upload_voice`:
# it is the only audio-adjacent action left, but clients show it as
# "sending voice message", which a downloaded track is not.
DELIVERY_ACTION = ChatAction.UPLOAD_DOCUMENT
DEFAULT_ACTION = ChatAction.UPLOAD_DOCUMENT

# Stage keys are the ones `bot.status.STAGES` labels on the placeholder
# button, plus "resolving" for the metadata lookup that happens before a
# placeholder exists. Unknown stages fall back to DEFAULT_ACTION.
STAGE_ACTIONS: dict[str, str] = {
    "resolving": ChatAction.TYPING,
    "queued": ChatAction.TYPING,
    "downloading": ChatAction.UPLOAD_DOCUMENT,
    "decrypting": ChatAction.UPLOAD_DOCUMENT,
    "converting": ChatAction.UPLOAD_DOCUMENT,
    "cleaning": ChatAction.UPLOAD_DOCUMENT,
    "fitting": ChatAction.UPLOAD_DOCUMENT,
    "tagging": ChatAction.UPLOAD_DOCUMENT,
    "uploading": DELIVERY_ACTION,
}

# Which action wins when several jobs share one chat: the one closest to
# actually handing the user a file. Highest number wins.
ACTION_PRIORITY: dict[str, int] = {
    ChatAction.TYPING: 0,
    ChatAction.UPLOAD_DOCUMENT: 1,
}


def action_for_stage(stage: str) -> str:
    """Map a pipeline stage key to the Bot API action to display."""
    return STAGE_ACTIONS.get(stage, DEFAULT_ACTION)


class _ChatKeeper:
    """One refresh loop for one chat, shared by every live handle.

    Handles are keyed by an opaque int so two jobs in the same chat can
    hold different stages at the same time; the loop always sends the
    highest-priority one."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._actions: dict[int, str] = {}
        # Set whenever a handle's stage changes, so the loop re-sends the
        # new action immediately instead of at the next 4s tick.
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def chat_id(self) -> int:
        return self._chat_id

    @property
    def empty(self) -> bool:
        return not self._actions

    def add(self, key: int, action: str) -> None:
        self._actions[key] = action
        self._wake.set()
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    def update(self, key: int, action: str) -> None:
        if key not in self._actions or self._actions[key] == action:
            return
        self._actions[key] = action
        self._wake.set()

    def remove(self, key: int) -> None:
        self._actions.pop(key, None)
        self._wake.set()

    def current(self) -> Optional[str]:
        if not self._actions:
            return None
        return max(
            self._actions.values(),
            key=lambda a: ACTION_PRIORITY.get(a, 0),
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        # `gather(return_exceptions=True)` hands back the loop's own
        # CancelledError as a value instead of raising it here — a bare
        # `await task` inside a `suppress` would also swallow a
        # cancellation aimed at the *caller* (a worker being shut down).
        await asyncio.gather(task, return_exceptions=True)

    async def _loop(self) -> None:
        errors = 0
        while True:
            action = self.current()
            if action is None:
                return
            self._wake.clear()
            try:
                await self._bot.send_chat_action(
                    chat_id=self._chat_id,
                    action=action,
                    request_timeout=SEND_TIMEOUT_S,
                )
                errors = 0
            except asyncio.CancelledError:
                raise
            except (TelegramForbiddenError, TelegramBadRequest) as e:
                # No open DM (user never pressed /start, or blocked the
                # bot) or the chat is gone. That never clears mid-job,
                # and the delivery itself surfaces it properly — stop.
                logger.debug(
                    "<cyan>[chat-action]</cyan> chat {} unavailable ({}); stopping",
                    self._chat_id,
                    e,
                )
                return
            except (TelegramAPIError, OSError) as e:
                errors += 1
                logger.debug(
                    "<cyan>[chat-action]</cyan> send failed for chat {} "
                    "({}/{}): {}",
                    self._chat_id,
                    errors,
                    MAX_CONSECUTIVE_ERRORS,
                    e,
                )
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    return
            if self.current() != action:
                continue  # stage moved while sending — refresh right away
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=REFRESH_S)


class ChatActionHandle:
    """One job's claim on a chat's activity indicator.

    Usable directly (`h = hub.acquire(...)` … `await h.release()`) or as
    an async context manager. All methods are no-ops when the chat is
    unknown (inline-only targets), so callers never branch on it."""

    def __init__(
        self,
        hub: Optional["ChatActionHub"] = None,
        keeper: Optional[_ChatKeeper] = None,
        key: int = 0,
    ) -> None:
        self._hub = hub
        self._keeper = keeper
        self._key = key
        self._released = False

    def set_stage(self, stage: str) -> None:
        """Switch this handle to `stage`'s action. Cheap and idempotent —
        safe to call on every stage callback, including repeats."""
        if self._keeper is not None and not self._released:
            self._keeper.update(self._key, action_for_stage(stage))

    async def release(self) -> None:
        """Drop this claim. The chat's refresh loop stops once the last
        handle is released."""
        if self._released:
            return
        self._released = True
        if self._hub is not None and self._keeper is not None:
            await self._hub._release(self._keeper, self._key)

    async def __aenter__(self) -> "ChatActionHandle":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.release()


class ChatActionHub:
    """Per-chat `sendChatAction` refresh loops, shared and refcounted."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot
        self._keepers: dict[int, _ChatKeeper] = {}
        self._next_key = 0

    def acquire(self, chat_id: Optional[int], stage: str) -> ChatActionHandle:
        """Start (or join) the indicator for `chat_id` at `stage`. Returns
        an inert handle when `chat_id` is None — inline-mode targets with
        no DM to show anything in."""
        if chat_id is None:
            return ChatActionHandle()
        keeper = self._keepers.get(chat_id)
        if keeper is None:
            keeper = _ChatKeeper(self._bot, chat_id)
            self._keepers[chat_id] = keeper
        self._next_key += 1
        key = self._next_key
        keeper.add(key, action_for_stage(stage))
        return ChatActionHandle(self, keeper, key)

    async def _release(self, keeper: _ChatKeeper, key: int) -> None:
        keeper.remove(key)
        if not keeper.empty:
            return
        # Evict before awaiting the stop: an acquire landing during that
        # await gets a fresh keeper with a live loop, instead of joining
        # the one being torn down.
        if self._keepers.get(keeper.chat_id) is keeper:
            self._keepers.pop(keeper.chat_id, None)
        await keeper.stop()

    async def close(self) -> None:
        """Stop every loop — called from `JobRunner.close()` on shutdown."""
        keepers = list(self._keepers.values())
        self._keepers.clear()
        for keeper in keepers:
            await keeper.stop()
