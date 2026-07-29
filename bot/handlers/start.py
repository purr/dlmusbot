"""/start onboarding and its language switcher."""

from __future__ import annotations

import contextlib
from typing import Any

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from providers.registry import Registry

from ..dm_probe import DMProbe
from ..onboarding import (
    LANG_CALLBACK_PREFIX,
    LANGS,
    format_start_message,
    start_kb,
    visible_help_providers,
)

router = Router(name="start")


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    dm_probe: DMProbe,
    registry: Registry,
    config: Any,
    bot_username: str,
) -> None:
    dm_probe.mark_open(message.from_user.id)
    visible = visible_help_providers(registry, config)
    await message.answer(
        format_start_message(bot_username, visible),
        reply_markup=start_kb(),
        disable_web_page_preview=True,
    )


@router.callback_query(F.data.startswith(LANG_CALLBACK_PREFIX))
async def on_language(
    cb: CallbackQuery,
    registry: Registry,
    config: Any,
    bot_username: str,
) -> None:
    """Language button tap → re-render the /start message in that language.
    The tapped slot becomes English in the new keyboard, so the grid keeps
    its shape and switching back is one tap."""
    lang = (cb.data or "").removeprefix(LANG_CALLBACK_PREFIX)
    if lang not in LANGS or cb.message is None:
        await cb.answer()
        return
    visible = visible_help_providers(registry, config)
    # "message is not modified" when the same language is tapped twice.
    with contextlib.suppress(TelegramBadRequest):
        await cb.message.edit_text(
            format_start_message(bot_username, visible, lang),
            reply_markup=start_kb(lang),
            disable_web_page_preview=True,
        )
    await cb.answer()
