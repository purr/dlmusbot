"""/start onboarding."""

from __future__ import annotations

from typing import Any

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, Message

from providers.registry import Registry

from ..dm_probe import DMProbe
from ..status import example_inline_search_button
from ..ui import format_start_message, visible_help_providers

router = Router(name="start")


def _start_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        example_inline_search_button("drain gang"),
    ]])


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    dm_probe: DMProbe,
    registry: Registry,
    config: Any,
) -> None:
    dm_probe.mark_open(message.from_user.id)
    me = await message.bot.get_me()
    visible = visible_help_providers(registry, config)
    await message.answer(
        format_start_message(me.username or "", visible),
        reply_markup=_start_kb(),
        disable_web_page_preview=True,
    )
