"""/stats — rolling-window delivery stats."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from core import stats

router = Router(name="stats")


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    text = await stats.format_stats_message()
    await message.answer(text, disable_web_page_preview=True)
