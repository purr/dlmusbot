"""Callback queries.

download:<provider>:<track_id>     start / retry a download (purr-style)
status:<stage>                     status-button click → alert popup
                                   describing what the bot is doing
                                   (downloading / decrypting / converting
                                   / tagging / uploading / failed /
                                   final_failed). See `STATUS_ALERTS` in
                                   bot/status.py for the message text.
download_status                    legacy no-op label (kept for any old
                                   inline messages still in transit).
permission_info                    no-op alert (Permission Required)
"""

from __future__ import annotations

import contextlib
from typing import Optional

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery
from loguru import logger

from core.cache import FileIdCache
from core.exceptions import DlmusError, ProviderError
from providers.registry import Registry

from ..dm_probe import DMProbe
from ..jobs import DeliveryTarget, JobRunner
from ..status import failed_kb, final_failed_kb, lookup_status_alert, permission_required_kb

router = Router(name="callbacks")


@router.callback_query(lambda q: q.data == "download_status")
async def on_status_label(cb: CallbackQuery) -> None:
    # Legacy no-op (kept for old inline messages still in flight before
    # the status:<stage> rollout).
    with contextlib.suppress(TelegramBadRequest):
        await cb.answer()


@router.callback_query(lambda q: q.data and q.data.startswith("status:"))
async def on_status_alert(cb: CallbackQuery) -> None:
    """Status button tap → show a popup explaining what's happening at
    that stage. Callback data shape: `status:<stage>[:reason]`. Reason is
    optional — `lookup_status_alert` falls back from the full key down
    to the bare stage name, so generic stages need no per-reason entry.
    If the callback_query is too old (Telegram only accepts answers
    within a few seconds) we suppress the resulting BadRequest."""
    raw = cb.data or ""
    key = raw.split(":", 1)[1] if ":" in raw else ""
    msg = lookup_status_alert(key)
    with contextlib.suppress(TelegramBadRequest):
        await cb.answer(msg, show_alert=True)


@router.callback_query(lambda q: q.data == "permission_info")
async def on_permission_info(cb: CallbackQuery) -> None:
    with contextlib.suppress(TelegramBadRequest):
        await cb.answer(
            "I can't message you until you start a chat with me. "
            "Tap the button below.",
            show_alert=True,
        )


@router.callback_query(lambda q: q.data and q.data.startswith("download:"))
async def on_download(
    cb: CallbackQuery,
    registry: Registry,
    cache: FileIdCache,
    job_runner: JobRunner,
    queue,
    dm_probe: DMProbe,
    bot_username: str,
) -> None:
    parts = (cb.data or "").split(":", 2)
    if len(parts) != 3:
        await cb.answer("Bad callback")
        return

    _verb, provider_name, track_id = parts
    provider = registry.get(provider_name)
    if provider is None:
        await cb.answer(f"Unknown provider: {provider_name}", show_alert=True)
        return

    target = _target_for(cb)

    if target.inline_message_id and not await dm_probe.can_dm(cb.bot, cb.from_user.id):
        with contextlib.suppress(TelegramBadRequest):
            await cb.bot.edit_message_reply_markup(
                inline_message_id=target.inline_message_id,
                reply_markup=permission_required_kb(bot_username, provider_name, track_id),
            )
        await cb.answer("Open the DM with the bot first.")
        return

    try:
        track = await provider.get_track(track_id)
    except ProviderError as e:
        logger.warning("callback track fetch failed [{}:{}]: {}", provider_name, track_id, e)
        if target.inline_message_id:
            with contextlib.suppress(TelegramBadRequest):
                await cb.bot.edit_message_reply_markup(
                    inline_message_id=target.inline_message_id,
                    reply_markup=final_failed_kb(e.reason) if e.reason else failed_kb(provider_name, track_id),
                )
        await cb.answer("Couldn't fetch that one. Try again?")
        return
    except DlmusError as e:
        logger.warning("callback track fetch failed [{}:{}]: {}", provider_name, track_id, e)
        if target.inline_message_id:
            with contextlib.suppress(TelegramBadRequest):
                await cb.bot.edit_message_reply_markup(
                    inline_message_id=target.inline_message_id,
                    reply_markup=failed_kb(provider_name, track_id),
                )
        await cb.answer("Couldn't fetch that one. Try again?")
        return

    job_runner.enqueue(queue, provider, track, target)
    await cb.answer("Download started! The track will appear shortly.")


def _target_for(cb: CallbackQuery) -> DeliveryTarget:
    if cb.inline_message_id:
        return DeliveryTarget(
            user_id=cb.from_user.id,
            inline_message_id=cb.inline_message_id,
            request_source="callback_inline",
        )
    msg = cb.message
    chat_id: Optional[int] = msg.chat.id if msg else cb.from_user.id
    return DeliveryTarget(
        chat_id=chat_id,
        user_id=cb.from_user.id,
        reply_to_message_id=msg.message_id if msg else None,
        request_source="callback_chat",
        request_query=((msg.text or msg.caption or "").strip()[:220] if msg else None),
    )
