"""ChosenInlineResult handler.

After a user picks an inline result we own only the inline_message_id —
swap its button row through the pipeline; audio gets sent to the user's DM.
"""

from __future__ import annotations

import contextlib

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import ChosenInlineResult
from core.cache import FileIdCache
from core.exceptions import DlmusError, ProviderError
from core.logging_setup import logger
from providers.registry import Registry

from ..dm_probe import DMProbe
from ..jobs import DeliveryTarget, JobRunner
from ..status import failed_kb, final_failed_kb, permission_required_kb

router = Router(name="chosen")


@router.chosen_inline_result()
async def on_chosen(
    chosen: ChosenInlineResult,
    registry: Registry,
    cache: FileIdCache,
    job_runner: JobRunner,
    queue,
    dm_probe: DMProbe,
    bot_username: str,
) -> None:
    rid = chosen.result_id or ""
    # The empty-query "Search for tracks" placeholder uses id="example1"
    # — picking it has no track to download, so silently ignore.
    if rid in {"example1"} or ":" not in rid:
        return
    try:
        provider_name, track_id = rid.split(":", 1)
    except ValueError:
        logger.warning("malformed inline result id: {!r}", rid)
        return

    provider = registry.get(provider_name)
    if provider is None:
        logger.warning("inline pick references unknown provider: {}", provider_name)
        return

    inline_msg_id = chosen.inline_message_id
    user_id = chosen.from_user.id
    logger.info(
        "<cyan>[chosen]</cyan> user={} rid={} query={!r}",
        user_id,
        rid,
        chosen.query,
    )

    cached = await cache.get(provider_name, track_id)
    if cached and not inline_msg_id:
        return  # cached audio inserted inline directly; nothing to do.

    if not await dm_probe.can_dm(chosen.bot, user_id):
        if inline_msg_id:
            with contextlib.suppress(TelegramBadRequest):
                await chosen.bot.edit_message_reply_markup(
                    inline_message_id=inline_msg_id,
                    reply_markup=permission_required_kb(bot_username, provider_name, track_id),
                )
        return

    try:
        track = await provider.get_track(track_id)
    except ProviderError as e:
        logger.warning("inline track fetch failed [{}:{}]: {}", provider_name, track_id, e)
        if inline_msg_id:
            with contextlib.suppress(TelegramBadRequest):
                await chosen.bot.edit_message_reply_markup(
                    inline_message_id=inline_msg_id,
                    reply_markup=final_failed_kb(e.reason) if e.reason else failed_kb(provider_name, track_id),
                )
        return
    except DlmusError as e:
        logger.warning("inline track fetch failed [{}:{}]: {}", provider_name, track_id, e)
        if inline_msg_id:
            with contextlib.suppress(TelegramBadRequest):
                await chosen.bot.edit_message_reply_markup(
                    inline_message_id=inline_msg_id,
                    reply_markup=failed_kb(provider_name, track_id),
                )
        return

    target = DeliveryTarget(
        user_id=user_id,
        inline_message_id=inline_msg_id,
        request_query=(chosen.query or "").strip() or None,
        request_source="chosen_inline_result",
    )
    job_runner.enqueue(queue, provider, track, target)
