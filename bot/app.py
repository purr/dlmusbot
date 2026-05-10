"""Bot factory + lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand

from core import stats as bot_stats
from core.cache import FileIdCache
from core.queue import DownloadQueue
from providers.registry import build_default_registry

from .dm_probe import DMProbe
from .handlers import ROUTERS
from .jobs import JobRunner

log = logging.getLogger(__name__)


async def _set_default_commands(bot: Bot) -> None:
    await bot.set_my_commands([
        BotCommand(command="start", description="Intro and how to use the bot"),
        BotCommand(command="stats", description="Delivery stats (rolling windows)"),
    ])


async def run(cfg: Any) -> None:
    """Build everything from `cfg` and run the bot until cancelled."""
    registry = build_default_registry(cfg)
    if not registry.all():
        raise RuntimeError(
            "no providers registered — set SP_DC and/or check logs above"
        )
    await registry.start_all()

    cache = FileIdCache(cfg.CACHE_FILE)
    stats_file = getattr(cfg, "STATS_FILE", "") or "data/bot_stats.json"
    bot_stats.configure(stats_file)
    await bot_stats.load()
    queue: DownloadQueue = DownloadQueue(concurrency=cfg.DOWNLOAD_CONCURRENCY)
    await queue.start()

    bot = Bot(
        token=cfg.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    me = await bot.get_me()
    log.info("logged in as @%s (id=%s)", me.username, me.id)

    dm_probe = DMProbe()

    job_runner = JobRunner(
        bot=bot,
        cache=cache,
        registry=registry,
        max_file_mb=cfg.MAX_FILE_MB,
        bot_username=me.username or "",
        dm_probe=dm_probe,
        forward_log_channel_id=getattr(cfg, "FORWARD_LOG_CHANNEL_ID", "") or "",
    )
    await job_runner.start()

    dp = Dispatcher()
    dp["config"] = cfg
    dp["registry"] = registry
    dp["cache"] = cache
    dp["queue"] = queue
    dp["job_runner"] = job_runner
    dp["dm_probe"] = dm_probe
    dp["bot_username"] = me.username or ""
    dp["inline_results_limit"] = int(cfg.INLINE_RESULTS)
    dp["per_provider_limit"] = int(cfg.SEARCH_PER_PROVIDER)
    dp["inline_search_providers"] = list(cfg.INLINE_SEARCH_PROVIDERS)

    for r in ROUTERS:
        dp.include_router(r)

    await _set_default_commands(bot)

    try:
        await dp.start_polling(
            bot,
            allowed_updates=[
                "message", "edited_message",
                "inline_query", "chosen_inline_result",
                "callback_query",
            ],
        )
    finally:
        await queue.stop()
        await job_runner.close()
        await bot_stats.shutdown_flush()
        await registry.close_all()
        await bot.session.close()
