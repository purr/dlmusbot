"""Bot factory + lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand

from core import stats as bot_stats
from core.cache import FileIdCache
from core.queue import DownloadQueue
from providers.registry import build_default_registry

from . import backup as backup_mod
from .dm_probe import DMProbe
from .flood import flood_control_middleware
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
    # Block startup until every provider has minted its tokens / scraped
    # its client_id. Polling does NOT begin until this finishes, so the
    # very first inline query — including a Spotify URL paste — already
    # has a hot Spotify session and answers inside Telegram's ~10s
    # inline_query window. Trade-off: ~5-40s slower cold start, but the
    # user never sees an empty result on the first paste after restart.
    # Upper bound on warmup. A hung TCP connect to Spotify AP or the
    # SoundCloud homepage scrape could otherwise pin startup forever.
    # 60s is generous (Spotify TOTP+AP typically completes in <40s);
    # past that we accept that the first user query may still pay the
    # cold-start tax rather than block polling indefinitely.
    log.info("warming providers up before polling starts...")
    try:
        results = await asyncio.wait_for(
            registry.warmup_all(), timeout=60.0,
        )
        summary = ", ".join(
            f"{name}={state.split(':', 1)[0]}"
            for name, state in results.items()
        )
        log.info("warmup status: %s", summary)
        # If a provider warmed up dirty, surface it loudly so the
        # operator knows their first query may suffer (and what to
        # check — usually a network/auth issue).
        for name, state in results.items():
            if not state.startswith("ok"):
                log.warning("provider '%s' is NOT hot: %s", name, state)
    except asyncio.TimeoutError:
        log.warning(
            "provider warmup exceeded 60s; starting polling anyway "
            "(first query may pay cold-start latency)"
        )

    # The Bot must exist before the cache/stats are loaded: the optional
    # Telegram backup restore talks to Telegram AND may replace cache.json /
    # bot_stats.json on disk, so it has to run first.
    bot = Bot(
        token=cfg.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    # Sits below every bot API call: waits out Telegram flood limits
    # (TelegramRetryAfter) and retries, so throttling never crashes a job.
    bot.session.middleware(flood_control_middleware)
    me = await bot.get_me()
    log.info("logged in as @%s (id=%s)", me.username, me.id)

    stats_file = getattr(cfg, "STATS_FILE", "") or "data/bot_stats.json"

    # Automatic durable backup: before loading cache/stats, MERGE whatever is
    # in the backup channel into the local files, so a fresh/switched server
    # self-heals (and a partial local set gets topped up). Auto-on whenever a
    # channel exists. Fully best-effort — never blocks or crashes startup.
    backup_boot: dict = {}
    backup_channel = (
        getattr(cfg, "BACKUP_CHANNEL_ID", "")
        or getattr(cfg, "FORWARD_LOG_CHANNEL_ID", "")
    )
    if backup_channel:
        try:
            backup_boot = await backup_mod.merge_on_boot(
                bot,
                channel_id=backup_channel,
                cache_path=cfg.CACHE_FILE,
                stats_path=stats_file,
                me_id=me.id,
            )
        except Exception:
            log.exception("[backup] merge-on-boot failed; continuing")

    cache = FileIdCache(cfg.CACHE_FILE)
    bot_stats.configure(stats_file)
    await bot_stats.load()
    # Download concurrency: one worker per CPU core by default. A positive
    # DOWNLOAD_CONCURRENCY in config.py overrides the auto value.
    configured = int(getattr(cfg, "DOWNLOAD_CONCURRENCY", 0) or 0)
    if configured > 0:
        concurrency = configured
        log.info("download queue: %d workers (config override)", concurrency)
    else:
        concurrency = os.cpu_count() or 1
        log.info("download queue: %d workers (1 per cpu core)", concurrency)
    queue: DownloadQueue = DownloadQueue(concurrency=concurrency)
    await queue.start()

    dm_probe = DMProbe()

    job_runner = JobRunner(
        bot=bot,
        cache=cache,
        registry=registry,
        max_file_mb=cfg.MAX_FILE_MB,
        bot_username=me.username or "",
        dm_probe=dm_probe,
        forward_log_channel_id=getattr(cfg, "FORWARD_LOG_CHANNEL_ID", "") or "",
        max_user_queue=int(getattr(cfg, "MAX_USER_QUEUE", 10) or 10),
    )
    await job_runner.start()

    # Start the periodic backup loop (no-op unless a valid, admin-with-Edit-
    # Messages channel exists — it validates + self-disables otherwise).
    backup_manager = None
    if backup_channel:
        backup_manager = backup_mod.BackupManager(
            bot=bot, cache=cache, cfg=cfg, me_id=me.id, boot=backup_boot,
        )
        try:
            await backup_manager.start()
        except Exception:
            log.exception("[backup] failed to start backup loop")

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
        # Final backup after stats are flushed, while the Bot session is
        # still open. stop() cancels the loop then does one last upload.
        if backup_manager is not None:
            try:
                await backup_manager.stop()
            except Exception:
                log.exception("[backup] shutdown backup failed")
        await registry.close_all()
        await bot.session.close()
