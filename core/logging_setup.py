"""Loguru-based logging with stdlib intercept.

Routes everything (including aiogram, aiohttp, yt-dlp) through loguru so the
console output is consistently colored and formatted. Verbosity per-logger is
tuned to keep noise down without losing useful signal.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys

from loguru import logger

# Windows PowerShell / CMD don't enable ANSI escape sequences by default,
# so loguru's auto-detect emits no color. colorama's one-liner enables
# Virtual Terminal Processing on the current console so the codes that
# loguru produces actually render.
if os.name == "nt":
    try:
        import colorama
        colorama.just_fix_windows_console()
    except ImportError:
        pass


# Suppress these to WARN (they're noisy at INFO).
_NOISY_LOGGERS = {
    "aiohttp.access": "WARNING",
    "aiohttp.server": "WARNING",
    "yt_dlp": "WARNING",
    "asyncio": "WARNING",
    "urllib3": "WARNING",
}


class InterceptHandler(logging.Handler):
    """Forwards stdlib LogRecords into loguru without losing source location."""

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level -> loguru level name.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the caller frame outside of stdlib logging so file:line
        # in the loguru output points to the actual emitter (not the handler).
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<dim><cyan>{name}</cyan>:<cyan>{line}</cyan></dim> "
    "<level>{message}</level>"
)


def setup(level: str = "INFO") -> None:
    # Strip any prior loguru sinks (avoid duplicate output on re-init).
    logger.remove()
    # INFO/WARNING and below -> stdout.
    logger.add(
        sys.stdout,
        format=_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,  # don't dump local vars on tracebacks (noisy + leaks secrets)
        enqueue=False,
        filter=lambda r: r["level"].no < logger.level("ERROR").no,
    )
    # ERROR/CRITICAL -> stderr.
    logger.add(
        sys.stderr,
        format=_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,
        enqueue=False,
        filter=lambda r: r["level"].no >= logger.level("ERROR").no,
    )

    # Route stdlib logging through us.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)
