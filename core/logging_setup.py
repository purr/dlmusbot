"""Single source of truth for project logging.

Exports a pre-configured `logger` so every module in the project just
does:

    from core.logging_setup import logger

…and gets a loguru Logger with markup parsing enabled (so inline tags
like `<cyan>[inline]</cyan>` render correctly) and a unified stdout
sink that includes coloured level + module:line context.

`setup(level=...)` MUST be called once at process start (main.py
does this) to install the sink and route stdlib logging (aiogram,
aiohttp, yt-dlp, …) through the same pipeline.

No file sink — log to stdout only. Operators wanting persistence pipe
through `tee` or their process supervisor.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys

from loguru import logger as _base


# Enable Virtual Terminal processing on Windows so loguru's ANSI escape
# sequences actually render in PowerShell / CMD. No-op on Unix-likes.
if os.name == "nt":
    try:
        import colorama
        colorama.just_fix_windows_console()
    except ImportError:
        pass


# Project-wide logger. `.opt(colors=True)` enables markup parsing inside
# log messages — without it, `<cyan>...</cyan>` tags inside the call's
# message string would be emitted as literal text. Modules import this
# rather than `loguru.logger` directly so colour markup works uniformly.
logger = _base.opt(colors=True)


# Suppress noisy stdlib loggers to WARNING so the operator sees signal.
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
        try:
            level = _base.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the caller frame outside stdlib logging so the
        # file:line in the loguru output points to the actual emitter.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        _base.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<dim><cyan>{name}</cyan>:<cyan>{line}</cyan></dim> "
    "<level>{message}</level>"
)


def setup(level: str = "INFO") -> None:
    """Install the single stdout sink + route stdlib logging through us.

    Called once at process start (main.py). Safe to call again — any
    prior sinks are removed first so we never duplicate output.
    """
    _base.remove()
    _base.add(
        sys.stdout,
        format=_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,  # don't dump local vars on tracebacks
        enqueue=False,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name, lvl in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(lvl)
