"""Aiogram routers, one per concern."""

from . import callbacks, chosen, dm, inline, start, stats

ROUTERS = (
    start.router,
    stats.router,
    dm.router,
    inline.router,
    chosen.router,
    callbacks.router,
)
