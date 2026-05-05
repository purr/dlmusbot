"""Aiogram routers, one per concern."""

from . import callbacks, chosen, dm, inline, start

ROUTERS = (start.router, dm.router, inline.router, chosen.router, callbacks.router)
