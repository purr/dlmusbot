"""Provider registry — single source of truth for which providers exist.

Lazy-imported so adding a provider with a missing dep doesn't break the bot.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Provider

log = logging.getLogger(__name__)


class Registry:
    def __init__(self) -> None:
        self._by_name: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        self._by_name[provider.name] = provider

    def get(self, name: str) -> Optional[Provider]:
        return self._by_name.get(name)

    def all(self) -> list[Provider]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    async def start_all(self) -> None:
        for p in self._by_name.values():
            try:
                await p.start()
            except Exception:
                log.exception("provider %s failed to start", p.name)

    async def warmup_all(self) -> dict[str, str]:
        """Run every provider's warmup() concurrently. Returns a
        `{provider_name: "ok" | "failed: <reason>"}` map so the caller
        can surface per-provider status accurately (the previous
        "warmup complete" log lied when a single provider failed).
        Failures are swallowed per-provider so one slow handshake
        never blocks the others."""
        import asyncio

        results: dict[str, str] = {}

        async def _one(p: Provider) -> None:
            try:
                await p.warmup()
                results[p.name] = "ok"
            except Exception as e:
                results[p.name] = f"failed: {type(e).__name__}: {e}"
                log.warning("provider %s warmup failed: %s", p.name, e)

        await asyncio.gather(*(_one(p) for p in self._by_name.values()))
        return results

    async def close_all(self) -> None:
        for p in self._by_name.values():
            try:
                await p.close()
            except Exception:
                log.exception("provider %s failed to close", p.name)


def build_default_registry(cfg) -> Registry:
    """Build a Registry from the user's config module. Skips providers that
    are unconfigured / fail to import."""
    reg = Registry()

    if getattr(cfg, "SP_DC", None):
        try:
            from .spotify.provider import SpotifyProvider
            reg.register(SpotifyProvider(sp_dc=cfg.SP_DC))
        except Exception:
            log.exception("failed to load Spotify provider")
    else:
        log.warning("Spotify disabled: SP_DC not set in config")

    # SoundCloud — client_id is auto-fetched from soundcloud.com on cold
    # start and refreshed on 401. No config required.
    try:
        from .soundcloud.provider import SoundCloudProvider
        reg.register(SoundCloudProvider())
    except Exception:
        log.exception("failed to load SoundCloud provider")

    # YouTube Music — works for ~all public Music tracks via the
    # android_music player client trick, no cookies needed.
    try:
        from .youtube_music.provider import YouTubeMusicProvider
        reg.register(YouTubeMusicProvider(
            cookies_file=getattr(cfg, "YT_COOKIES_FILE", None) or None,
        ))
    except Exception:
        log.exception("failed to load YouTube Music provider")

    return reg
