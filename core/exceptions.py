"""Bot- and provider-agnostic exception hierarchy."""

from __future__ import annotations


class DlmusError(Exception):
    """Base exception for all dlmus errors."""


class ProviderError(DlmusError):
    """Provider-side failure: API down, auth bad, track gone.

    Optional `reason` carries a permanent-failure key (e.g. "goplus",
    "unavailable") that maps onto an entry in `bot.status.STATUS_ALERTS`.
    When set, the bot surfaces the failure as a `final_failed_kb(reason)`
    popup instead of the retry path — retrying a Go+ track 100 times
    won't help, the user just needs the reason."""

    def __init__(self, message: str = "", *, reason: str | None = None):
        super().__init__(message)
        self.reason: str | None = reason


class TrackNotFoundError(ProviderError):
    """The requested track does not exist or is not playable for us."""


class UnsupportedURLError(DlmusError):
    """No registered provider claims this URL."""


class FileTooLargeError(DlmusError):
    """Output file exceeds the configured Telegram upload cap."""

    def __init__(self, size_mb: float, limit_mb: int):
        super().__init__(
            f"file is {size_mb:.1f} MB, exceeds the {limit_mb} MB cap"
        )
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class DMNotOpenError(DlmusError):
    """The user has not opened a DM with the bot, so audio delivery to their
    private chat is impossible. Surfaced as a `permission_required_kb` button
    row instead of the generic `failed_kb`, mirroring purr/soundcloud-aiogram."""
