"""Library-specific exceptions."""


class SpotifyDownloaderError(Exception):
    """Base exception for all errors raised by this library."""


class AuthError(SpotifyDownloaderError):
    """Failure obtaining or using a Spotify access token."""


class TokenExpiredError(SpotifyDownloaderError):
    """Spotify endpoint returned 401 — the cached bearer token went stale.
    Caller should drop the token and retry with a freshly minted one."""


class HandshakeError(SpotifyDownloaderError):
    """Failure during the librespot Diffie-Hellman / Shannon handshake."""


class LoginError(SpotifyDownloaderError):
    """The librespot AP server rejected our login (bad/expired token, etc.)."""


class MercuryError(SpotifyDownloaderError):
    """Mercury request returned a non-2xx status."""


class TrackUnavailableError(SpotifyDownloaderError):
    """No usable audio file in the track's metadata (region-locked, removed)."""


class StorageResolveError(SpotifyDownloaderError):
    """storage-resolve endpoint did not return a usable CDN URL."""


class DownloadError(SpotifyDownloaderError):
    """Failure downloading the encrypted audio bytes from the CDN."""
