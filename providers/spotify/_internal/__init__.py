"""Spotify direct-download package.

Public surface:
    fetch_track_info(sp_dc, track) -> Track            (metadata only)
    download_track(sp_dc, track, ...) -> DownloadResult (full pipeline)

Lower-level:
    Session                 (async librespot client)
    get_access_token        (sp_dc -> access_token)
    parse_track             (Track protobuf -> Track model)
    select_best_file
    resolve_cdn_url

Models:
    Track, Album, Artist, Image, AudioFile, DownloadResult

Exceptions:
    SpotifyDownloaderError, AuthError, HandshakeError, LoginError,
    MercuryError, TrackUnavailableError, StorageResolveError, DownloadError
"""

from .auth import get_access_token
from .api import (
    DEFAULT_FORMAT_PRIORITY,
    FORMAT_NAMES,
    fetch_track,
    parse_track,
    resolve_cdn_url,
    select_best_file,
)
from .downloader import download_track, download_track_with_session, fetch_track_info
from .exceptions import (
    AuthError,
    DownloadError,
    HandshakeError,
    LoginError,
    MercuryError,
    SpotifyDownloaderError,
    StorageResolveError,
    TrackUnavailableError,
)
from .ids import base62_to_gid, gid_to_base62, parse_track_id
from .librespot import Session
from .models import Album, Artist, AudioFile, DownloadResult, Image, Track

__all__ = [
    "Album", "Artist", "AudioFile", "AuthError", "DEFAULT_FORMAT_PRIORITY",
    "DownloadError", "DownloadResult", "FORMAT_NAMES", "HandshakeError", "Image",
    "LoginError", "MercuryError", "Session", "SpotifyDownloaderError",
    "StorageResolveError", "Track", "TrackUnavailableError", "base62_to_gid",
    "download_track", "download_track_with_session", "fetch_track",
    "fetch_track_info", "get_access_token", "gid_to_base62", "parse_track",
    "parse_track_id", "resolve_cdn_url", "select_best_file",
]
