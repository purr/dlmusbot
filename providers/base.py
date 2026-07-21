"""Abstract Provider interface — every provider implements this surface.

Bot code only sees Provider instances; it never imports any concrete class.
Adding a new music source = subclass + register in providers/registry.py.

Each provider also owns its URL patterns and canonical-URL formatting, so
the URL parser stays provider-agnostic.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from core.models import Album, ArtistRef, DownloadResult, Playlist, Track


# Async callable invoked by a provider as it transitions between download
# stages. Argument is one of the keys in `bot.status.STAGES` (downloading,
# decrypting, converting, tagging, uploading). The callback updates the
# user-facing placeholder so the UI reflects what's happening live.
StageCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class URLMatch:
    """One URL recognised by a Provider's patterns."""
    kind: str  # "track" | "album" | "playlist" | "artist" | "url"
    entity_id: str


class Provider(abc.ABC):
    """A music source. Concrete implementations are stateful (cache HTTP
    sessions, tokens) — instantiate once at startup, share across handlers."""

    name: str = ""
    """Lower-snake_case provider key. Matches keys in cache + handlers."""

    label: str = ""
    """Human-readable name for status messages, buttons, etc."""

    URL_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
    """[(kind, regex)]. Group(1) of each regex must capture entity_id.
    Order matters — first match wins, so put the most specific patterns
    earlier."""

    # ---- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Called once on bot startup. Open sessions, fetch tokens, etc."""

    async def close(self) -> None:
        """Called once on bot shutdown. Release sockets."""

    async def warmup(self) -> None:
        """Optional cold-start primer. Called from a background task after
        `start_all()` returns, so the bot can begin polling immediately
        without waiting on slow auth handshakes. Implementations should
        do whatever the first user request would otherwise pay for
        (mint tokens, fetch rotating ids, open long-lived sessions).
        Errors are swallowed by the caller — never block startup."""

    # ---- url plumbing -----------------------------------------------------

    def parse_url(self, text: str) -> Optional[URLMatch]:
        """First URL_PATTERNS match in `text`, or None."""
        for kind, pat in self.URL_PATTERNS:
            m = pat.search(text or "")
            if m:
                return URLMatch(kind=kind, entity_id=m.group(1))
        return None

    def parse_all(self, text: str) -> list[URLMatch]:
        """Every distinct URL_PATTERNS match in `text`."""
        seen: set[tuple[str, str]] = set()
        out: list[URLMatch] = []
        for kind, pat in self.URL_PATTERNS:
            for m in pat.finditer(text or ""):
                eid = m.group(1)
                key = (kind, eid)
                if key in seen:
                    continue
                seen.add(key)
                out.append(URLMatch(kind=kind, entity_id=eid))
        return out

    def canonical_url(self, kind: str, entity_id: str) -> str:
        """Default: pass entity_id through unchanged. Override per provider
        to reconstruct a clean URL (no tracking params, no locale prefix)."""
        return entity_id

    def artist_url(self, artist_id: str) -> Optional[str]:
        """Optional: format an artist URL. None if provider doesn't expose."""
        return None

    # ---- search -----------------------------------------------------------

    async def search(self, query: str, limit: int = 25) -> list[Track]:
        """Free-text track search. Best-effort — empty list if not supported."""
        return []

    # ---- URL fetch --------------------------------------------------------

    @abc.abstractmethod
    async def get_track(self, entity_id: str) -> Track:
        """Fetch a single track by provider-native id."""

    async def get_album(
        self, entity_id: str, *, offset: int = 0, limit: Optional[int] = None
    ) -> Optional[Album]:
        """Fetch an album. None if provider doesn't expose albums.
        `offset`/`limit` window the returned tracks (inline pagination);
        `total_tracks` always reflects the full container size."""
        return None

    async def get_playlist(
        self, entity_id: str, *, offset: int = 0, limit: Optional[int] = None
    ) -> Optional[Playlist]:
        """Fetch a playlist. None if provider doesn't expose playlists.
        Same `offset`/`limit` windowing contract as `get_album`."""
        return None

    async def get_artist(self, entity_id: str) -> Optional[Playlist]:
        """Fetch an artist's catalog as a Playlist (their *own* uploads /
        releases, no reposts). None if provider doesn't expose artists or
        the artist has no tracks. Returns Playlist (rather than a new
        Artist type) because the bot already knows how to render
        Playlist results — title becomes "Artist Name", tracks are the
        artist's catalogue."""
        return None

    # ---- download ---------------------------------------------------------

    @abc.abstractmethod
    async def download(
        self, track: Track, dest_dir: str,
        *, on_stage: Optional[StageCallback] = None,
    ) -> DownloadResult:
        """Download `track` into `dest_dir`. Returns DownloadResult with
        local path + format metadata. Caller deletes the file after upload.

        `on_stage`, if given, is awaited each time the provider transitions
        between sub-stages of the pipeline (e.g. downloading → decrypting
        → converting). Providers may invoke a subset; SoundCloud only
        emits "downloading", Spotify also emits "decrypting" + optionally
        "converting"."""


def make_artist_ref(provider: Provider, name: str, artist_id: Optional[str]) -> ArtistRef:
    """Build an ArtistRef whose URL comes from the provider's `artist_url`.
    Saves boilerplate at every call site."""
    url = provider.artist_url(artist_id) if artist_id else None
    return ArtistRef(name=name, artist_id=artist_id, url=url)
