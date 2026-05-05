"""Provider-agnostic music entities.

Every provider parses its own API/protobuf into these shared shapes so the bot
layer never has to branch on provider name.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


class ArtistRef(_Frozen):
    """Lightweight artist pointer. `name` is always present; `artist_id` and
    `url` are best-effort — providers that don't expose them leave them None.
    Used by the bot to render clickable artist buttons under audio messages."""

    name: str
    artist_id: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def of(cls, name: str) -> "ArtistRef":
        return cls(name=name)


class Track(_Frozen):
    """A single playable track. The (provider, track_id) pair is unique."""

    provider: str           # "spotify" | "soundcloud" | "youtube_music"
    track_id: str           # provider-native id (base62 / numeric / video id)
    title: str
    artists: list[ArtistRef] = Field(default_factory=list)
    album: Optional[str] = None
    duration_seconds: int = 0
    artwork_url: Optional[str] = None
    url: str = ""           # canonical clean URL on the provider site
    isrc: Optional[str] = None
    extra: dict = Field(default_factory=dict)  # provider-specific scratch

    @property
    def artists_str(self) -> str:
        return ", ".join(a.name for a in self.artists) or "Unknown Artist"

    @property
    def display_title(self) -> str:
        return f"{self.artists_str} — {self.title}"

    @property
    def duration_str(self) -> str:
        """Format duration as H:MM:SS for >= 1h, M:SS otherwise.
        Avoids "90:34" for hour-long tracks (90 minutes? ambiguous)."""
        s = self.duration_seconds
        if s >= 3600:
            return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
        return f"{s // 60}:{s % 60:02d}"

    @property
    def primary_artist(self) -> Optional[ArtistRef]:
        return self.artists[0] if self.artists else None


class Album(_Frozen):
    """Container of tracks belonging to one release."""

    provider: str
    album_id: str
    title: str
    artists: list[ArtistRef] = Field(default_factory=list)
    artwork_url: Optional[str] = None
    url: str = ""
    tracks: list[Track] = Field(default_factory=list)
    # Real release length reported by the provider, even if `tracks` is
    # capped (e.g. when hydration was limited for latency). When None,
    # callers should fall back to len(tracks).
    total_tracks: Optional[int] = None


class Playlist(_Frozen):
    """User-curated container of tracks."""

    provider: str
    playlist_id: str
    title: str
    owner: Optional[str] = None
    artwork_url: Optional[str] = None
    url: str = ""
    tracks: list[Track] = Field(default_factory=list)
    # Real playlist length reported by the provider. Populated even when
    # we only hydrate a prefix of the tracks for latency reasons; UI uses
    # this for the "Found N tracks" header.
    total_tracks: Optional[int] = None


class DownloadResult(_Frozen):
    """Output of `Provider.download` — local file ready to upload."""

    track: Track
    file_path: str
    format_name: str        # "OGG_VORBIS_320", "mp3 128k", ...
    size_bytes: int
    mime_type: str          # "audio/ogg", "audio/mpeg", ...
