"""Pydantic data models for Spotify entities and download results."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


class Image(_Frozen):
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    size_label: str = "DEFAULT"


class Artist(_Frozen):
    gid_hex: str
    spotify_id: str
    name: str
    role: Optional[str] = None
    spotify_url: str


class Album(_Frozen):
    gid_hex: str
    spotify_id: str
    name: str
    label: Optional[str] = None
    release_date: Optional[str] = None
    artists: list[Artist] = Field(default_factory=list)
    images: list[Image] = Field(default_factory=list)
    spotify_url: str

    @property
    def cover_url(self) -> Optional[str]:
        if not self.images:
            return None
        # Prefer LARGE > DEFAULT > SMALL.
        order = {"LARGE": 0, "XLARGE": 0, "DEFAULT": 1, "SMALL": 2}
        return sorted(self.images, key=lambda i: order.get(i.size_label, 9))[0].url


class AudioFile(_Frozen):
    file_id_hex: str
    format_id: int
    format_name: str

    @property
    def is_lossless(self) -> bool:
        return "FLAC" in self.format_name

    @property
    def is_librespot_decryptable(self) -> bool:
        """Whether this format is delivered with the librespot AES-CTR scheme
        we can actually decrypt. MP4_*, MP4_FLAC and FLAC_FLAC use PlayPlay or
        Widevine DRM (require Spotify.dll or a CDM device), so they are out."""
        return self.format_name.startswith(("OGG_VORBIS_", "MP3_", "AAC_"))


class Track(_Frozen):
    gid_hex: str
    spotify_id: str
    name: str
    artists: list[Artist] = Field(default_factory=list)
    featured_artists: list[Artist] = Field(default_factory=list)
    album: Optional[Album] = None
    duration_ms: int = 0
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    isrc: Optional[str] = None
    popularity: Optional[int] = None
    languages: list[str] = Field(default_factory=list)
    spotify_url: str
    files: list[AudioFile] = Field(default_factory=list)

    @property
    def duration_str(self) -> str:
        s = self.duration_ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    @property
    def has_lossless(self) -> bool:
        return any(f.is_lossless for f in self.files)

    @property
    def has_decryptable_lossless(self) -> bool:
        return any(f.is_lossless and f.is_librespot_decryptable for f in self.files)

    @property
    def best_decryptable_format(self) -> Optional[str]:
        for f in self.files:
            if f.is_librespot_decryptable:
                return f.format_name
        return None

    @property
    def all_artist_names(self) -> list[str]:
        return [a.name for a in self.artists] + [a.name for a in self.featured_artists]


class DownloadResult(_Frozen):
    track: Track
    selected_format: str
    file_path: str
    output_size_bytes: int
    encrypted_size_bytes: int
    cdn_url: str
    aes_key_hex: str
