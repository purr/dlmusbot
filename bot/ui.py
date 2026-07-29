"""User-facing text — purr/soundcloud-aiogram caption format 1:1."""

from __future__ import annotations

import html
from typing import Optional

from core.models import Track


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def n_of(count: int, noun: str, plural: Optional[str] = None) -> str:
    """Count + correctly pluralized noun: "1 track" / "5 tracks".
    Pass `plural` for irregular nouns; default appends "s"."""
    if count == 1:
        return f"1 {noun}"
    return f"{count:,} {plural or noun + 's'}"


def track_one_line(t: Track) -> str:
    return f"{t.artists_str} — {t.title} ({t.duration_str})"


_PLATFORM_LABELS = {
    "spotify": "Spotify",
    "soundcloud": "SoundCloud",
    "youtube_music": "YouTube Music",
}


def _platform_label(track: Track) -> str:
    return _PLATFORM_LABELS.get((track.provider or "").strip().lower(), "Link")


# ---- The purr caption ----------------------------------------------------
# Exact format:
#   𝄞 <a href='{track_url}'>Spotify/SoundCloud/YouTube Music</a>
#   ❀ <a href='{spotify_url}'>Spotify</a>
#   ꕤ <a href='{cover_url}'>Cover</a> ♬ @{bot_username}
# `❀ Spotify` only appears when we used a Spotify URL but delivered from
# another source (i.e. Spotify→SoundCloud fallback).

def format_track_caption(
    track: Track,
    bot_username: str,
    *,
    original_spotify_url: Optional[str] = None,
) -> str:
    parts: list[str] = []
    if track.url:
        parts.append(f"𝄞 <a href='{esc(track.url)}'>{_platform_label(track)}</a>")
    if original_spotify_url:
        parts.append(f"❀ <a href='{esc(original_spotify_url)}'>Spotify</a>")
    if track.artwork_url:
        parts.append(f"ꕤ <a href='{esc(track.artwork_url)}'>Cover</a>")
    if bot_username:
        parts.append(f"♬ @{esc(bot_username)}")
    return " ".join(parts) or esc(track_one_line(track))


def format_url_caption(
    url: str, bot_username: str, provider: str = "",
) -> str:
    """Stable message body for failures where no track metadata could be
    fetched. Same shape as `format_track_caption` (link + ♬ @bot) so a
    failed reply looks identical to every other message in the chat — only
    its button differs. Built from just the URL the user sent."""
    label = _PLATFORM_LABELS.get((provider or "").strip().lower(), "Link")
    parts: list[str] = []
    if url:
        parts.append(f"𝄞 <a href='{esc(url)}'>{label}</a>")
    if bot_username:
        parts.append(f"♬ @{esc(bot_username)}")
    return " ".join(parts) or esc(url)


# ---- Generic placeholder shown when no track-info known yet -------------
# Used for album/playlist landing message before we resolve individual tracks.
PREPARING_TEXT = "🔄 Preparing your download..."
