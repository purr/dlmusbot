"""User-facing text — purr/soundcloud-aiogram caption format 1:1."""

from __future__ import annotations

import html
from typing import Any, Optional

from core.models import Track
from providers.registry import Registry


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def track_one_line(t: Track) -> str:
    return f"{t.artists_str} — {t.title} ({t.duration_str})"


def _platform_label(track: Track) -> str:
    provider = (track.provider or "").strip().lower()
    if provider == "spotify":
        return "Spotify"
    if provider == "soundcloud":
        return "SoundCloud"
    if provider == "youtube_music":
        return "YouTube Music"
    return "Link"


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


# ---- Generic placeholder shown when no track-info known yet -------------
# Used for album/playlist landing message before we resolve individual tracks.
PREPARING_TEXT = "🔄 Preparing your download..."


# ---- Onboarding (/start + empty inline article, shared HTML) ------------

# Order for taglines and "X, Y, and Z" lists.
_ONBOARDING_ORDER: tuple[str, ...] = ("spotify", "soundcloud", "youtube_music")
_SERVICE_LABEL: dict[str, str] = {
    "spotify": "Spotify",
    "soundcloud": "SoundCloud",
    "youtube_music": "YouTube Music",
}


def visible_help_providers(registry: Registry, cfg: Any) -> frozenset[str]:
    """Which services to name in user-facing help.

    SoundCloud: no config, include if registered.
    Spotify: include if registered (requires ``SP_DC`` at build time).
    YouTube Music: include only if registered and ``YT_COOKIES_FILE`` is set
    (help text matches a "fully configured" YT setup; the provider may
    still work for public tracks without cookies).
    """
    names = set(registry.names())
    out: set[str] = set()
    if "soundcloud" in names:
        out.add("soundcloud")
    if "spotify" in names:
        out.add("spotify")
    cookies = (getattr(cfg, "YT_COOKIES_FILE", None) or "").strip()
    if "youtube_music" in names and cookies:
        out.add("youtube_music")
    return frozenset(out)


def _ordered_labels(visible: frozenset[str]) -> list[str]:
    return [_SERVICE_LABEL[k] for k in _ONBOARDING_ORDER if k in visible]


def _join_and(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _format_tagline_html(visible: frozenset[str]) -> str:
    labels = _ordered_labels(visible)
    if not labels:
        return "<i>Music</i>\n\n"
    return "<i>" + " · ".join(labels) + "</i>\n\n"


def format_onboarding_card_description(visible: frozenset[str]) -> str:
    """One-line summary for the empty inline-query result card."""
    parts: list[str] = []
    if "spotify" in visible:
        parts.append("Spotify")
    if "soundcloud" in visible:
        parts.append("SC")
    if "youtube_music" in visible:
        parts.append("YT")
    head = " · ".join(parts) if parts else "Music"
    return f"{head} · inline or DM · up to 320 kbps"


def format_onboarding_message(username: str, visible_providers: frozenset[str]) -> str:
    """Rich text for <code>/start</code> and the empty inline help article (HTML)."""
    tag = _format_tagline_html(visible_providers)
    labels = _ordered_labels(visible_providers)
    if labels:
        links_bullet = f"• {_join_and(labels)} links are supported\n"
    else:
        links_bullet = "• Links work for whichever sources you have enabled\n"
    pl_parts: list[str] = []
    if "spotify" in visible_providers:
        pl_parts.append("Spotify")
    if "soundcloud" in visible_providers:
        pl_parts.append("SoundCloud")
    pl_line = (
        "• Playlist and album URLs work for " + _join_and(pl_parts) + "\n"
        if pl_parts
        else ""
    )
    q1 = "Best quality by default; ID3 tags and embedded cover when available."
    q2 = (
        "\nSpotify audio is taken from Spotify, not cross-matched from other "
        "platforms."
        if "spotify" in visible_providers
        else ""
    )
    q3 = "\nMP3 up to <b>320 kbps</b>."
    return (
        tag
        + "<b>Inline</b>\n"
        "• Type <code>@{username}</code> with a search or URL in any chat\n"
        "• Example: <code>@{username} drain gang</code> or paste a link after the mention\n"
        "• Pick a result; it downloads automatically\n"
        f"{links_bullet}"
        f"{pl_line}"
        "\n"
        "<b>Direct</b>\n"
        "• Send a track, album, or playlist link in this chat\n\n"
        "<b>Quality</b>\n"
        f"{q1}{q2}{q3}"
    ).format(username=username)


def format_start_message(username: str, visible_providers: frozenset[str]) -> str:
    return (
        format_onboarding_message(username, visible_providers)
        + "\n\n"
        + "<i>Tip: mute this bot and archive the chat so downloads stay out of the way.</i>"
    )


def format_inline_empty_message(username: str, visible_providers: frozenset[str]) -> str:
    return format_onboarding_message(username, visible_providers)
