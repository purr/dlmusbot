"""Button vocabulary — purr/soundcloud-aiogram 1:1.

Callback data schema (kept short — Telegram caps at 64 bytes):
    download:<provider>:<track_id>     start / retry a download
    download_status                    no-op label (Downloading...)
    permission_info                    no-op alert (Permission Required)
"""

from __future__ import annotations

from typing import Optional

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.models import Track

# Live download-pipeline stages, surfaced as the "status" button on the
# placeholder. Providers cherry-pick which ones apply (SoundCloud only
# needs "downloading" + "uploading"; Spotify also has "decrypting" and
# optionally "converting" when transcoding to MP3). Keeping them in a
# single dict avoids re-defining label/emoji at every call site.
STAGES: dict[str, tuple[str, str]] = {
    "downloading": ("⏳", "Downloading"),
    "decrypting": ("🔓", "Decrypting"),
    "converting": ("🎚", "Converting"),
    "cleaning": ("🧹", "Cleaning"),
    "fitting": ("📦", "Compressing to fit"),
    "tagging": ("🏷", "Tagging"),
    "uploading": ("📤", "Uploading"),
}

# One-line, plain-English explanation of each stage. Surfaced as an alert
# popup when the user taps the status button — gives a quick "what's it
# doing right now?" without bloating the button text. Keep these short:
# Telegram alerts truncate around ~200 chars on most clients.
#
# Keys can be plain (`final_failed`) or reason-tagged
# (`final_failed:too_long`, `final_failed:drm`, ...). The callbacks
# handler tries the full key first and falls back to the base. Add new
# reasons here without touching the dispatcher.
STATUS_ALERTS: dict[str, str] = {
    "queued": "Waiting in the download queue — your track will start soon.",
    "downloading": "Downloading the audio.",
    "decrypting": "Unscrambling the audio.",
    "converting": "Converting the audio.",
    "cleaning": "Removing long silence from the start or end.",
    "fitting": "Compressing the file to fit the upload size limit.",
    "tagging": "Adding cover art and metadata.",
    "uploading": "Sending the file.",
    "reencoded": "This file was re-encoded to fit the upload size limit.",
    "failed": "Couldn't download — tap Try Again to retry.",
    "final_failed": "Couldn't download. Try a different track or service.",
    "final_failed:too_long": ("This track is too long to upload to Telegram, sorry!"),
    "final_failed:too_big": (
        "This track is too big to send and shrinking it more would make it sound bad."
    ),
    "final_failed:goplus": (
        "This is a SoundCloud Go+ track — only paying SoundCloud "
        "subscribers can play the full song. We can only get a 30-second "
        "preview, so we can't deliver this one."
    ),
    "final_failed:unavailable": (
        "This track isn't playable in your region or has been removed."
    ),
    "final_failed:drm": (
        "This SoundCloud track is DRM-protected (encrypted streams) and "
        "we couldn't find a free copy on Spotify or YouTube Music either. "
        "Try a different upload of the same song."
    ),
}


def lookup_status_alert(stage_key: str) -> str:
    """Resolve an alert message for a `status:<key>` callback. Tries the
    full key (e.g. `final_failed:too_long`) first, then strips trailing
    `:reason` segments until a match is found. Falls back to a generic
    "Working on it..." so unknown keys never blow up the handler.

    `status:reencoded:<kbps>` is special-cased so the alert can mention
    the actual MP3 bitrate the file was shrunk to ("re-encoded to 128
    kbps MP3..."). Other dynamic params can be added similarly without
    bloating the static STATUS_ALERTS dict."""
    if stage_key.startswith("reencoded:"):
        try:
            kbps = int(stage_key.split(":", 1)[1])
            return (
                f"This file was re-encoded to {kbps} kbps to fit the "
                "upload size limit."
            )
        except (ValueError, IndexError):
            pass
    key = stage_key
    while key:
        if key in STATUS_ALERTS:
            return STATUS_ALERTS[key]
        if ":" not in key:
            break
        key = key.rsplit(":", 1)[0]
    return "Working on it..."


def stage_button(stage: str) -> InlineKeyboardButton:
    emoji, label = STAGES.get(stage, ("⏳", "Working"))
    return InlineKeyboardButton(
        text=f"{emoji} {label}...",
        callback_data=f"status:{stage}",
    )


# Per-platform "open the track on the source site" button label.
SOURCE_LABELS = {
    "spotify": "🎧 Spotify",
    "soundcloud": "🔊 SoundCloud",
    "youtube_music": "📺 YouTube Music",
}

# id3_robot is a public Telegram bot purr links to for users to fix wrong
# tagging. The `?start=dlmus` deep-link is the same trick.
ID3_ROBOT_URL = "https://t.me/id3_robot?start=dlmus"


def _safe_cb(*parts: str) -> str:
    cb = ":".join(parts)
    if len(cb.encode("utf-8")) > 64:
        cb = cb.encode("utf-8")[:64].decode("utf-8", errors="ignore")
    return cb


# ---- atomic buttons (purr literal labels) -------------------------------


def downloading_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="⏳ Downloading...",
        callback_data="status:downloading",
    )


def queue_position_button(position: int, total: int) -> InlineKeyboardButton:
    """Status button shown while a job is still waiting in a backed-up
    download queue. Surfaces "you are Nth of M" so users on a busy bot can
    see it's queued, not stalled. callback_data points at the `queued`
    alert."""
    return InlineKeyboardButton(
        text=f"⏳ In queue {position}/{total}",
        callback_data="status:queued",
    )


def try_again_button(provider: str, track_id: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="🔄 Try Again",
        callback_data=_safe_cb("download", provider, track_id),
    )


def failed_status_button() -> InlineKeyboardButton:
    """No-op label that makes the failed state visible. Inline messages
    don't let us edit text — only buttons — so a clear "❌ Download
    failed" row is how the user sees that something went wrong."""
    return InlineKeyboardButton(
        text="❌ Download failed",
        callback_data="status:failed",
    )


def final_failed_button(reason: Optional[str] = None) -> InlineKeyboardButton:
    """Permanent-failure label shown after the inline retry budget is
    exhausted, or when we know upfront a track can't be delivered (too
    long, DRM-locked, etc). When `reason` is given the callback_data
    encodes it (`status:final_failed:<reason>`) so tapping the button
    pops up a specific explanation instead of the generic one. Reasons
    live in `STATUS_ALERTS` — add new keys there, no dispatcher change
    needed."""
    cb = "status:final_failed" + (f":{reason}" if reason else "")
    text = "❌ Couldn't download" if reason else "❌ Couldn't download"
    return InlineKeyboardButton(text=text, callback_data=cb)


def artist_button(url: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text="👤 Artist", url=url)


def source_button(provider: str, url: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=SOURCE_LABELS.get(provider, "🎵 Open"),
        url=url,
    )


def wrong_metadata_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="❓ Wrong Artist/Title? Click here!",
        url=ID3_ROBOT_URL,
    )


def reencoded_warning_button(kbps: Optional[int] = None) -> InlineKeyboardButton:
    """Permanent indicator that the file was re-encoded at a lower
    bitrate to fit the upload cap. Encodes the actual target bitrate
    in the callback so the popup alert can be specific
    (`Re-encoded to 128 kbps MP3...`)."""
    if kbps and kbps > 0:
        text = f"⚠️ Re-encoded to {kbps} kbps"
        cb = f"status:reencoded:{kbps}"
    else:
        text = "⚠️ Re-encoded to fit size limit"
        cb = "status:reencoded"
    return InlineKeyboardButton(text=text, callback_data=cb)


def start_chat_button(bot_username: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="💬 Please send /start",
        url=f"https://t.me/{bot_username}?start=open_dms",
    )


def permission_required_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="🔒 Permission Required",
        callback_data="permission_info",
    )


def example_inline_search_button(seed: str = "drain gang") -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text="🔍 Click here to start searching",
        switch_inline_query_current_chat=seed,
    )


# ---- composed keyboards (purr layout) -----------------------------------


def _row_source_artist(track: Track) -> list[InlineKeyboardButton]:
    """Row 1 of every track-related kb: [🔊 Source] [👤 Artist]."""
    row: list[InlineKeyboardButton] = []
    if track.url:
        row.append(source_button(track.provider, track.url))
    primary = track.primary_artist
    if primary and primary.url:
        row.append(artist_button(primary.url))
    return row


def placeholder_kb(track: Track) -> InlineKeyboardMarkup:
    """Buttons under the initial placeholder message (text-with-info that
    purr later replaces with the audio). Row 1: source + artist. Row 2:
    the "⏳ Downloading..." status."""
    rows: list[list[InlineKeyboardButton]] = []
    r1 = _row_source_artist(track)
    if r1:
        rows.append(r1)
    rows.append([downloading_button()])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def stage_kb(track: Track, stage: str) -> InlineKeyboardMarkup:
    """Same shape as `placeholder_kb` but with a dynamic stage label
    (Downloading → Decrypting → Converting → Tagging → Uploading) so the
    user can see what's happening at any moment."""
    rows: list[list[InlineKeyboardButton]] = []
    r1 = _row_source_artist(track)
    if r1:
        rows.append(r1)
    rows.append([stage_button(stage)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def queue_kb(track: Track, position: int, total: int) -> InlineKeyboardMarkup:
    """Chat-mode placeholder kb with a live queue-position label in place
    of the plain "Downloading..." button. Same row layout as `stage_kb`."""
    rows: list[list[InlineKeyboardButton]] = []
    r1 = _row_source_artist(track)
    if r1:
        rows.append(r1)
    rows.append([queue_position_button(position, total)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def audio_kb(
    track: Track,
    *,
    reencoded: bool = False,
    reencoded_kbps: Optional[int] = None,
) -> InlineKeyboardMarkup:
    """Final buttons under the delivered audio. Row 1: source + artist.
    Row 2 (only when re-encoded): the warning indicator with bitrate.
    Row 3: the "❓ Wrong Artist/Title?" suggestion link."""
    rows: list[list[InlineKeyboardButton]] = []
    r1 = _row_source_artist(track)
    if r1:
        rows.append(r1)
    if reencoded or reencoded_kbps:
        rows.append([reencoded_warning_button(reencoded_kbps)])
    rows.append([wrong_metadata_button()])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def inline_audio_kb(
    track: Track,
    *,
    reencoded_kbps: Optional[int] = None,
) -> Optional[InlineKeyboardMarkup]:
    """Buttons attached to the delivered inline audio (after the article
    swap via edit_message_media). Inline mode stays clean — no source /
    artist links — only the permanent "re-encoded to N kbps MP3"
    indicator when applicable. Returns None for normal-fit deliveries so
    the inline message ends up button-less, matching purr's original
    behaviour."""
    if not reencoded_kbps:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[[reencoded_warning_button(reencoded_kbps)]],
    )


def failed_kb(provider: str, track_id: str) -> InlineKeyboardMarkup:
    """First-failure kb: explicit "❌ Download failed" indicator on top of
    a "🔄 Try Again" retry. Two rows so the failure state is visible even
    on inline messages where we can't edit caption text."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [failed_status_button()],
            [try_again_button(provider, track_id)],
        ]
    )


def final_failed_kb(reason: Optional[str] = None) -> InlineKeyboardMarkup:
    """Terminal-failure kb. Pass `reason` (a key under STATUS_ALERTS, e.g.
    "too_long" / "drm" / "auth") to make the button tappable with a
    detailed explanation popup. With no reason, generic give-up label."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [final_failed_button(reason)],
        ]
    )


def permission_required_kb(
    bot_username: str,
    provider: str,
    track_id: str,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [permission_required_button()],
            [start_chat_button(bot_username)],
            [try_again_button(provider, track_id)],
        ]
    )


def inline_downloading_kb(track: Optional[Track] = None) -> InlineKeyboardMarkup:
    """Inline-mode placeholder kb — single [⏳ Downloading...] button per
    purr's `download_status_button`. After a successful delivery the inline
    article is converted to an audio message via edit_message_media with NO
    reply_markup, so no done-state kb is needed."""
    return InlineKeyboardMarkup(inline_keyboard=[[downloading_button()]])


def inline_stage_kb(stage: str) -> InlineKeyboardMarkup:
    """Inline-mode dynamic stage kb. **No** Source / Artist row — inline
    messages only carry the live status button (Downloading → Decrypting
    → Converting → Tagging → Uploading), and the kb is dropped entirely
    when delivery succeeds. Permission failures swap in
    `permission_required_kb` instead."""
    return InlineKeyboardMarkup(inline_keyboard=[[stage_button(stage)]])


def inline_queue_kb(position: int, total: int) -> InlineKeyboardMarkup:
    """Inline-mode queue-position kb — bare status button, no Source/Artist
    row, mirroring `inline_stage_kb`."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[queue_position_button(position, total)]]
    )


# Alias retained for older imports.
NOOP_CALLBACK = "download_status"
downloading_kb = inline_downloading_kb
