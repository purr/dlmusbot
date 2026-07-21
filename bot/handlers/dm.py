"""DM link handler.

Posts the FULL placeholder (purr-style):
    text   = format_track_caption(track, bot_username)
    buttons= [Source · Artist] [⏳ Downloading...]

Then the job runner sends the audio with the same caption + final buttons
[Source · Artist] [❓ Wrong Artist/Title?] and deletes the placeholder.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
from core.exceptions import (
    DlmusError,
    ProviderError,
    TrackNotFoundError,
    UnsupportedURLError,
)
from core.logging_setup import logger
from core.models import Track
from core.shortlink import resolve as resolve_short_url
from core.url_parser import ParsedURL, parse_all
from providers.registry import Registry

from ..dm_probe import DMProbe
from ..jobs import DeliveryTarget, JobRunner
from ..status import failed_kb, final_failed_kb, placeholder_kb
from ..ui import PREPARING_TEXT, format_url_caption

# Pretty labels for the album/playlist DM-rejection notice.
_KIND_LABEL = {
    "album": ("💿", "albums"),
    "playlist": ("📚", "playlists"),
}


async def _reject_collection(
    message: Message,
    parsed: ParsedURL,
    kind_label: str = "album",
) -> None:
    """DMs only deliver single tracks — albums / playlists would hammer the
    download queue with dozens of jobs at once. Politely refuse with a
    button that fills the user's chatbox with the same URL via inline
    mode (`switch_inline_query_current_chat`), so they pick individual
    tracks from the inline result list."""
    emoji, plural = _KIND_LABEL.get(kind_label, ("📁", "collections"))
    await message.reply(
        f"{emoji} <b>{plural.capitalize()} aren't supported in DM.</b>\n"
        f"Tap below to open the {kind_label} inline and pick the tracks "
        f"you want.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"🔍 Open {kind_label} inline",
                        switch_inline_query_current_chat=parsed.url,
                    ),
                ]
            ]
        ),
        disable_notification=True,
        disable_web_page_preview=True,
    )


async def _reject_artist(
    message: Message,
    parsed: ParsedURL,
    provider,
    artist_name: str | None = None,
) -> None:
    """Artist pages have no single track to download — mirror the
    album/playlist bounce: a switch-inline button that fills the user's
    chatbox. Seeded with the artist's NAME when resolvable (a name
    search finds their songs across every provider); falls back to the
    URL itself, which inline mode also expands into the artist's
    tracks."""
    seed = artist_name
    if not seed and provider is not None:
        try:
            artist = await provider.get_artist(parsed.entity_id)
            if artist is not None and artist.title:
                seed = artist.title
        except Exception as e:
            logger.warning(
                "artist name resolve failed for {} ({}): {}",
                parsed.url,
                type(e).__name__,
                e,
            )
    await message.reply(
        "👤 <b>Artist links aren't supported in DM.</b>\n"
        "Tap below to search this artist inline and pick the tracks "
        "you want.",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔍 Search this artist inline",
                        switch_inline_query_current_chat=seed or parsed.url,
                    )
                ]
            ]
        ),
        disable_notification=True,
        disable_web_page_preview=True,
    )


router = Router(name="dm")


async def _post_placeholder(
    message: Message,
    track: Track,
    job_runner: JobRunner,
    *,
    original_spotify_url: str | None = None,
) -> int:
    """Send the placeholder reply: full caption + Source/Artist/Downloading
    buttons. Returns the message_id we'll later delete."""
    sent = await message.reply(
        job_runner.caption(track, original_spotify_url=original_spotify_url),
        reply_markup=placeholder_kb(track),
        disable_notification=True,
        disable_web_page_preview=True,
    )
    return sent.message_id


async def _enqueue_track(
    provider,
    track: Track,
    message: Message,
    job_runner: JobRunner,
    queue,
    *,
    original_spotify_url: str | None = None,
    request_query: str | None = None,
) -> None:
    status_id = await _post_placeholder(
        message,
        track,
        job_runner,
        original_spotify_url=original_spotify_url,
    )
    target = DeliveryTarget(
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        reply_to_message_id=message.message_id,
        status_message_id=status_id,
        original_spotify_url=original_spotify_url,
        request_query=request_query,
        request_source="dm_link",
    )
    job_runner.enqueue(queue, provider, track, target)


async def _enqueue_url(
    parsed: ParsedURL,
    message: Message,
    registry: Registry,
    job_runner: JobRunner,
    queue,
) -> None:
    logger.info(
        "url received: provider={} kind={} id={} from user={}",
        parsed.provider,
        parsed.kind,
        parsed.entity_id,
        message.from_user.id if message.from_user else "?",
    )
    provider = registry.get(parsed.provider)
    if provider is None:
        raise UnsupportedURLError(f"no provider for {parsed.provider}")

    # Albums / playlists are inline-only. DM downloads only allowed for
    # single tracks — pasting a 35-track album would otherwise queue a
    # mass download the user almost never wants. Bounce them to inline
    # via `switch_inline_query_current_chat`.
    if parsed.kind in ("album", "playlist"):
        await _reject_collection(message, parsed, kind_label=parsed.kind)
        return

    if parsed.kind == "artist":
        await _reject_artist(message, parsed, provider)
        return

    if parsed.kind == "track":
        track = await provider.get_track(parsed.entity_id)
        logger.info(
            "track resolved: [{}:{}] {} ({})",
            parsed.provider,
            parsed.entity_id,
            track.display_title,
            track.duration_str,
        )
        await _enqueue_track(
            provider,
            track,
            message,
            job_runner,
            queue,
            request_query=parsed.url,
        )
        return

    if parsed.kind == "url" and parsed.provider in {"spotify", "soundcloud"}:
        resolved = await resolve_short_url(parsed.entity_id)
        reparsed = parse_all(resolved, registry)
        if reparsed and reparsed[0].kind != "url":
            parsed = reparsed[0]
        elif parsed.provider == "soundcloud":
            parsed = parsed.__class__(
                provider=parsed.provider,
                kind=parsed.kind,
                entity_id=resolved,
                url=resolved,
            )
        else:
            raise UnsupportedURLError(
                f"could not resolve {parsed.provider} shortlink: {parsed.entity_id}"
            )

    if parsed.kind == "url" and parsed.provider == "soundcloud":
        from providers.soundcloud.provider import (  # noqa: SLF001
            SoundCloudProvider,
            _track_from_json,
        )

        if not isinstance(provider, SoundCloudProvider):
            raise UnsupportedURLError("expected SoundCloudProvider")
        kind, data = await provider.resolve_kind(parsed.entity_id)
        if kind == "track":
            t = _track_from_json(data)
            if t is None:
                raise TrackNotFoundError("could not parse SoundCloud track")
            if (t.extra or {}).get("is_goplus"):
                await message.reply(
                    job_runner.caption(t),
                    reply_markup=final_failed_kb("goplus"),
                    disable_web_page_preview=True,
                )
                return
            await _enqueue_track(
                provider,
                t,
                message,
                job_runner,
                queue,
                request_query=parsed.url,
            )
            return
        if kind == "playlist":
            # SoundCloud lumps both albums and playlists under
            # kind="playlist"; only the `is_album` flag distinguishes
            # them. Either way DM rejects — bounce to inline.
            label = "album" if data.get("is_album") else "playlist"
            await _reject_collection(message, parsed, kind_label=label)
            return
        if kind == "user":
            # Artist profile — same bounce as the generic artist branch;
            # the resolved user object already carries the display name,
            # so seed the search button without a second API call.
            await _reject_artist(
                message, parsed, provider, artist_name=data.get("username"),
            )
            return
        raise UnsupportedURLError(f"unknown SoundCloud entity kind: {kind}")

    raise UnsupportedURLError(f"don't know how to handle {parsed.kind}")


@router.message(F.chat.type == ChatType.PRIVATE, F.text)
async def on_dm_text(
    message: Message,
    registry: Registry,
    job_runner: JobRunner,
    queue,
    dm_probe: DMProbe,
    bot_username: str,
) -> None:
    dm_probe.mark_open(message.from_user.id)

    text = message.text or ""
    urls = parse_all(text, registry)
    if not urls:
        return
    logger.info(
        "<cyan>[dm]</cyan> user={} urls={} text={!r}",
        message.from_user.id,
        len(urls),
        text.strip()[:220],
    )

    # Instant ack only when we'll actually queue more than one track —
    # single-track URLs get the rich placeholder, album/playlist URLs get
    # the inline-bounce notice (sending PREPARING_TEXT in those cases
    # would just be a duplicate confirmation message).
    track_urls = sum(1 for u in urls if u.kind == "track")
    if track_urls > 1:
        try:
            await message.reply(PREPARING_TEXT, disable_notification=True)
        except Exception as e:
            logger.error("PREPARING_TEXT reply failed ({}): {}", type(e).__name__, e)

    for parsed in urls:
        try:
            await _enqueue_url(parsed, message, registry, job_runner, queue)
        except ProviderError as e:
            logger.error(
                "dm enqueue failed [{}] (ProviderError reason={}): {}",
                parsed.url,
                getattr(e, "reason", None),
                e,
            )
            try:
                await message.reply(
                    format_url_caption(parsed.url, bot_username, parsed.provider),
                    reply_markup=final_failed_kb(e.reason)
                    if e.reason
                    else failed_kb(parsed.provider, parsed.entity_id),
                    disable_web_page_preview=True,
                )
            except Exception as re:
                logger.error(
                    "dm failure-reply send failed [{}] ({}): {}",
                    parsed.url,
                    type(re).__name__,
                    re,
                )
        except DlmusError as e:
            logger.error(
                "dm enqueue failed [{}] ({}): {}", parsed.url, type(e).__name__, e
            )
            try:
                await message.reply(
                    format_url_caption(parsed.url, bot_username, parsed.provider),
                    reply_markup=failed_kb(parsed.provider, parsed.entity_id),
                    disable_web_page_preview=True,
                )
            except Exception as re:
                logger.error(
                    "dm failure-reply send failed [{}] ({}): {}",
                    parsed.url,
                    type(re).__name__,
                    re,
                )
        except Exception:
            logger.exception("dm enqueue crashed [{}]", parsed.url)
            try:
                await message.reply(
                    format_url_caption(parsed.url, bot_username, parsed.provider),
                    reply_markup=failed_kb(parsed.provider, parsed.entity_id),
                    disable_web_page_preview=True,
                )
            except Exception as re:
                logger.error(
                    "dm crash-reply send failed [{}] ({}): {}",
                    parsed.url,
                    type(re).__name__,
                    re,
                )
