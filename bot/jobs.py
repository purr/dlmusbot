"""Download job orchestrator.

UX (mirrors purr/soundcloud-aiogram exactly):
    1. handler posts the **full** placeholder message (not just text):
       caption = format_track_caption + buttons [Source · Artist] [⏳ Downloading...]
    2. job runner downloads + tags + builds Telegram thumbnail.
    3. send_audio (replying to the user's link) with the SAME caption +
       final buttons [Source · Artist] [❓ Wrong Artist/Title?].
    4. delete the placeholder so the chat shows just the audio.
    5. file_id cached for next time.

Inline mode: only reply_markup is editable; audio still gets sent to the
user's DM. Inline buttons swap from "⏳ Downloading..." to the Source/Artist
links.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
)
from aiogram.types import (
    BufferedInputFile,
    FSInputFile,
    InlineKeyboardMarkup,
    InputMediaAudio,
)
from loguru import logger

from core.audio_convert import (
    estimate_size_mb,
    target_bitrate_for_size,
    trim_long_edge_silence,
    transcode_to_mp3,
)
from core.cache import CachedAudio, FileIdCache
from core.exceptions import (
    DlmusError,
    DMNotOpenError,
    FileTooLargeError,
    ProviderError,
)
from core.models import DownloadResult, Track
from providers.base import Provider
from providers.registry import Registry

from .dm_probe import DMProbe
from .status import (
    audio_kb,
    failed_kb,
    final_failed_kb,
    inline_stage_kb,
    permission_required_kb,
    stage_kb,
)
from .tagging import embed_metadata, fetch_cover, prepare_telegram_thumbnail
from .ui import format_track_caption


# Hard cap on retries per inline message. After this many consecutive
# failures we replace the kb with `final_failed_kb` (no retry button) so
# the user gets clear feedback that further clicks won't help. Counts are
# scoped per inline_message_id and reset on a successful delivery.
MAX_INLINE_RETRIES = 2

# Internal per-attempt retry budget for the actual download step. Spotify
# CDN edges, SoundCloud client_id rotation, transient TLS hiccups — all
# usually resolve on a second or third try, so we silently re-attempt
# instead of immediately surfacing failure. Distinct from the inline
# retry counter above (that's user-driven, this is bot-driven).
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_S = 0.5
MAX_UPLOAD_ATTEMPTS = 3
UPLOAD_RETRY_DELAY_S = 1.0
UPLOAD_REQUEST_TIMEOUT_S = 180


@dataclass
class DeliveryTarget:
    """Where to send the audio + which message(s) to update.

    `status_message_id` is the chat-mode placeholder (full track-info text +
    buttons + Downloading…); we delete it after the audio lands.
    `inline_message_id` is for inline mode — only its reply_markup edits."""
    chat_id: Optional[int] = None
    reply_to_message_id: Optional[int] = None
    inline_message_id: Optional[str] = None
    status_message_id: Optional[int] = None
    user_id: Optional[int] = None
    original_spotify_url: Optional[str] = None
    request_query: Optional[str] = None
    request_source: Optional[str] = None


class JobRunner:
    def __init__(
        self,
        bot: Bot,
        cache: FileIdCache,
        registry: Registry,
        max_file_mb: int,
        bot_username: str,
        dm_probe: Optional[DMProbe] = None,
        forward_log_channel_id: Optional[str] = None,
    ):
        self._bot = bot
        self._cache = cache
        self._registry = registry
        self._max_file_mb = max_file_mb
        self._bot_username = bot_username
        self._dm_probe = dm_probe
        self._forward_log_channel_id = self._normalize_channel_id(forward_log_channel_id)
        self._http: Optional[aiohttp.ClientSession] = None
        # Inline-mode retry counter. Key = inline_message_id, value =
        # consecutive failure count. Bumped on _mark_failed, cleared on a
        # successful delivery. Once it reaches MAX_INLINE_RETRIES we swap
        # in `final_failed_kb` so the user knows further clicks won't help.
        self._inline_failures: dict[str, int] = {}

    @staticmethod
    def _normalize_channel_id(channel_id: Optional[str]) -> Optional[str]:
        raw = (channel_id or "").strip()
        if not raw:
            return None
        # Telegram channels often get pasted as "100..." (without "-100").
        # Normalize to bot API chat_id format.
        if raw.isdigit() and raw.startswith("100") and len(raw) >= 12:
            return f"-{raw}"
        return raw

    async def start(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    @property
    def bot_username(self) -> str:
        return self._bot_username

    def caption(self, track: Track, original_spotify_url: Optional[str] = None) -> str:
        return format_track_caption(
            track, self._bot_username,
            original_spotify_url=original_spotify_url,
        )

    async def run(
        self,
        provider: Provider,
        track: Track,
        target: DeliveryTarget,
    ) -> None:
        try:
            await self._do_run(provider, track, target)
        except DMNotOpenError as e:
            logger.warning(
                "[{}:{}] DM closed at delivery time: {}",
                track.provider, track.track_id, e,
            )
            await self._mark_dm_blocked(target, track)
        except ProviderError as e:
            # When a provider tags the error with a permanent-failure
            # reason (`e.reason`), retrying won't help (Go+ won't become
            # free, region locks won't lift). Skip straight to the
            # explanation popup. Otherwise treat as a transient failure
            # and fall through to the standard retry kb.
            if getattr(e, "reason", None):
                logger.warning(
                    "[{}:{}] permanent failure ({}): {}",
                    track.provider, track.track_id, e.reason, e,
                )
                await self._mark_dead(e.reason, target, track)
            else:
                logger.warning(
                    "job failed [{}:{}]: {}",
                    track.provider, track.track_id, e,
                )
                await self._mark_failed(target, track)
        except (FileTooLargeError, DlmusError) as e:
            logger.warning("job failed [{}:{}]: {}", track.provider, track.track_id, e)
            await self._mark_failed(target, track)
        except Exception:
            logger.exception("unhandled job failure [{}:{}]", track.provider, track.track_id)
            await self._mark_failed(target, track)

    # ------------------------------------------------------------------

    async def _do_run(
        self, provider: Provider, track: Track, target: DeliveryTarget
    ) -> None:
        tag = f"{track.provider}:{track.track_id}"
        logger.info(
            "<cyan>[job]</cyan> <magenta>{}</magenta> source={} user={} query={!r}",
            tag,
            target.request_source or "unknown",
            target.user_id,
            target.request_query,
        )
        cached = await self._cache.get(track.provider, track.track_id)
        if cached and cached.file_id:
            logger.info("[{}] cache hit -> file_id={}", tag, cached.file_id)
            await self._deliver_cached(track, cached, target)
            return

        # Upfront feasibility check: Telegram caps bot uploads at 50 MB.
        # If the track is so long that even compressing to the lowest
        # listenable bitrate (128 kbps) won't fit, surface the failure
        # *now* — better than letting the user wait through a 2-minute
        # download just to bonk on the cap. Any duration that fits at
        # MIN_LISTENABLE_KBPS is allowed through; we'll re-encode after
        # download if needed (see fit-to-cap block below).
        if track.duration_seconds > 0:
            min_possible_mb = estimate_size_mb(track.duration_seconds, 128)
            if min_possible_mb > self._max_file_mb:
                logger.warning(
                    "[{}] track {} is {} long — even at 128 kbps would be "
                    "{:.1f} MB, exceeds {} MB cap; failing fast",
                    tag, track.track_id, track.duration_str,
                    min_possible_mb, self._max_file_mb,
                )
                await self._mark_dead("too_long", target, track)
                return

        logger.info(
            "[{}] queued for download: {}",
            tag, track.display_title,
        )
        # Per-target stage setter — keeps the placeholder caption stable
        # and just edits the status button between Downloading → Decrypting
        # → Converting → Tagging → Uploading. Last-stage memo is kept on
        # the closure so providers can re-emit the same stage cheaply
        # without firing extra Telegram edits.
        last_stage = {"v": ""}

        async def set_stage(stage: str) -> None:
            if stage == last_stage["v"]:
                return
            last_stage["v"] = stage
            try:
                if target.inline_message_id:
                    # Inline messages get the bare status button only —
                    # no Source/Artist row. Final delivery clears the kb.
                    await self._set_inline_kb(target, inline_stage_kb(stage))
                elif target.status_message_id is not None:
                    chat_id = target.chat_id or target.user_id
                    if chat_id is not None:
                        # Chat-mode placeholder keeps the full row
                        # ([Source · Artist] above the status button).
                        await self._bot.edit_message_reply_markup(
                            chat_id=chat_id,
                            message_id=target.status_message_id,
                            reply_markup=stage_kb(track, stage),
                        )
            except TelegramBadRequest:
                # Already edited / can't edit / removed. Best-effort.
                pass

        await set_stage("downloading")
        with tempfile.TemporaryDirectory(prefix="dlmus_") as tmp:
            result = await self._download_with_retries(
                provider, track, tmp, on_stage=set_stage,
            )
            trimmed_path, removed_silence_s = await trim_long_edge_silence(
                result.file_path,
                min_edge_seconds=15.0,
                on_trim_started=lambda: set_stage("cleaning"),
            )
            if removed_silence_s > 0 and trimmed_path != Path(result.file_path):
                new_size = trimmed_path.stat().st_size
                result = result.model_copy(update={
                    "file_path": str(trimmed_path),
                    "size_bytes": new_size,
                })
                logger.info(
                    "<cyan>[audio]</cyan> [{}] removed {:.1f}s edge silence",
                    tag,
                    removed_silence_s,
                )
            else:
                logger.debug("<cyan>[audio]</cyan> [{}] no long edge silence", tag)

            size_mb = result.size_bytes / 1024 / 1024
            if size_mb > self._max_file_mb:
                # Try to squeeze the file under the cap by re-encoding
                # to a lower CBR bitrate. We only do this when the
                # target rate stays >= MIN_LISTENABLE_KBPS — below
                # that, the audio sounds bad enough that delivering
                # it isn't worth it.
                target_kbps = target_bitrate_for_size(
                    result.track.duration_seconds, self._max_file_mb,
                )
                if target_kbps == 0:
                    logger.warning(
                        "[{}] downloaded {:.1f} MiB > {} MB cap and even "
                        "low-bitrate re-encoding wouldn't fit (duration "
                        "{}); marking too_big",
                        tag, size_mb, self._max_file_mb,
                        result.track.duration_str,
                    )
                    await self._mark_dead("too_big", target, track)
                    return
                logger.info(
                    "[{}] {:.1f} MiB > {} MB cap; re-encoding at {} kbps to fit",
                    tag, size_mb, self._max_file_mb, target_kbps,
                )
                await set_stage("fitting")
                new_path = await transcode_to_mp3(
                    result.file_path, bitrate_kbps=target_kbps,
                )
                if new_path is None:
                    # ffmpeg unavailable — can't shrink, give up.
                    logger.warning(
                        "[{}] ffmpeg unavailable; can't fit {:.1f} MiB to {} MB cap",
                        tag, size_mb, self._max_file_mb,
                    )
                    await self._mark_dead("too_big", target, track)
                    return
                new_size = new_path.stat().st_size
                new_mb = new_size / 1024 / 1024
                if new_mb > self._max_file_mb:
                    logger.warning(
                        "[{}] re-encoded to {} kbps but still {:.1f} MiB > {} MB; giving up",
                        tag, target_kbps, new_mb, self._max_file_mb,
                    )
                    await self._mark_dead("too_big", target, track)
                    return
                # Swap the result to point at the shrunk file.
                result = result.model_copy(update={
                    "file_path": str(new_path),
                    "size_bytes": new_size,
                    "format_name": f"MP3_{target_kbps}_CBR",
                    "mime_type": "audio/mpeg",
                })
                size_mb = new_mb

            logger.debug(
                "[{}] embedding metadata + cover into {}",
                tag, Path(result.file_path).name,
            )
            await set_stage("tagging")
            cover_bytes: Optional[bytes] = None
            if self._http is not None:
                cover_bytes = await embed_metadata(
                    result,
                    self._http,
                    original_spotify_url=target.original_spotify_url,
                )

            logger.info(
                "[{}] uploading to Telegram ({:.2f} MiB, {})",
                tag, size_mb, result.format_name,
            )
            await set_stage("uploading")
            sent = await self._deliver_audio(
                target, result, cover_bytes,
                original_spotify_url=target.original_spotify_url,
            )
            if sent and sent.audio is not None:
                logger.info(
                    "[{}] delivered to chat={} file_id={}",
                    tag, target.chat_id or target.user_id,
                    sent.audio.file_id,
                )
                # Successful delivery — wipe any prior retry tally
                # for this inline message so a future re-trigger
                # (cache miss, etc.) starts the budget fresh.
                if target.inline_message_id is not None:
                    self._inline_failures.pop(target.inline_message_id, None)
            file_id_for_inline: Optional[str] = None
            if sent is not None and sent.audio is not None:
                file_id_for_inline = sent.audio.file_id
                await self._cache.put(
                    provider=track.provider,
                    track_id=track.track_id,
                    entry=CachedAudio(
                        file_id=sent.audio.file_id,
                        file_unique_id=sent.audio.file_unique_id,
                        title=result.track.title,
                        performer=result.track.artists_str,
                        duration=result.track.duration_seconds,
                        mime_type=result.mime_type,
                    ),
                )

        # Swap the inline article into the audio in-place (purr pattern).
        # No reply_markup -> all buttons disappear after success.
        if file_id_for_inline:
            await self._update_inline_with_audio(
                target, result.track, file_id_for_inline,
                original_spotify_url=target.original_spotify_url,
            )

    async def _download_with_retries(
        self, provider: Provider, track: Track, dest_dir: str, *,
        on_stage,
    ):
        """Run `provider.download(...)` with up to `MAX_DOWNLOAD_ATTEMPTS`
        attempts. Transient ProviderErrors (CDN flake, stale session, SC
        client_id rotation, etc.) usually clear on the next try.

        Reasoned ProviderErrors (`.reason` set, e.g. Go+ track / DRM /
        unavailable) are NOT retried — those are permanent states no
        amount of retrying fixes. They propagate immediately so the
        outer handler can show the right `final_failed:<reason>` popup.

        FileTooLargeError isn't caught either — it's deterministic."""
        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                return await provider.download(
                    track, dest_dir, on_stage=on_stage,
                )
            except FileTooLargeError:
                raise
            except ProviderError as e:
                if getattr(e, "reason", None):
                    raise
                last_err = e
                if attempt >= MAX_DOWNLOAD_ATTEMPTS:
                    break
                logger.warning(
                    "[{}:{}] download attempt {}/{} failed ({}); retrying in {:.1f}s",
                    track.provider, track.track_id, attempt,
                    MAX_DOWNLOAD_ATTEMPTS, e, DOWNLOAD_RETRY_DELAY_S,
                )
                await asyncio.sleep(DOWNLOAD_RETRY_DELAY_S)
        # All attempts exhausted — re-raise the last ProviderError so the
        # outer flow surfaces the failure-kb to the user.
        assert last_err is not None
        raise last_err

    async def _deliver_cached(
        self, track: Track, cached: CachedAudio, target: DeliveryTarget
    ) -> None:
        synthetic = _SyntheticResult(
            track=track,
            file_path="",
            format_name=cached.mime_type or "",
            size_bytes=0,
            mime_type=cached.mime_type or "audio/mpeg",
            file_id=cached.file_id,
        )
        await self._deliver_audio(
            target, synthetic, cover_bytes=None,
            original_spotify_url=target.original_spotify_url,
        )
        # Inline article → audio in-place via cached file_id (no buttons).
        await self._update_inline_with_audio(
            target, track, cached.file_id,
            original_spotify_url=target.original_spotify_url,
        )

    # ---- delivery -----------------------------------------------------

    async def _deliver_audio(
        self,
        target: DeliveryTarget,
        result: "DownloadResult | _SyntheticResult",
        cover_bytes: Optional[bytes],
        original_spotify_url: Optional[str] = None,
    ):
        chat_id = target.chat_id or target.user_id
        if chat_id is None:
            return None

        thumb_bytes = (
            await prepare_telegram_thumbnail(cover_bytes) if cover_bytes else None
        )

        if isinstance(result, _SyntheticResult) and result.file_id:
            audio_ref = result.file_id
        else:
            audio_ref = FSInputFile(result.file_path)

        thumb_ref = (
            BufferedInputFile(thumb_bytes, filename="cover.jpg")
            if thumb_bytes else None
        )

        cap = self.caption(result.track, original_spotify_url=original_spotify_url)

        try:
            sent = await self._send_audio_with_retries(
                chat_id=chat_id,
                audio=audio_ref,
                caption=cap,
                performer=result.track.artists_str,
                title=result.track.title,
                duration=result.track.duration_seconds or None,
                thumbnail=thumb_ref,
                reply_to_message_id=target.reply_to_message_id,
                reply_markup=audio_kb(result.track),
            )
        except TelegramForbiddenError as e:
            raise DMNotOpenError(
                f"can't deliver to chat {chat_id} — user has not opened a DM"
            ) from e

        # Drop the placeholder. Telegram allows a bot to delete its own
        # messages within 48h; in DMs this always works.
        if target.status_message_id is not None:
            try:
                await self._bot.delete_message(
                    chat_id=chat_id, message_id=target.status_message_id,
                )
            except TelegramBadRequest as e:
                logger.warning(
                    "couldn't delete placeholder {} in {}: {}",
                    target.status_message_id, chat_id, e,
                )
            target.status_message_id = None

        await self._forward_to_log_channel(chat_id, sent.message_id, target)

        return sent

    async def _send_audio_with_retries(self, **kwargs):
        """Retry transient Telegram network errors during upload."""
        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
            try:
                return await self._bot.send_audio(
                    parse_mode="HTML",
                    disable_notification=True,
                    request_timeout=UPLOAD_REQUEST_TIMEOUT_S,
                    **kwargs,
                )
            except (TelegramNetworkError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt >= MAX_UPLOAD_ATTEMPTS:
                    break
                logger.warning(
                    "telegram upload attempt {}/{} failed ({}); retrying in {:.1f}s",
                    attempt,
                    MAX_UPLOAD_ATTEMPTS,
                    e,
                    UPLOAD_RETRY_DELAY_S,
                )
                await asyncio.sleep(UPLOAD_RETRY_DELAY_S)
        assert last_err is not None
        raise last_err

    async def _forward_to_log_channel(
        self, source_chat_id: int, source_message_id: int, target: DeliveryTarget
    ) -> None:
        if not self._forward_log_channel_id:
            return
        try:
            fwd = await self._bot.forward_message(
                chat_id=self._forward_log_channel_id,
                from_chat_id=source_chat_id,
                message_id=source_message_id,
                disable_notification=True,
            )
            if target.user_id:
                await self._send_forward_attribution(target.user_id, fwd.message_id)
        except Exception as e:
            logger.warning(
                "forward-log copy failed to {}: {} (hint: channels usually need -100...)",
                self._forward_log_channel_id,
                e,
            )

    async def _send_forward_attribution(self, user_id: int, reply_to_message_id: int) -> None:
        if not self._forward_log_channel_id:
            return
        try:
            user = await self._bot.get_chat(user_id)
        except Exception as e:
            logger.warning("forward attribution lookup failed for {}: {}", user_id, e)
            return

        first = getattr(user, "first_name", "") or ""
        last = getattr(user, "last_name", "") or ""
        username = getattr(user, "username", None) or ""
        usernames = list(getattr(user, "active_usernames", None) or [])

        display_parts: list[str] = []
        full = " ".join(p for p in (first, last) if p).strip()
        if full:
            display_parts.append(full)
        if username:
            display_parts.append(f"@{username}")
        for u in usernames:
            handle = f"@{u}"
            if handle not in display_parts:
                display_parts.append(handle)
        display = " ".join(display_parts).strip() or "unknown"
        text = f"Requested by {display} ({user_id})"
        try:
            await self._bot.send_message(
                chat_id=self._forward_log_channel_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                disable_notification=True,
            )
        except Exception as e:
            logger.warning("forward attribution send failed for {}: {}", user_id, e)

    # ---- status helpers ----------------------------------------------

    async def _set_inline_kb(
        self, target: DeliveryTarget, kb: InlineKeyboardMarkup
    ) -> None:
        if not target.inline_message_id:
            return
        with contextlib.suppress(TelegramBadRequest):
            await self._bot.edit_message_reply_markup(
                inline_message_id=target.inline_message_id,
                reply_markup=kb,
            )

    async def _update_inline_with_audio(
        self,
        target: DeliveryTarget,
        track: Track,
        file_id: str,
        original_spotify_url: Optional[str] = None,
    ) -> None:
        """Convert the inline article into an audio message in-place via
        edit_message_media with a cached file_id. No reply_markup is passed,
        so the inline message ends up with NO buttons after success — exactly
        purr's `update_inline_message_with_audio` behaviour."""
        if not target.inline_message_id:
            return

        thumb: Optional[BufferedInputFile] = None
        if self._http is not None and track.artwork_url:
            cover_bytes = await fetch_cover(self._http, track.artwork_url)
            if cover_bytes:
                thumb_bytes = await prepare_telegram_thumbnail(cover_bytes)
                if thumb_bytes:
                    thumb = BufferedInputFile(thumb_bytes, filename="cover.jpg")

        media = InputMediaAudio(
            media=file_id,
            caption=self.caption(track, original_spotify_url=original_spotify_url),
            parse_mode="HTML",
            title=track.title,
            performer=track.artists_str,
            duration=track.duration_seconds or None,
            thumbnail=thumb,
        )
        try:
            await self._bot.edit_message_media(
                inline_message_id=target.inline_message_id,
                media=media,
            )
        except (TelegramBadRequest, TelegramNetworkError) as e:
            logger.warning(
                "inline edit_message_media failed: {} — falling back to button edit", e,
            )
            with contextlib.suppress(TelegramBadRequest):
                await self._bot.edit_message_reply_markup(
                    inline_message_id=target.inline_message_id,
                    reply_markup=None,
                )

    async def _mark_failed(self, target: DeliveryTarget, track: Track) -> None:
        # For inline targets, track consecutive failures per inline message.
        # After MAX_INLINE_RETRIES we drop the Try-Again button and show a
        # terminal-failure label — clicking won't help, so don't suggest it.
        if target.inline_message_id is not None:
            mid = target.inline_message_id
            attempts = self._inline_failures.get(mid, 0) + 1
            self._inline_failures[mid] = attempts
            if attempts >= MAX_INLINE_RETRIES:
                kb = final_failed_kb()
                self._inline_failures.pop(mid, None)
                logger.warning(
                    "[{}:{}] inline retry budget exhausted ({}/{}); marking final-fail",
                    track.provider, track.track_id, attempts, MAX_INLINE_RETRIES,
                )
            else:
                kb = failed_kb(track.provider, track.track_id)
            await self._set_inline_kb(target, kb)
            return

        kb = failed_kb(track.provider, track.track_id)
        chat_id = target.chat_id or target.user_id
        if chat_id is None:
            return
        if target.status_message_id is not None:
            with contextlib.suppress(Exception):
                await self._bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=target.status_message_id,
                    reply_markup=kb,
                )
                return
        with contextlib.suppress(Exception):
            await self._bot.send_message(
                chat_id=chat_id,
                text="❌ <b>Couldn't deliver that one</b>",
                reply_to_message_id=target.reply_to_message_id,
                reply_markup=kb,
            )

    async def _mark_dead(
        self, reason: str, target: DeliveryTarget, track: Track,
    ) -> None:
        """Permanent-failure exit: replaces the placeholder kb with
        `final_failed_kb(reason)` and clears any retry counter. The
        button stays clickable but only opens the explanation popup —
        no download retry, since we know the track can't be delivered
        in this configuration. `reason` is one of the keys under
        `final_failed:*` in `STATUS_ALERTS`."""
        kb = final_failed_kb(reason)
        if target.inline_message_id is not None:
            self._inline_failures.pop(target.inline_message_id, None)
            await self._set_inline_kb(target, kb)
            return
        chat_id = target.chat_id or target.user_id
        if chat_id is None:
            return
        if target.status_message_id is not None:
            with contextlib.suppress(Exception):
                await self._bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=target.status_message_id,
                    reply_markup=kb,
                )
                return
        with contextlib.suppress(Exception):
            await self._bot.send_message(
                chat_id=chat_id,
                text="❌ <b>Couldn't deliver that one</b>",
                reply_to_message_id=target.reply_to_message_id,
                reply_markup=kb,
            )

    async def _mark_dm_blocked(
        self, target: DeliveryTarget, track: Track,
    ) -> None:
        """User hasn't opened the DM. Show the full purr-style permission row
        ([🔒 Permission Required] [💬 Please send /start] [🔄 Try Again])
        instead of the bare Try-Again button."""
        # Forget any stale "DM is open" cache so the next attempt re-probes
        # rather than racing into the same TelegramForbidden again.
        if self._dm_probe is not None:
            self._dm_probe.drop_open(target.user_id)
        kb = permission_required_kb(
            self._bot_username, track.provider, track.track_id,
        )
        if target.inline_message_id:
            await self._set_inline_kb(target, kb)
            return
        # Chat-mode: by definition the user just messaged us, so we *can*
        # reach this chat. Prefer editing the placeholder; otherwise post
        # a new message. (Same shape as _mark_failed — only the kb differs.)
        chat_id = target.chat_id or target.user_id
        if chat_id is None:
            return
        if target.status_message_id is not None:
            with contextlib.suppress(Exception):
                await self._bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=target.status_message_id,
                    reply_markup=kb,
                )
                return
        with contextlib.suppress(Exception):
            await self._bot.send_message(
                chat_id=chat_id,
                text="🔒 <b>I can't DM you yet</b> — open a chat with me first.",
                reply_to_message_id=target.reply_to_message_id,
                reply_markup=kb,
            )


@dataclass
class _SyntheticResult:
    """Adapter so cached file_id deliveries flow through `_deliver_audio`
    using the same code path as fresh downloads."""
    track: Track
    file_path: str
    format_name: str
    size_bytes: int
    mime_type: str
    file_id: str

    @property
    def path(self) -> Path:
        return Path(self.file_path)
