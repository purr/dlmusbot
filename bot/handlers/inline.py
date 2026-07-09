"""Inline query handler — purr/soundcloud-aiogram 1:1 result format.

Result shapes:
- single track URL  → title: "🎵 {title}",       desc: "By {artist} • {duration}"
- playlist track    → title: "{i+1}. {title}",   desc: "By {artist} • {duration}"
- free-text search  → title: "{provider_emoji} {title}",
                                                  desc: "By {artist} • {duration}"
- empty query       → "Search for tracks" article with example button

Every result uses InlineQueryResultArticle with:
- thumbnail_url    = artwork (low-res for search list, high-res in caption)
- input_message_content = the full purr caption (𝄞 Link ꕤ Cover ♬ @bot)
- reply_markup     = single [⏳ Downloading...] button
After the user picks, chosen_inline_result triggers the actual download
which delivers the audio to the user's DM.

Search is parallel across providers with a hard per-provider timeout (so a
single rate-limited or slow API doesn't stall the whole inline response).
Whatever has come back when the deadline hits is deduped and interleaved
round-robin (Spotify first, each provider's own ranking preserved) —
typically both Spotify + SoundCloud, but worst case one of them alone if
the other is throttled.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Any, Optional

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
from core.cache import FileIdCache
from core.exceptions import DlmusError, ProviderError, TrackNotFoundError
from core.fallback import FALLBACK_REASONS, find_alternative_track
from core.fuzz import (
    dedupe_embedded_title,
    dedupe_near_duplicates,
    dedupe_tracks,
    interleave_by_provider,
)
from core.logging_setup import logger
from core.models import Track
from core.shortlink import resolve as resolve_short_url
from core.url_parser import parse as parse_url
from providers.registry import Registry

from ..jobs import JobRunner
from ..status import downloading_button, example_inline_search_button
from ..ui import (
    format_inline_empty_message,
    format_onboarding_card_description,
    format_track_caption,
    visible_help_providers,
)

router = Router(name="inline")


# Catches *anything* that looks URL-shaped, including bare domains
# ("foo.com/bar") and full schemes. Used to detect typed URLs that no
# registered provider recognises so the inline UI can show "Unsupported
# URL" instead of falling through to free-text search (which would
# produce nonsensical results when the user clearly intended a link).
_URL_LIKE_RE = re.compile(
    r"""(
        https?://\S+                        |  # explicit scheme
        (?:[a-z0-9-]+\.)+[a-z]{2,}/\S+       |  # bare host + path
        spotify:[a-z]+:[A-Za-z0-9]+             # spotify URI
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_url(text: str) -> bool:
    return bool(_URL_LIKE_RE.fullmatch(text.strip()))


# Power-user shortcut: `<provider>:<track_id>` skips search entirely and
# fetches the track straight from the provider. Useful when you've copied
# an ID out of a log line or another tool and don't want to round-trip
# through a URL. Aliases mirror common shorthands; resolution always lands
# on the canonical registry name.
_DIRECT_ID_PROVIDER_ALIASES: dict[str, str] = {
    "spotify": "spotify",
    "sp": "spotify",
    "soundcloud": "soundcloud",
    "sc": "soundcloud",
    "youtube": "youtube_music",
    "youtube_music": "youtube_music",
    "ytm": "youtube_music",
    "yt": "youtube_music",
}

_DIRECT_ID_RE = re.compile(
    r"^(?P<provider>[A-Za-z_]+):(?P<track_id>[A-Za-z0-9_-]+)\s*$"
)


def _parse_direct_id(text: str) -> Optional[tuple[str, str]]:
    """`(canonical_provider_name, track_id)` if `text` matches the direct
    `provider:id` shortcut, else None. Stays restrictive: alphanumerics +
    `_`/`-` for the id portion, alphabetic for the provider, no spaces."""
    m = _DIRECT_ID_RE.match(text)
    if not m:
        return None
    alias = m.group("provider").lower()
    canonical = _DIRECT_ID_PROVIDER_ALIASES.get(alias)
    if not canonical:
        return None
    return canonical, m.group("track_id")


# Generic fallback thumbnail when a track has no artwork. Hosted on
# catbox so we don't lean on a provider-branded asset (the old SoundCloud
# logo made search results look SC-only even when most hits were Spotify).
DEFAULT_THUMBNAIL = "https://files.catbox.moe/6vh75g.jpg"

# Per-provider emoji shown as the leading character of each search-mode
# result's description, so the platform is identifiable at a glance.
PROVIDER_EMOJI = {
    "spotify": "🟢",
    "soundcloud": "☁️",
    "youtube_music": "▶️",
}

# Hard per-provider deadline for inline search. Telegram only gives the bot
# ~10s to answer the inline_query, so we cap each provider individually
# below that and ship whatever the fast ones returned.
PROVIDER_SEARCH_TIMEOUT_S = 3.0

# Inline fallback budget. find_alternative_track triggers a full
# `provider.search()` round-trip per registered provider — when a
# provider's auth handshake stalls (e.g. Spotify TOTP secrets endpoint
# timing out at 30s), the inline_query response blows past Telegram's
# ~10s window and the user sees nothing. Cap aggressively here so DRM
# URLs always answer within budget, with or without a fallback hit.
INLINE_FALLBACK_TIMEOUT_S = 5.0


async def _inline_fallback(
    registry: Registry, track: Track,
) -> tuple[Optional[Track], bool]:
    """Wrapper around `find_alternative_track` that enforces the inline
    latency budget and swallows transient provider failures.

    Returns `(track_or_none, timed_out)`:
      * `(Track, False)`  — found an alt match
      * `(None,  True)`   — fallback exceeded INLINE_FALLBACK_TIMEOUT_S
                            (Spotify auth stall, etc.) — caller can
                            distinguish "no match" from "couldn't ask"
                            and show a try-again hint to the user
      * `(None,  False)`  — fallback completed but found no match"""
    try:
        result = await asyncio.wait_for(
            find_alternative_track(registry, track),
            timeout=INLINE_FALLBACK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "inline fallback timed out after {:.1f}s for [{}:{}]",
            INLINE_FALLBACK_TIMEOUT_S, track.provider, track.track_id,
        )
        return None, True
    except Exception:
        logger.exception(
            "inline fallback crashed for [{}:{}]",
            track.provider, track.track_id,
        )
        return None, False
    return (result[1] if result is not None else None), False


def _result_id(t: Track, idx: int = 0) -> str:
    rid = f"{t.provider}:{t.track_id}"
    return rid[:64]


def _downloading_kb() -> InlineKeyboardMarkup:
    """Single button. Purr's `download_status_button` literal."""
    return InlineKeyboardMarkup(inline_keyboard=[[downloading_button()]])


def _description_for(t: Track, *, show_provider: bool = False) -> str:
    """Compact one-line description. For search-mode results we prefix the
    provider emoji so the platform is identifiable at a glance without
    spelling out the full name (the icon alone is enough — Spotify green
    / SoundCloud cloud / YT triangle). For URL-paste / playlist results
    no emoji is added since the source is already implied.

    Format:
        search:   "🟢 By Artist • 2:34"
        single:   "By Artist • 2:34"
        playlist: "By Artist • 2:34"
    """
    parts: list[str] = []
    if show_provider:
        parts.append(PROVIDER_EMOJI.get(t.provider, "🎵"))
    parts.append(f"By {t.artists_str}")
    out = " ".join(parts)
    if t.duration_seconds:
        out += f" • {t.duration_str}"
    return out


def _article(
    t: Track,
    *,
    title: str,
    registry: Registry,
    bot_username: str,
    idx: int = 0,
    show_provider: bool = False,
) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id=_result_id(t, idx),
        title=title,
        description=_description_for(t, show_provider=show_provider),
        thumbnail_url=t.artwork_url or DEFAULT_THUMBNAIL,
        input_message_content=InputTextMessageContent(
            message_text=format_track_caption(t, bot_username),
            parse_mode="HTML",
            disable_web_page_preview=True,
        ),
        reply_markup=_downloading_kb(),
    )


# ---- result builders per query kind --------------------------------------


async def _build_results(
    query: str,
    registry: Registry,
    cache: FileIdCache,
    search_providers: list[str],
    per_provider: int,
    limit: int,
) -> tuple[list[Track], str, int]:
    """Returns (tracks, mode, total) where:
    tracks  — list capped at `limit` for display.
    mode    — 'single' | 'playlist' | 'search'.
    total   — *uncapped* hit count behind the response. The UI uses
              this for the "Found N tracks" header so the user sees
              the real total even when display is truncated."""
    direct = _parse_direct_id(query)
    if direct is not None:
        provider_name, track_id = direct
        provider = registry.get(provider_name)
        if provider is None:
            return [], "unsupported_url", 0
        try:
            t = await provider.get_track(track_id)
            if provider_name == "soundcloud":
                from providers.soundcloud.provider import (  # noqa: SLF001
                    SoundCloudProvider,
                )

                if isinstance(provider, SoundCloudProvider):
                    try:
                        await provider.preflight_track(t)
                    except ProviderError as pe:
                        if pe.reason in FALLBACK_REASONS:
                            alt, timed_out = await _inline_fallback(registry, t)
                            if alt is not None:
                                return [alt], "single", 1
                            if timed_out and pe.reason == "drm":
                                return [], "drm_fallback_timeout", 0
                        raise
        except ProviderError as e:
            if e.reason == "goplus":
                return [], "goplus_blocked", 0
            if e.reason == "unavailable":
                return [], "track_unavailable", 0
            if e.reason == "drm":
                return [], "drm_blocked", 0
            return [], "track_not_found", 0
        except TrackNotFoundError:
            return [], "track_not_found", 0
        except DlmusError as e:
            logger.debug(
                "direct-id lookup failed [{}:{}]: {}: {}",
                provider_name,
                track_id,
                type(e).__name__,
                e,
            )
            return [], "track_not_found", 0
        except Exception:
            logger.exception(
                "direct-id lookup crashed [%s:%s]",
                provider_name,
                track_id,
            )
            return [], "track_not_found", 0
        return ([t] if t else []), "single", (1 if t else 0)

    parsed = parse_url(query, registry)
    if parsed is None and _looks_like_url(query):
        # User typed a URL that no provider claims. Don't pretend it's a
        # search query — surface "Unsupported URL" so they know which
        # platforms are actually wired up.
        return [], "unsupported_url", 0
    if parsed and parsed.kind == "url" and parsed.provider in {"spotify", "soundcloud"}:
        resolved = await resolve_short_url(parsed.entity_id)
        reparsed = parse_url(resolved, registry)
        if reparsed and reparsed.kind != "url":
            parsed = reparsed
        elif parsed.provider == "soundcloud":
            parsed = parsed.__class__(
                provider=parsed.provider,
                kind=parsed.kind,
                entity_id=resolved,
                url=resolved,
            )
        else:
            return [], "unsupported_url", 0
    if parsed:
        provider = registry.get(parsed.provider)
        if provider is not None:
            try:
                if parsed.kind == "track":
                    try:
                        t = await provider.get_track(parsed.entity_id)
                        if parsed.provider == "soundcloud":
                            from providers.soundcloud.provider import (  # noqa: SLF001
                                SoundCloudProvider,
                            )

                            if isinstance(provider, SoundCloudProvider):
                                try:
                                    await provider.preflight_track(t)
                                except ProviderError as pe:
                                    if pe.reason in FALLBACK_REASONS:
                                        alt, timed_out = await _inline_fallback(registry, t)
                                        if alt is not None:
                                            return [alt], "single", 1
                                        if timed_out and pe.reason == "drm":
                                            return [], "drm_fallback_timeout", 0
                                    raise
                        return [t], "single", 1
                    except ProviderError as e:
                        if e.reason == "goplus":
                            return [], "goplus_blocked", 0
                        if e.reason == "unavailable":
                            return [], "track_unavailable", 0
                        if e.reason == "drm":
                            return [], "drm_blocked", 0
                        raise
                if parsed.kind == "album":
                    album = await provider.get_album(parsed.entity_id)
                    if album is None:
                        return [], "playlist", 0
                    total = album.total_tracks or len(album.tracks)
                    return list(album.tracks)[:limit], "playlist", total
                if parsed.kind == "playlist":
                    pl = await provider.get_playlist(parsed.entity_id)
                    if pl is None:
                        return [], "playlist", 0
                    total = pl.total_tracks or len(pl.tracks)
                    return list(pl.tracks)[:limit], "playlist", total
                if parsed.kind == "artist":
                    # Wrap separately so a Mercury 404 (bogus id) maps
                    # to "artist_not_found" rather than the generic
                    # outer "unsupported_url" handler — the user's
                    # intent (artist URL) is unambiguous here.
                    try:
                        artist = await provider.get_artist(parsed.entity_id)
                    except Exception:
                        logger.exception(
                            "artist resolve failed: {}",
                            parsed.url,
                        )
                        return [], "artist_not_found", 0
                    if artist is None:
                        return [], "artist_not_found", 0
                    total = artist.total_tracks or len(artist.tracks)
                    if not artist.tracks:
                        return [], "artist_empty", 0
                    return list(artist.tracks)[:limit], "artist", total
                if parsed.kind == "url" and parsed.provider == "soundcloud":
                    from providers.soundcloud.provider import (  # noqa: SLF001
                        SoundCloudProvider,
                        _track_from_json,
                    )

                    if isinstance(provider, SoundCloudProvider):
                        try:
                            kind, data = await provider.resolve_kind(parsed.entity_id)
                        except TrackNotFoundError:
                            return [], "unsupported_url", 0
                        if kind == "track":
                            t = _track_from_json(data)
                            if t is not None and (t.extra or {}).get("is_goplus"):
                                alt, _timed_out = await _inline_fallback(registry, t)
                                if alt is not None:
                                    return [alt], "single", 1
                                return [], "goplus_blocked", 0
                            if t is not None:
                                try:
                                    await provider.preflight_track(t)
                                except ProviderError as pe:
                                    if pe.reason in FALLBACK_REASONS:
                                        alt, timed_out = await _inline_fallback(registry, t)
                                        if alt is not None:
                                            return [alt], "single", 1
                                        if timed_out and pe.reason == "drm":
                                            return [], "drm_fallback_timeout", 0
                                    if pe.reason == "goplus":
                                        return [], "goplus_blocked", 0
                                    if pe.reason == "drm":
                                        return [], "drm_blocked", 0
                                    if pe.reason == "unavailable":
                                        return [], "track_unavailable", 0
                                    raise
                            return ([t] if t else []), "single", (1 if t else 0)
                        if kind == "playlist":
                            # Delegate to the album/playlist resolver so
                            # stub tracks (SC only inlines metadata for
                            # first ~5 entries) get batch-hydrated. Without
                            # this every entry past the 5th shows up as
                            # "Unknown Artist - Unknown".
                            container = await (
                                provider.get_album(parsed.entity_id)
                                if data.get("is_album")
                                else provider.get_playlist(parsed.entity_id)
                            )
                            if container is None:
                                return [], "playlist", 0
                            total = container.total_tracks or len(container.tracks)
                            return list(container.tracks)[:limit], "playlist", total
                        if kind == "user":
                            # Artist profile on SoundCloud — pull their
                            # own uploads (no reposts) via get_artist.
                            artist = await provider.get_artist(parsed.entity_id)
                            if artist is None:
                                return [], "artist_not_found", 0
                            if not artist.tracks:
                                return [], "artist_empty", 0
                            total = artist.total_tracks or len(artist.tracks)
                            return list(artist.tracks)[:limit], "artist", total
                        # SC URL resolved to something we can't handle.
                        logger.info(
                            "soundcloud URL resolved to unsupported kind={!r}: {}",
                            kind,
                            parsed.url,
                        )
                        return [], "unsupported_url", 0
            except TrackNotFoundError:
                return [], "track_not_found", 0
            except DlmusError as e:
                # URL was recognised + claimed by a provider, so this is
                # a fetch failure (token rotation, AP socket drop, region
                # lock, etc.) — not a genuinely unsupported URL. Surface
                # it as such so the user knows to retry instead of
                # assuming the platform isn't wired up.
                logger.warning(
                    "inline url fetch failed [{}:{}:{}]: {}: {}",
                    parsed.provider,
                    parsed.kind,
                    parsed.entity_id,
                    type(e).__name__,
                    e,
                )
                return [], "url_fetch_failed", 0
            except Exception:
                logger.exception(
                    "inline url resolve crashed [{}:{}:{}]",
                    parsed.provider,
                    parsed.kind,
                    parsed.entity_id,
                )
                return [], "url_fetch_failed", 0

    tasks: list[asyncio.Task[list[Track]]] = []
    for name in search_providers:
        p = registry.get(name)
        if p is None:
            continue
        tasks.append(asyncio.create_task(_safe_search(p, query, per_provider)))
    if not tasks:
        return [], "search", 0
    bundles = await asyncio.gather(*tasks)
    merged: list[Track] = [t for bundle in bundles for t in bundle]
    total_found = sum(len(bundle) for bundle in bundles)
    # Three-pass dedupe — first catches exact (artist, title) duplicates;
    # second catches near-duplicates (same artist + duration, fuzzy
    # title overlap) like a SoundCloud re-upload of a Spotify track with
    # uploader noise tacked on ("(MV IN DESC)" etc); third catches
    # re-uploads whose ARTIST field differs (uploader name) but whose
    # TITLE embeds the Spotify artist+song at the same duration
    # ("GIORGI SANIKIDZE — Psychonaut 4 - Suicide Is Legal"). All prefer
    # Spotify since its metadata is more authoritative.
    deduped = dedupe_tracks(merged)
    deduped = dedupe_near_duplicates(deduped)
    deduped = dedupe_embedded_title(deduped)
    # No local re-scoring — provider ranking is authoritative (see
    # `interleave_by_provider` for the rationale).
    ranked = interleave_by_provider(deduped, limit=limit)
    # `total` is the raw summed provider count for the header text:
    # "Found X tracks" should reflect all provider hits together.
    return ranked, "search", total_found


async def _safe_search(provider, query: str, limit: int) -> list[Track]:
    """Run a single-provider search with a hard timeout. Empty list on
    timeout or any failure — never let one slow provider stall the inline
    response."""
    try:
        return await asyncio.wait_for(
            provider.search(query, limit=limit),
            timeout=PROVIDER_SEARCH_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "provider {} search timed out after {:.1f}s",
            provider.name,
            PROVIDER_SEARCH_TIMEOUT_S,
        )
        return []
    except Exception:
        logger.exception("provider {} search failed", provider.name)
        return []


# ---- empty-query default -------------------------------------------------


def _empty_query_results(
    bot_username: str, visible: frozenset[str]
) -> list[InlineQueryResultArticle]:
    """Placeholder when the query box is empty: same story as /start, inline slant."""
    return [
        InlineQueryResultArticle(
            id="example1",
            title="Search or paste a link",
            description=format_onboarding_card_description(visible),
            input_message_content=InputTextMessageContent(
                message_text=format_inline_empty_message(bot_username, visible),
                parse_mode="HTML",
            ),
            thumbnail_url=DEFAULT_THUMBNAIL,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        example_inline_search_button("drain gang"),
                    ]
                ]
            ),
        )
    ]


# ---- main handler --------------------------------------------------------


@router.inline_query()
async def on_inline_query(
    query: InlineQuery,
    registry: Registry,
    cache: FileIdCache,
    job_runner: JobRunner,
    inline_results_limit: int,
    per_provider_limit: int,
    inline_search_providers: list[str],
    config: Any,
) -> None:
    bot_username = job_runner.bot_username
    inline_cache_s = max(0, int(getattr(config, "INLINE_CACHE_SECONDS", 60 * 60)))

    text = (query.query or "").strip()
    # Detect URL vs. free-text up front so the start-log accurately
    # reflects which path the query will actually take. URL queries
    # only ever hit the resolver for the matching provider — listing
    # the configured search providers there was misleading (made it
    # look like SoundCloud was being queried for a Spotify URL).
    url_provider = None
    if text:
        for p in registry.all():
            if p.parse_url(text):
                url_provider = p.name
                break
    if url_provider:
        logger.info(
            "<cyan>[inline]</cyan> user={} url={!r} <magenta>resolver={}</magenta>",
            query.from_user.id if query.from_user else "?",
            text,
            url_provider,
        )
    else:
        logger.info(
            "<cyan>[inline]</cyan> user={} query={!r} providers={}",
            query.from_user.id if query.from_user else "?",
            text,
            ",".join(inline_search_providers),
        )
    if not text:
        visible = visible_help_providers(registry, config)
        await query.answer(
            results=_empty_query_results(bot_username, visible),
            cache_time=inline_cache_s,
            is_personal=False,
        )
        return

    tracks, mode, total = await _build_results(
        text,
        registry,
        cache,
        inline_search_providers,
        per_provider_limit,
        inline_results_limit,
    )

    results: list[InlineQueryResultArticle] = []
    for i, t in enumerate(tracks):
        # Title shape per mode:
        #   single    -> 🎵 <title>             (one track URL paste; no
        #                                       index, no platform tag)
        #   playlist  -> {i+1}. <title>          (album / playlist track list)
        #   artist    -> {i+1}. <title>          (artist's catalogue)
        #   search    -> <title>                  (free-text query; platform
        #                                       indicator goes in description)
        if mode == "single":
            title = f"🎵 {t.title}"
        elif mode in ("playlist", "artist"):
            title = f"{i + 1}. {t.title}"
        else:
            title = t.title or "Untitled Track"
        results.append(
            _article(
                t,
                title=title,
                registry=registry,
                bot_username=bot_username,
                idx=i,
                show_provider=(mode == "search"),
            )
        )
        if len(results) >= inline_results_limit:
            break

    # Header line shown above the inline results in some Telegram clients.
    # Each user-visible mode gets its own switch_pm_text — including the
    # artist failure modes — so the user always sees *why* there are no
    # results instead of a generic "No results".
    if mode == "unsupported_url":
        switch_pm_text: Optional[str] = "❌ Unsupported URL"
    elif mode == "url_fetch_failed":
        # URL was recognised + claimed by a provider but the lookup
        # itself failed (token rotation, AP socket drop, region lock,
        # ...). Surface as a retry-friendly hint instead of pretending
        # the URL is unsupported.
        switch_pm_text = "⚠️ Please try again later"
    elif mode == "track_not_found":
        switch_pm_text = "❌ Track ID not found"
    elif mode == "goplus_blocked":
        switch_pm_text = "❌ Go+ tracks can't be downloaded"
    elif mode == "drm_blocked":
        switch_pm_text = "🔒 Can't download, its DRM protected"
    elif mode == "drm_fallback_timeout":
        switch_pm_text = "🔒 DRM track — Spotify slow, try again"
    elif mode == "track_unavailable":
        switch_pm_text = "❌ Track unavailable for download"
    elif mode == "artist_not_found":
        switch_pm_text = "❌ Artist not found"
    elif mode == "artist_empty":
        switch_pm_text = "🎤 Artist has no playable tracks"
    elif not results:
        switch_pm_text = "No results"
    elif mode == "single":
        switch_pm_text = None
    elif mode == "artist":
        switch_pm_text = f"🎤 {total} tracks by this artist"
    else:
        switch_pm_text = f"Found {total} tracks"

    # Free-text search results are safe to cache — same query, same
    # results within a reasonable window. URL pastes (`single`, playlist,
    # artist) and any URL-resolution failure must NOT cache: if Spotify's
    # cold-start handshake stalled the first attempt, Telegram would
    # otherwise keep serving the empty/error response when the user
    # retypes the same URL.
    is_url_mode = mode != "search"
    answer_kwargs: dict = {
        "results": results,
        "cache_time": 0 if is_url_mode else inline_cache_s,
        "is_personal": True,
    }
    if switch_pm_text is not None:
        answer_kwargs["switch_pm_text"] = switch_pm_text
        answer_kwargs["switch_pm_parameter"] = "from_inline"
    # Telegram only honours an inline_query response within ~10 seconds
    # of the original keystroke. Slow provider hops can blow past that
    # (especially on cold-start when sp_dc + TOTP + AP handshake stack
    # up), so swallow the resulting "query is too old" BadRequest — the
    # user has typed something newer by then anyway.
    with contextlib.suppress(TelegramBadRequest):
        await query.answer(**answer_kwargs)
    logger.info(
        "<cyan>[inline]</cyan> done user={} mode={} total={} shown={}",
        query.from_user.id if query.from_user else "?",
        mode,
        total,
        len(results),
    )
