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
Whatever has come back when the deadline hits is what gets fuzzy-merged and
shown — typically both Spotify + SoundCloud, but worst case one of them
alone if the other is throttled.
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
from loguru import logger

from core.cache import FileIdCache
from core.exceptions import DlmusError, ProviderError, TrackNotFoundError
from core.fuzz import dedupe_tracks, rank_balanced
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
                from providers.soundcloud.provider import SoundCloudProvider  # noqa: SLF001
                if isinstance(provider, SoundCloudProvider):
                    await provider.preflight_track(t)
        except ProviderError as e:
            if e.reason == "goplus":
                return [], "goplus_blocked", 0
            if e.reason == "unavailable":
                return [], "track_unavailable", 0
            return [], "track_not_found", 0
        except TrackNotFoundError:
            return [], "track_not_found", 0
        except DlmusError as e:
            logger.debug(
                "direct-id lookup failed [{}:{}]: {}: {}",
                provider_name, track_id, type(e).__name__, e,
            )
            return [], "track_not_found", 0
        except Exception:
            logger.exception(
                "direct-id lookup crashed [%s:%s]", provider_name, track_id,
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
                            from providers.soundcloud.provider import SoundCloudProvider  # noqa: SLF001
                            if isinstance(provider, SoundCloudProvider):
                                await provider.preflight_track(t)
                        return [t], "single", 1
                    except ProviderError as e:
                        if e.reason == "goplus":
                            return [], "goplus_blocked", 0
                        if e.reason == "unavailable":
                            return [], "track_unavailable", 0
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
                            "artist resolve failed: {}", parsed.url,
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
                                return [], "goplus_blocked", 0
                            if t is not None:
                                await provider.preflight_track(t)
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
                            kind, parsed.url,
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
                    parsed.provider, parsed.kind, parsed.entity_id,
                    type(e).__name__, e,
                )
                return [], "url_fetch_failed", 0
            except Exception:
                logger.exception(
                    "inline url resolve crashed [{}:{}:{}]",
                    parsed.provider, parsed.kind, parsed.entity_id,
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
    # Drop cross-provider duplicates (same artist+title) — prefer Spotify
    # since its metadata is more authoritative — then interleave by
    # provider so neither catalogue monopolises the list when one
    # provider's titles happen to score systematically higher.
    deduped = dedupe_tracks(merged)
    ranked = rank_balanced(query, deduped, limit=limit)
    if not ranked and merged:
        # Safety net: if dedupe/ranking ever collapses unexpectedly,
        # still return visible results instead of an empty inline list.
        ranked = merged[:limit]
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


def _empty_query_results(bot_username: str, visible: frozenset[str]) -> list[InlineQueryResultArticle]:
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
        switch_pm_text = "⚠️ Couldn't load that link — try again"
    elif mode == "track_not_found":
        switch_pm_text = "❌ Track ID not found"
    elif mode == "goplus_blocked":
        switch_pm_text = "❌ SoundCloud Go+ track can't be downloaded"
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

    answer_kwargs: dict = {
        "results": results,
        "cache_time": inline_cache_s,
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
