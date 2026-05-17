"""SoundCloud provider — search, resolve URLs, download via HLS/progressive."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import asyncio

import aiohttp

from core.audio_convert import ffmpeg_available
from core.exceptions import ProviderError, TrackNotFoundError
from core.models import Album, ArtistRef, DownloadResult, Playlist, Track

from .api import DEFAULT_HEADERS, SoundCloudAPI
from .titles import clean_title, strip_redundant_artist_prefix
from ..base import Provider, StageCallback

log = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".") or "track"


def _artist_ref_for(user: dict, fallback_name: Optional[str] = None) -> Optional[ArtistRef]:
    if not isinstance(user, dict):
        return ArtistRef(name=fallback_name or "Unknown Artist") if fallback_name else None
    name = fallback_name or user.get("username")
    if not name:
        return None
    perm = user.get("permalink_url")
    uid = user.get("id")
    # Append `?urn=soundcloud:users:<id>` for the same reason we tag
    # track URLs: SoundCloud user permalinks change when the artist
    # renames their account, but the numeric user_id is immutable. The
    # URN keeps the link resolvable even after a rename. SC ignores
    # unknown query params on its web pages.
    if perm and uid is not None:
        sep = "&" if "?" in perm else "?"
        perm = f"{perm}{sep}urn=soundcloud:users:{uid}"
    return ArtistRef(
        name=name,
        artist_id=str(uid) if uid is not None else None,
        url=perm,
    )


def _track_from_json(data: dict) -> Optional[Track]:
    if not isinstance(data, dict):
        return None
    tid = data.get("id")
    if tid is None:
        return None
    user = data.get("user") or {}
    raw_title = data.get("title") or "<unknown>"
    pub = data.get("publisher_metadata") or {}
    pub_artist = pub.get("artist")

    # If publisher metadata names an artist explicitly, trust it.
    # Otherwise, use the SoundCloud account name. We do NOT split artist
    # from title heuristically; we only drop a leading "Artist - " prefix
    # when it matches the known artist field exactly.
    if pub_artist:
        title = strip_redundant_artist_prefix(clean_title(raw_title), pub_artist)
        primary = _artist_ref_for(user, fallback_name=pub_artist)
    else:
        primary = _artist_ref_for(user, fallback_name=None)
        title = strip_redundant_artist_prefix(
            clean_title(raw_title),
            (primary.name if primary else None),
        )

    artists = [primary] if primary else []
    artwork = (data.get("artwork_url") or user.get("avatar_url") or "")
    if artwork:
        artwork = artwork.replace("-large.", "-t1080x1080.")
    perm = data.get("permalink_url") or ""
    # Append the SoundCloud URN (`soundcloud:tracks:<id>`) as a query
    # param so the URL stays resolvable even after the slug changes —
    # artists rename tracks, but track IDs are immutable. `urn` is SC's
    # own field name; SC ignores unknown query params on its web pages,
    # and our URL parser picks up `?urn=` to recover the numeric ID
    # without needing to re-resolve the (potentially stale) permalink.
    if perm and tid is not None:
        sep = "&" if "?" in perm else "?"
        perm_with_urn = f"{perm}{sep}urn=soundcloud:tracks:{tid}"
    else:
        perm_with_urn = perm
    return Track(
        provider="soundcloud",
        track_id=str(tid),
        title=title,
        artists=artists,
        album=pub.get("album_title"),
        duration_seconds=int((data.get("duration") or 0) // 1000),
        artwork_url=artwork or None,
        url=perm_with_urn,
        isrc=pub.get("isrc"),
        extra={
            "transcodings": (data.get("media") or {}).get("transcodings") or [],
            "permalink": data.get("permalink"),
            "permalink_url": perm,  # raw, no urn — for fallback resolves
            "kind": data.get("kind"),
            # `policy` flags Go+ subscription-only tracks ("SNIP" → snippet
            # streams only, full audio behind the paywall). Used by the
            # provider's download() to short-circuit with a goplus reason
            # before pulling 30s of audio that we can't deliver anyway.
            "policy": data.get("policy"),
            "monetization_model": data.get("monetization_model"),
            "is_goplus": _is_goplus_json(data),
            # `is_drm_only` flags tracks whose only real streams are
            # CommonEncryption (FairPlay sample-AES / Widevine). SC keeps
            # phantom legacy MP3 entries on those, but they 404 on resolve
            # — so download() short-circuits with reason="drm" and the
            # cross-provider fallback layer can try Spotify / YT Music.
            "is_drm_only": _has_encrypted_only_transcodings(
                (data.get("media") or {}).get("transcodings") or []
            ),
        },
    )


def _is_goplus_json(data: dict) -> bool:
    """Best-effort Go+ detector for SoundCloud tracks.

    `policy == "SNIP"` is the canonical flag for snippet-only playback.
    Some responses are sparse, so we also treat "all transcodings are
    snipped previews" as Go+ fallback evidence.
    """
    if not isinstance(data, dict):
        return False
    if data.get("policy") == "SNIP":
        return True
    media = data.get("media") or {}
    transcodings = media.get("transcodings") or []
    if transcodings and all(bool(t.get("snipped")) for t in transcodings):
        return True
    return False


# Protocol substrings SC uses for CommonEncryption variants. `cbcs` =
# AES-CBC sample-AES (Apple FairPlay-style), `cenc`/`ctr` = AES-CTR
# (Widevine / PlayReady). We can resolve the manifest, but the segments
# are wrapped in a real DRM key system — no chance of decryption without
# breaking DRM, so we treat any track whose only real variants are these
# as `reason="drm"` and let the fallback layer try another provider.
_ENCRYPTED_PROTO_HINTS = ("cbc", "cenc", "ctr", "encrypted")


def _is_encrypted_protocol(proto: str) -> bool:
    p = (proto or "").lower()
    return any(h in p for h in _ENCRYPTED_PROTO_HINTS)


def _has_encrypted_only_transcodings(transcodings: list[dict]) -> bool:
    """True iff the track has real (non-snipped) encrypted variants and
    no plain non-legacy variant we could fall back on.

    SC keeps phantom legacy `progressive`/`hls` entries alongside the
    real encrypted ones — those 404 when actually resolved, so we can't
    rely on them. A track is DRM-locked-from-our-POV when:
      * at least one real (non-snipped) variant uses encrypted HLS, AND
      * no real non-legacy plain variant exists
    Tracks with a plain non-legacy variant get the runtime to try it
    first (defence-in-depth re-check in download()/preflight() catches
    cases where SC mislabels the legacy flag)."""
    if not transcodings:
        return False
    real = [t for t in transcodings if not t.get("snipped")]
    if not real:
        return False

    def _proto(t: dict) -> str:
        return (t.get("format") or {}).get("protocol") or ""

    has_encrypted = any(_is_encrypted_protocol(_proto(t)) for t in real)
    if not has_encrypted:
        return False
    has_plain_non_legacy = any(
        not _is_encrypted_protocol(_proto(t))
        and not t.get("is_legacy_transcoding")
        for t in real
    )
    return not has_plain_non_legacy


def _transcoding_rank(t: dict) -> tuple[int, int, int]:
    """Rank a SoundCloud transcoding by audio quality.

    SC exposes several variants per track. Order of preference (high → low):
        1. `quality == "hq"`           — Go+ / HQ premium 256 kbps streams
        2. AAC / MP4                   — better psychoacoustic at any rate
        3. MP3                         — standard
        4. Opus / OGG                  — usually only the 64 kbps preview
    Within the same audio tier we slightly prefer **progressive** over HLS
    purely because progressive is a single GET (simpler + more reliable);
    quality is identical when `quality` and `preset` match. Higher tuple
    values win in `max(..., key=_transcoding_rank)`."""
    fmt = t.get("format") or {}
    mime = (fmt.get("mime_type") or "").lower()
    proto = fmt.get("protocol") or ""
    quality_tier = 2 if (t.get("quality") == "hq") else 1
    if "aac" in mime or "mp4" in mime:
        codec_tier = 3
    elif "mpeg" in mime or "mp3" in mime:
        codec_tier = 2
    else:
        codec_tier = 1
    proto_tier = 1 if proto == "progressive" else 0
    return (quality_tier, codec_tier, proto_tier)


def _pick_transcoding(
    transcodings: list[dict],
    *,
    allow_hls_fmp4: bool = True,
) -> Optional[dict]:
    """Pick the highest-quality decryptable transcoding. Skips snippet
    streams (`snipped == True`) since they're 30-second previews, not
    full tracks."""
    if not transcodings:
        return None
    full = [t for t in transcodings if not t.get("snipped")]
    pool = full or transcodings
    # Skip encrypted transport variants - we can only ingest plain
    # `progressive` and `hls` streams. Encrypted HLS protocols can
    # resolve but yield unusable output in our downloader path.
    compatible_protocols = {"progressive", "hls"}
    protocol_ok = [
        t for t in pool
        if ((t.get("format") or {}).get("protocol") or "").lower() in compatible_protocols
    ]
    if protocol_ok:
        pool = protocol_ok
    if not allow_hls_fmp4:
        compatible: list[dict] = []
        for t in pool:
            fmt = t.get("format") or {}
            mime = (fmt.get("mime_type") or "").lower()
            proto = (fmt.get("protocol") or "").lower()
            is_hls_fmp4 = proto == "hls" and ("aac" in mime or "mp4" in mime)
            if not is_hls_fmp4:
                compatible.append(t)
        if compatible:
            pool = compatible
    return max(pool, key=_transcoding_rank)


def _candidate_transcodings(
    transcodings: list[dict],
    *,
    allow_hls_fmp4: bool = True,
) -> list[dict]:
    """Compatible transcodings sorted best-first."""
    if not transcodings:
        return []
    full = [t for t in transcodings if not t.get("snipped")]
    pool = full or transcodings
    compatible_protocols = {"progressive", "hls"}
    protocol_ok = [
        t for t in pool
        if ((t.get("format") or {}).get("protocol") or "").lower() in compatible_protocols
    ]
    if protocol_ok:
        pool = protocol_ok
    if not allow_hls_fmp4:
        compatible: list[dict] = []
        for t in pool:
            fmt = t.get("format") or {}
            mime = (fmt.get("mime_type") or "").lower()
            proto = (fmt.get("protocol") or "").lower()
            is_hls_fmp4 = proto == "hls" and ("aac" in mime or "mp4" in mime)
            if not is_hls_fmp4:
                compatible.append(t)
        if compatible:
            pool = compatible
    return sorted(pool, key=_transcoding_rank, reverse=True)


def _format_metadata(t: dict) -> tuple[str, str, str]:
    """Returns (format_name, file_ext, mime_type) for a transcoding."""
    fmt = t.get("format") or {}
    proto = fmt.get("protocol", "?")
    mime = fmt.get("mime_type", "audio/mpeg")
    if "opus" in mime or "ogg" in mime:
        ext = "opus" if "opus" in mime else "ogg"
        return f"opus_{proto}", ext, mime
    if "mp4" in mime or "aac" in mime:
        return f"aac_{proto}", "m4a", "audio/mp4"
    return f"mp3_{proto}", "mp3", "audio/mpeg"


class SoundCloudProvider(Provider):
    name = "soundcloud"
    label = "SoundCloud"

    URL_PATTERNS = [
        # `?urn=soundcloud:tracks:<id>` — our outbound permalink format. We
        # capture the numeric ID directly so future re-shares keep working
        # even after the artist renames the track (immutable ID, mutable
        # slug). Must come before the generic permalink pattern.
        ("track", re.compile(
            r"https?://(?:www\.|m\.|on\.)?soundcloud\.com/[^\s]*?"
            r"[?&]urn=soundcloud:tracks:(\d+)"
        )),
        # `on.soundcloud.com/<short>` -> needs follow-redirect; resolve handles
        # both forms by accepting the URL verbatim. The negative lookahead
        # skips URLs that already carry our `?urn=soundcloud:tracks:` tag,
        # otherwise `parse_all` would emit *both* a track-kind hit (from
        # the URN regex above) **and** a url-kind hit, triggering double
        # downloads on the same URL.
        ("url", re.compile(
            r"(https?://(?:www\.|m\.|on\.)?soundcloud\.com/"
            r"(?![^\s]*[?&]urn=soundcloud:tracks:)"
            r"[^\s?#]+)"
        )),
        ("url", re.compile(r"(https?://snd\.sc/[^\s?#]+)")),
    ]

    def __init__(self, client_id: Optional[str] = None):
        self._api = SoundCloudAPI(client_id=client_id)

    async def start(self) -> None:
        await self._api.start()

    async def close(self) -> None:
        await self._api.close()

    @property
    def api(self) -> SoundCloudAPI:
        return self._api

    # ---- url plumbing ----------------------------------------------------

    def canonical_url(self, kind: str, entity_id: str) -> str:
        # entity_id is the full URL — strip query string only.
        return entity_id.split("?", 1)[0]

    def artist_url(self, artist_id: str) -> Optional[str]:
        # We can't reverse a numeric SC user_id to a permalink without an
        # API hit. Artist URLs are populated directly into ArtistRef during
        # parsing, so leave this None.
        return None

    # ---- search / fetch --------------------------------------------------

    async def search(self, query: str, limit: int = 25) -> list[Track]:
        items = await self._api.search_tracks(query, limit=limit)
        tracks = [t for t in (_track_from_json(i) for i in items) if t]
        # Filter both Go+ (snippet-only) and DRM-locked (CommonEncryption)
        # so users never pick a hit we can't deliver. The cross-provider
        # fallback in JobRunner still rescues URL-paste paths for those
        # cases — this filter just keeps the inline picker clean.
        return [
            t for t in tracks
            if not (t.extra or {}).get("is_goplus")
            and not (t.extra or {}).get("is_drm_only")
        ]

    async def get_track(self, entity_id: str) -> Track:
        if entity_id.isdigit():
            data = await self._api.get_track_by_id(entity_id)
        else:
            data = await self._api.resolve(entity_id)
        if data.get("kind") != "track":
            raise TrackNotFoundError(f"{entity_id} is a {data.get('kind')}, not a track")
        if _is_goplus_json(data):
            raise ProviderError(
                f"soundcloud track {entity_id} is Go+ (snippet only)",
                reason="goplus",
            )
        t = _track_from_json(data)
        if t is None:
            raise TrackNotFoundError(f"soundcloud {entity_id} not found")
        # Don't pre-raise on `is_drm_only` here — the bot still needs the
        # Track object to drive cross-provider fallback (artist + title +
        # duration come from this exact metadata). download() / preflight
        # are the gates that surface the `drm` reason.
        return t

    async def resolve_kind(self, entity_id: str) -> tuple[str, dict]:
        """Resolve any SoundCloud URL once. Returns (kind, raw_json) so the
        bot layer can decide whether to treat it as track / playlist / album
        without doing the request twice."""
        data = await self._api.resolve(entity_id)
        return (data.get("kind") or "unknown"), data

    async def preflight_track(self, track: Track) -> None:
        """Fail fast when no playable stream URL can be resolved."""
        transcodings = (track.extra or {}).get("transcodings") or []
        if not transcodings:
            full = await self.get_track(track.track_id)
            transcodings = (full.extra or {}).get("transcodings") or []
            track = full
        if (track.extra or {}).get("is_goplus") or (track.extra or {}).get("policy") == "SNIP":
            raise ProviderError(
                f"soundcloud track {track.track_id} is Go+ (snippet only)",
                reason="goplus",
            )
        if (track.extra or {}).get("is_drm_only") or _has_encrypted_only_transcodings(transcodings):
            raise ProviderError(
                f"soundcloud track {track.track_id} is DRM-protected (CommonEncryption)",
                reason="drm",
            )
        candidates = _candidate_transcodings(
            transcodings,
            allow_hls_fmp4=ffmpeg_available(),
        )
        if not candidates:
            if transcodings and all(bool(t.get("snipped")) for t in transcodings):
                raise ProviderError(
                    f"soundcloud track {track.track_id} is Go+ (snippet only)",
                    reason="goplus",
                )
            raise ProviderError(
                f"no transcodings for soundcloud track {track.track_id}",
                reason="unavailable",
            )
        for cand in candidates:
            try:
                await self._api.transcoding_url(cand["url"])
                return
            except ProviderError as e:
                log.error("sc preflight candidate skipped url=%s (%s): %s", cand.get("url"), type(e).__name__, e)
                continue
        # All candidates 404'd. If the real variants are all encrypted,
        # this is DRM — surface that so the fallback layer kicks in.
        if _has_encrypted_only_transcodings(transcodings):
            raise ProviderError(
                f"soundcloud track {track.track_id} is DRM-protected (all decryptable transcodings 404'd)",
                reason="drm",
            )
        raise ProviderError(
            f"no playable transcodings for soundcloud track {track.track_id}",
            reason="unavailable",
        )

    async def get_album(self, entity_id: str) -> Optional[Album]:
        data = await self._api.resolve(entity_id)
        if data.get("kind") != "playlist" or not data.get("is_album"):
            return None
        owner_user = data.get("user") or {}
        hydrated, total = await self._hydrate_playlist_tracks(data)
        return Album(
            provider="soundcloud",
            album_id=str(data.get("id")),
            title=data.get("title") or "<unknown>",
            artists=[_artist_ref_for(owner_user)] if owner_user else [],
            artwork_url=(data.get("artwork_url") or "").replace("-large.", "-t1080x1080.") or None,
            url=data.get("permalink_url") or entity_id,
            tracks=hydrated,
            total_tracks=total,
        )

    async def get_artist(self, entity_id: str) -> Optional[Playlist]:
        """SoundCloud-side artist resolution.

        `entity_id` is the artist's full permalink URL (SC URLs have no
        path-only ID form — the slug is the only identifier). We resolve
        once to get the numeric user_id + display name, then fetch
        `/users/<id>/tracks` for their *own* uploads. Reposts come from
        a different endpoint (`/reposts`) and are intentionally NOT
        included — the user asked for actual artist content."""
        try:
            user = await self._api.resolve(entity_id)
        except TrackNotFoundError:
            return None
        if user.get("kind") != "user":
            return None
        uid = user.get("id")
        if uid is None:
            return None
        items = await self._api.get_user_tracks(uid, limit=50)
        tracks = [t for t in (_track_from_json(it) for it in items) if t]
        name = user.get("username") or "Unknown Artist"
        artwork = (user.get("avatar_url") or "").replace("-large.", "-t1080x1080.") or None
        perm = user.get("permalink_url") or entity_id
        if uid is not None:
            sep = "&" if "?" in perm else "?"
            perm = f"{perm}{sep}urn=soundcloud:users:{uid}"
        return Playlist(
            provider="soundcloud",
            playlist_id=str(uid),
            title=name,
            owner=name,
            artwork_url=artwork,
            url=perm,
            tracks=tracks,
            # SC reports `track_count` on the user object, but it counts
            # everything (incl. private / removed). Use the actual
            # hydrated count for the visible total — what we'd actually
            # show in inline.
            total_tracks=user.get("track_count") or len(tracks),
        )

    async def get_playlist(self, entity_id: str) -> Optional[Playlist]:
        data = await self._api.resolve(entity_id)
        if data.get("kind") != "playlist" or data.get("is_album"):
            return None
        hydrated, total = await self._hydrate_playlist_tracks(data)
        return Playlist(
            provider="soundcloud",
            playlist_id=str(data.get("id")),
            title=data.get("title") or "<unknown>",
            owner=(data.get("user") or {}).get("username"),
            artwork_url=(data.get("artwork_url") or "").replace("-large.", "-t1080x1080.") or None,
            url=data.get("permalink_url") or entity_id,
            tracks=hydrated,
            total_tracks=total,
        )

    async def _hydrate_playlist_tracks(
        self, playlist_data: dict,
    ) -> tuple[list[Track], int]:
        """Resolve a SoundCloud playlist's `tracks` array into core.Track[].

        SC's /playlists/<id> response only embeds full metadata for the
        first ~5 entries; the rest come back as `{id, kind,
        monetization_model, policy}` stubs. We batch-resolve those via
        /tracks?ids=... so all 50 entries have a real title + artist +
        duration. Returns (hydrated tracks, total count) — the total
        reflects the playlist's full length, even when batch hydration
        loses a few region-locked entries."""
        items = playlist_data.get("tracks") or []
        total = len(items)
        if not items:
            return [], 0

        # Find the stubs (no `title`) and batch-resolve them.
        stub_ids = [it.get("id") for it in items if not it.get("title") and it.get("id") is not None]
        hydrated_by_id: dict[str | int, dict] = {}
        if stub_ids:
            try:
                fetched = await self._api.get_tracks_by_ids(stub_ids)
            except Exception:
                log.exception("soundcloud batch-hydrate failed; serving stubs as 'unknown'")
                fetched = []
            for f in fetched:
                fid = f.get("id")
                if fid is not None:
                    hydrated_by_id[fid] = f

        out: list[Track] = []
        for it in items:
            full = it if it.get("title") else hydrated_by_id.get(it.get("id"), it)
            tr = _track_from_json(full)
            if tr is not None:
                out.append(tr)
        return out, total

    # ---- download --------------------------------------------------------

    async def download(
        self, track: Track, dest_dir: str,
        *, on_stage: Optional[StageCallback] = None,
    ) -> DownloadResult:
        # Track may be a stub from a playlist resolve. Re-hit /tracks for
        # transcodings if missing.
        transcodings = (track.extra or {}).get("transcodings") or []
        if not transcodings:
            full = await self.get_track(track.track_id)
            transcodings = (full.extra or {}).get("transcodings") or []
            track = full

        # Go+ short-circuit: if `policy == "SNIP"` SoundCloud will only
        # serve us 30-second previews, no full audio. Surface as a
        # permanent failure with a goplus reason so the bot shows the
        # right user-facing message instead of retrying / fallback.
        if (track.extra or {}).get("is_goplus") or (track.extra or {}).get("policy") == "SNIP":
            raise ProviderError(
                f"soundcloud track {track.track_id} is Go+ (snippet only)",
                reason="goplus",
            )

        # DRM short-circuit: if every real variant rides an encrypted HLS
        # protocol (cbcs / cenc — FairPlay-style sample-AES or Widevine),
        # we have no chance of decrypting the segments. Surface as `drm`
        # so the bot's cross-provider fallback can try Spotify / YT Music
        # for the same artist + title instead of churning here.
        if (track.extra or {}).get("is_drm_only") or _has_encrypted_only_transcodings(transcodings):
            raise ProviderError(
                f"soundcloud track {track.track_id} is DRM-protected (CommonEncryption)",
                reason="drm",
            )

        candidates = _candidate_transcodings(
            transcodings,
            allow_hls_fmp4=ffmpeg_available(),
        )
        if not candidates:
            if transcodings and all(bool(t.get("snipped")) for t in transcodings):
                raise ProviderError(
                    f"soundcloud track {track.track_id} is Go+ (snippet only)",
                    reason="goplus",
                )
            # No usable streams + no Go+ flag → likely region-locked or
            # taken down. Surface as `unavailable` for the right popup.
            raise ProviderError(
                f"no transcodings for soundcloud track {track.track_id}",
                reason="unavailable",
            )

        chosen: Optional[dict] = None
        stream_url: Optional[str] = None
        for candidate in candidates:
            try:
                stream_url = await self._api.transcoding_url(candidate["url"])
                chosen = candidate
                break
            except ProviderError as e:
                log.error("sc transcoding candidate skipped url=%s (%s): %s", candidate.get("url"), type(e).__name__, e)
                continue
        if chosen is None or stream_url is None:
            # All plain candidates 404'd. If real variants are encrypted,
            # this is DRM — caught above already, but defence-in-depth:
            # phantom legacy MP3 entries that 404 alongside DRM-only real
            # streams still hit this branch on tracks where the upfront
            # detector wasn't triggered (e.g. mixed legacy + encrypted).
            if _has_encrypted_only_transcodings(transcodings):
                raise ProviderError(
                    f"soundcloud track {track.track_id} is DRM-protected (all decryptable transcodings 404'd)",
                    reason="drm",
                )
            raise ProviderError(
                f"no playable transcodings for soundcloud track {track.track_id}",
                reason="unavailable",
            )

        format_name, ext, mime = _format_metadata(chosen)
        protocol = (chosen.get("format") or {}).get("protocol") or "progressive"

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        artist_part = ", ".join(a.name for a in track.artists) or "Unknown Artist"
        out_path = Path(dest_dir) / (
            _sanitize_filename(f"{artist_part} - {track.title}") + f".{ext}"
        )

        if on_stage is not None:
            try:
                await on_stage("downloading")
            except Exception:
                log.debug("on_stage(downloading) failed", exc_info=True)

        if protocol == "progressive":
            await _download_file(self._api.http, stream_url, out_path)
        else:
            await _download_hls(self._api.http, stream_url, out_path, ext)

        size = out_path.stat().st_size
        if size == 0:
            if transcodings and all(bool(t.get("snipped")) for t in transcodings):
                raise ProviderError(
                    f"soundcloud track {track.track_id} is Go+ (snippet only)",
                    reason="goplus",
                )
            raise ProviderError(
                f"downloaded empty file for soundcloud track {track.track_id}",
                reason="unavailable",
            )
        return DownloadResult(
            track=track,
            file_path=str(out_path),
            format_name=format_name,
            size_bytes=size,
            mime_type=mime,
        )


async def _download_file(
    http: aiohttp.ClientSession, url: str, out_path: Path
) -> None:
    async with http.get(url, headers=DEFAULT_HEADERS) as r:
        if r.status != 200:
            raise ProviderError(f"sc cdn {r.status}")
        with out_path.open("wb") as fh:
            async for chunk in r.content.iter_chunked(64 * 1024):
                fh.write(chunk)


async def _download_hls(
    http: aiohttp.ClientSession, m3u8_url: str, out_path: Path, ext: str
) -> None:
    """Pull a SoundCloud HLS stream into `out_path`.

    Prefers ffmpeg-direct ingest (`ffmpeg -i <m3u8_url> -c copy`) — that
    handles every SC variant correctly:
        * MP3 HLS (legacy)        — simple TS-like segments
        * AAC fMP4 HLS (premium)  — fragmented MP4 with `#EXT-X-MAP` init
                                    segment that the manual concat path
                                    would silently drop, producing the
                                    "trun track id unknown / no tfhd"
                                    errors we used to see.
    Falls back to manual segment concat only if ffmpeg isn't on PATH —
    that path is good enough for MP3 HLS and fails loudly for fMP4."""
    if ffmpeg_available():
        await _download_hls_via_ffmpeg(m3u8_url, out_path)
        return
    await _download_hls_manual_concat(http, m3u8_url, out_path)


async def _download_hls_via_ffmpeg(m3u8_url: str, out_path: Path) -> None:
    """Use ffmpeg to download + remux an HLS stream directly. ffmpeg
    natively handles `#EXT-X-MAP` init segments, which our manual concat
    skips, so this is the only path that works for SC's AAC fMP4 streams.
    Run via to_thread so subprocess wait doesn't block the loop."""
    args = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-user_agent", DEFAULT_HEADERS["User-Agent"],
        "-i", m3u8_url,
        "-c", "copy",
        # `aac_adtstoasc` is a no-op for non-AAC; for AAC streams it
        # rewrites ADTS frames into ASC (MP4-compatible), needed when
        # remuxing into an .m4a container.
        "-bsf:a", "aac_adtstoasc",
        str(out_path),
    ]

    def _run() -> tuple[int, bytes]:
        proc = subprocess.run(
            args, capture_output=True, timeout=300, check=False,
        )
        return proc.returncode, proc.stderr or b""

    try:
        rc, stderr = await asyncio.to_thread(_run)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise ProviderError(f"sc ffmpeg HLS failed: {e}") from e
    if rc != 0:
        msg = stderr.decode("utf-8", "replace")[-300:].strip()
        raise ProviderError(f"sc ffmpeg HLS exit {rc}: {msg}")


async def _download_hls_manual_concat(
    http: aiohttp.ClientSession, m3u8_url: str, out_path: Path,
) -> None:
    """Naive segment concat — fallback for environments without ffmpeg.
    Works for MP3 HLS (single-codec, no init segment); blows up loudly
    on AAC fMP4 HLS streams that need ffmpeg's init-segment handling."""
    async with http.get(m3u8_url, headers=DEFAULT_HEADERS) as r:
        if r.status != 200:
            raise ProviderError(f"sc m3u8 {r.status}")
        playlist = await r.text()

    if "#EXT-X-MAP" in playlist:
        raise ProviderError(
            "ffmpeg required for fMP4 HLS streams (install ffmpeg or "
            "lower SoundCloud quality preference to MP3)"
        )

    segments = [line.strip() for line in playlist.splitlines()
                if line.strip() and not line.startswith("#")]
    if not segments:
        raise ProviderError("empty hls playlist")

    with out_path.open("wb") as fh:
        for seg_url in segments:
            async with http.get(seg_url, headers=DEFAULT_HEADERS) as r:
                if r.status != 200:
                    raise ProviderError(f"sc hls segment {r.status}")
                async for chunk in r.content.iter_chunked(64 * 1024):
                    fh.write(chunk)


def _ffmpeg_available() -> bool:
    """Backwards-compat shim. Use `ffmpeg_available()` from
    core.audio_convert directly in new code."""
    return ffmpeg_available()
