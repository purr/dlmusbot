"""Cross-provider track fallback.

When a track's source provider can't deliver it (DRM-locked SoundCloud,
Go+ snippet-only, region-blocked, etc.), we search the other registered
providers for the same artist + title and pick the closest match by
fuzzy score + duration. Used by both the inline URL handler (so the
user sees a usable result instead of "couldn't load") and the JobRunner
(so a clicked URL still delivers audio via the alt provider).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .fuzz import score as fuzz_score
from .models import Track

if TYPE_CHECKING:
    from providers.base import Provider
    from providers.registry import Registry

log = logging.getLogger(__name__)


# Permanent ProviderError reasons that should trigger fallback rather
# than a hard failure. DRM (CommonEncryption HLS), Go+ (snippet only)
# and unavailable (region-lock / removed) all mean "this provider can't
# deliver, but the song might exist elsewhere as a free upload."
FALLBACK_REASONS: frozenset[str] = frozenset({"drm", "goplus", "unavailable"})

# Fuzzy-match floor on "<artist> <title>". 75 tolerates "feat." /
# remaster / punctuation differences without letting unrelated tracks
# slip through.
FALLBACK_MIN_SCORE: float = 80.0

# Hard duration delta (seconds) between original and candidate. SC
# uploads frequently differ by a few seconds of intro/outro silence vs
# the official release; 8s catches most without matching different
# recordings entirely.
FALLBACK_DURATION_TOLERANCE_S: int = 8

# Cross-provider fallback whitelist. Only these providers are searched
# for an equivalent recording when the original source can't deliver.
# YouTube Music is intentionally excluded — its catalogue is full of
# random user uploads (lyric videos, sped-up edits, low-quality rips)
# that would slip past the score+duration gate too easily.
FALLBACK_CROSS_PROVIDERS: frozenset[str] = frozenset({"spotify"})


# Recursion cap on fallback chain. Three attempts (original + 2 fallbacks)
# is plenty — beyond that we're matching increasingly unrelated tracks and
# should surface the failure to the user instead of pinning the queue.
MAX_FALLBACK_DEPTH: int = 3


async def find_alternative_track(
    registry: "Registry",
    original: Track,
    *,
    tried: Optional[set[str]] = None,
    tried_track_ids: Optional[set[str]] = None,
) -> Optional[tuple["Provider", Track]]:
    """Find any provider+track that plausibly matches `original`.

    Two-stage search:
      1. Same provider, different upload — when SC's official track is
         DRM-locked, there are usually re-uploads by other users that
         are plain progressive MP3. Stays on the user's platform.
      2. Cross-provider — Spotify / YouTube Music for the same artist
         + title.

    Acceptance per candidate:
      - fuzzy score on "artist title" >= FALLBACK_MIN_SCORE
      - duration within FALLBACK_DURATION_TOLERANCE_S of original
      - candidate.track_id != original.track_id
      - candidate.extra["is_drm_only"] / ["is_goplus"] both false
    """
    tried = tried or set()
    tried_track_ids = tried_track_ids or set()
    artist = (original.artists_str or "").strip()
    title = (original.title or "").strip()
    query = f"{artist} {title}".strip()
    if not query:
        return None

    # Exclude both the original and any previously-attempted track ids so
    # cascading failures can't loop on the same alt upload.
    exclude_ids = set(tried_track_ids) | {original.track_id}

    # Stage 1: same provider, alt upload.
    original_prov = registry.get(original.provider)
    if original_prov is not None:
        pick = await _pick_match(
            original_prov,
            original,
            query,
            exclude_track_ids=exclude_ids,
        )
        if pick is not None:
            return original_prov, pick

    # Stage 2: cross-provider, restricted to the whitelist.
    for prov in registry.all():
        if prov.name == original.provider or prov.name in tried:
            continue
        if prov.name not in FALLBACK_CROSS_PROVIDERS:
            continue
        pick = await _pick_match(
            prov, original, query, exclude_track_ids=exclude_ids,
        )
        if pick is not None:
            return prov, pick
    return None


# Title-token markers we treat as "different recording from the
# original". When the original query doesn't contain any of these but a
# candidate's title does, we skip it — duration alone isn't enough to
# tell "RIVER FLOWS IN YOU" apart from "RIVER FLOWS IN YOU (Sped Up)"
# since both can clock the same length.
_VARIANT_TOKENS = frozenset({
    "remix", "cover", "live", "acoustic", "edit", "flip", "bootleg",
    "remastered", "remaster", "instrumental", "karaoke", "sped",
    "slowed", "reverb", "8d", "loop", "snippet", "intro", "outro",
    "demo",
})


def _has_variant_marker(text: str) -> bool:
    lowered = (text or "").lower()
    return any(tok in lowered for tok in _VARIANT_TOKENS)


async def _pick_match(
    prov: "Provider",
    original: Track,
    query: str,
    *,
    exclude_track_ids: Optional[set[str]] = None,
) -> Optional[Track]:
    try:
        results = await prov.search(query, limit=10)
    except Exception:
        log.exception("[fallback] %s.search failed for %r", prov.name, query)
        return None
    if not results:
        return None
    excluded = exclude_track_ids or set()
    # Variant gate: if the original title has no remix/cover/etc marker,
    # reject candidates that introduce one. Same-duration variants
    # (sped-up, slowed, edits) would otherwise sneak through the
    # duration tolerance.
    original_has_variant = _has_variant_marker(original.title)
    best: Optional[tuple[float, Track]] = None
    for t in results:
        if t.track_id in excluded:
            continue
        extra = t.extra or {}
        if extra.get("is_drm_only") or extra.get("is_goplus"):
            continue
        if not original_has_variant and _has_variant_marker(t.title):
            continue
        s = fuzz_score(query, t)
        if s < FALLBACK_MIN_SCORE:
            continue
        if original.duration_seconds > 0 and t.duration_seconds > 0:
            if (
                abs(t.duration_seconds - original.duration_seconds)
                > FALLBACK_DURATION_TOLERANCE_S
            ):
                continue
        if best is None or s > best[0]:
            best = (s, t)
    if best is not None:
        log.info(
            "[fallback] %s match for %r: score=%.1f dur_delta=%ds -> %s",
            prov.name,
            query,
            best[0],
            abs(best[1].duration_seconds - original.duration_seconds),
            best[1].display_title,
        )
        return best[1]
    return None
