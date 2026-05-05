"""Fuzzy match scoring for unifying multi-provider search results.

Each provider returns its own ranked list. We re-rank the merged set by how
well "<artist> <title>" matches the user query, so the top of the list reflects
the user's typed words rather than provider-internal popularity heuristics.
"""

from __future__ import annotations

from typing import Iterable

from rapidfuzz import fuzz

from .models import Track


def score(query: str, track: Track) -> float:
    """0..100 — higher is a better match. Combines partial + token-set ratios
    so word-order variations and missing words don't tank the score."""
    if not query:
        return 0.0
    haystack = f"{track.artists_str} {track.title}".lower()
    q = query.lower()
    return max(
        fuzz.token_set_ratio(q, haystack),
        fuzz.partial_ratio(q, haystack),
    )


def rank(query: str, tracks: Iterable[Track], limit: int = 20) -> list[Track]:
    """Sort tracks by descending fuzzy score, keep top `limit`. Ties keep
    original order (stable sort)."""
    scored = [(score(query, t), t) for t in tracks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit]]


def _dedupe_key(t: Track) -> str:
    """Normalised (artists + title) used to detect cross-provider duplicates.
    Lowercased, whitespace collapsed, punctuation stripped — so "Track" /
    "track." / "TRACK" all collapse onto the same bucket."""
    import re
    raw = f"{t.artists_str} {t.title}".lower()
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", raw)).strip()


def dedupe_tracks(
    tracks: Iterable[Track],
    *, prefer: tuple[str, ...] = ("spotify", "soundcloud", "youtube_music"),
) -> list[Track]:
    """Drop duplicate (artist, title) hits across providers.

    When the same song surfaces from multiple catalogues, keep the one
    from the highest-priority provider listed in `prefer`. Order of the
    remaining items is preserved (stable). Useful right before
    `rank_balanced` so the round-robin doesn't waste a slot on a
    SoundCloud-uploaded copy of a Spotify single."""
    rank_by_provider = {p: i for i, p in enumerate(prefer)}
    fallback_rank = len(prefer)
    best: dict[str, tuple[int, int, Track]] = {}
    for idx, t in enumerate(tracks):
        key = _dedupe_key(t)
        if not key:
            continue
        prov_rank = rank_by_provider.get(t.provider, fallback_rank)
        cur = best.get(key)
        if cur is None or prov_rank < cur[0]:
            best[key] = (prov_rank, idx, t)
    # Restore original input order among the kept entries.
    return [t for _, _, t in sorted(best.values(), key=lambda x: x[1])]


def rank_balanced(
    query: str, tracks: Iterable[Track], limit: int = 20,
    *, min_score: float = 30.0, bucket_size: float = 4.0,
) -> list[Track]:
    """Score-first ranking with **soft** provider interleaving.

    Items are ordered by descending fuzzy score. Within each "tie bucket"
    (group of consecutive items whose scores fall inside `bucket_size`
    of each other) the providers are round-robin'd, so a clearly-better
    Spotify hit still beats a mediocre SoundCloud one, but two
    near-identically-scored hits from the same provider get a SoundCloud
    pick wedged between them for variety.

    Effect: high-confidence matches keep their natural order; only
    fuzzy-tied stretches get the platform-mixing nudge. Items scoring
    below `min_score` are dropped — they're noise."""
    scored: list[tuple[float, Track]] = []
    for t in tracks:
        s = score(query, t)
        if s >= min_score:
            scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[Track] = []
    i = 0
    while i < len(scored) and len(out) < limit:
        # Collect the next bucket: consecutive items within `bucket_size`
        # of the bucket's leading score.
        top = scored[i][0]
        bucket: list[tuple[float, Track]] = []
        while i < len(scored) and (top - scored[i][0]) <= bucket_size:
            bucket.append(scored[i])
            i += 1
        if len(bucket) <= 1:
            out.append(bucket[0][1])
            continue
        # Round-robin within the bucket by provider, preserving the
        # in-bucket score order inside each provider's queue.
        by_prov: dict[str, list[Track]] = {}
        for s, t in bucket:
            by_prov.setdefault(t.provider, []).append(t)
        while by_prov and len(out) < limit:
            for prov in list(by_prov.keys()):
                if not by_prov[prov]:
                    del by_prov[prov]
                    continue
                out.append(by_prov[prov].pop(0))
                if not by_prov[prov]:
                    del by_prov[prov]
                if len(out) >= limit:
                    break
    return out
