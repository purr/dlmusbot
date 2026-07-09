"""Multi-provider search-result merging + fuzzy track matching.

Search results are NOT re-scored locally — each provider's own ranking
is trusted (see `interleave_by_provider` for why); the inline pipeline
is dedupe (drop cross-provider re-uploads, keep the authoritative copy)
+ round-robin interleave.

`score` serves track-to-track matching in `core.fallback`: given a
known track's "artist title" string, pick the best-matching candidate
from another provider's search results. It scores artist and title as
separate fields and adds a token-coverage factor so a candidate that
contains *every* query token beats one that only contains some.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable

from rapidfuzz import fuzz

from .models import Track


# Boilerplate that uploaders pad track titles with — strip before scoring
# so "Fever (Official Audio)" and "Fever" match identically. Lower-case
# input; pattern is forgiving with brackets/parens and surrounding space.
_TITLE_NOISE_RE = re.compile(
    r"\s*[\(\[\{]\s*(?:"
    # "Official ..." variants — most common YouTube/SC convention.
    r"official\s*(?:audio|video|music\s*video|lyric\s*video|mv)?"
    # Bare medium descriptors.
    r"|audio|video|mv|lyric[s]?|lyric\s*video|music\s*video|visualizer"
    # Quality descriptors.
    r"|hd|hq|4k|8k|1080p|720p|hi[\s\-]?res"
    # Release descriptors.
    r"|remaster(?:ed)?(?:\s*\d{4})?|reissue|deluxe|expanded|anniversary"
    r"|explicit|clean|radio\s*edit|extended|full\s*version|single\s*version"
    # "with X" upload metadata (lyrics, subs, captions, etc.).
    r"|with\s*(?:lyrics?|subtitles?|captions?|audio|video|vocals?|sound|music)"
    # Live / acoustic / session tags that uploaders treat as noise but
    # we DO want to keep — left out intentionally (those are real
    # versions of a song, not metadata).
    r")\s*[\)\]\}]\s*",
    re.IGNORECASE,
)
# Bracketed feat clause: "(feat. X)", "[ft. X & Y]", "(with X)", etc.
# `with` is accepted *only* in this bracketed form — bare `with` in a
# title overwhelmingly means the English preposition (e.g. "Dancing
# with Myself", "Stay with Me"), not a featured-artist marker.
_FEAT_BRACKETED_RE = re.compile(
    r"[\(\[\{]\s*(?:feat\.?|ft\.?|featuring|with)\s+"
    r"(?P<names>[^\)\]\}]+?)\s*[\)\]\}]",
    re.IGNORECASE,
)
# Bare feat clause: "Title - feat. X", "Title ft. X", trailing or
# followed by a dash/pipe. Bare `with` intentionally excluded.
_FEAT_BARE_RE = re.compile(
    r"(?:^|\s|-)\s*(?:feat\.?|ft\.?|featuring)\s+"
    r"(?P<names>[^\(\[\)\]\-|]+?)\s*(?:[-|]|$)",
    re.IGNORECASE,
)
# Split feat-artist lists on common separators: " & ", " x ", " X ",
# " and ", ", " — so "feat. Alice & Bob, Carol" yields three names.
_FEAT_SPLIT_RE = re.compile(
    r"\s*(?:,|&|\bx\b|\band\b)\s*", re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _extract_feat(s: str) -> list[str]:
    """Return featured-artist names parsed out of a string (typically a
    track title). Empty list when no feat/ft/featuring/(with) clause.

    Noise brackets like "(With Lyrics)" or "(with subtitles)" are
    stripped first via `_TITLE_NOISE_RE` so they can never be misread
    as feat clauses — single source of truth for the noise list."""
    if not s:
        return []
    cleaned_input = _TITLE_NOISE_RE.sub(" ", s)
    out: list[str] = []
    for rx in (_FEAT_BRACKETED_RE, _FEAT_BARE_RE):
        for m in rx.finditer(cleaned_input):
            chunk = (m.group("names") or "").strip()
            if not chunk:
                continue
            for name in _FEAT_SPLIT_RE.split(chunk):
                name = name.strip()
                if name:
                    out.append(name)
    return out


def _clean(s: str) -> str:
    """Lowercase, strip diacritics, strip uploader noise + featured-
    artist tail, normalise separator punctuation, and drop apostrophes
    so "dont stop" matches "Don't Stop" (mobile users routinely skip
    the apostrophe).

    Diacritic-stripping (NFKD decomposition + ASCII filter) means that
    "Björk" and "Bjork", "café" and "cafe", "naïve" and "naive" all
    score identically — users typing without an accented keyboard
    still find the canonical track.

    The featured-artist names are still recoverable via `_extract_feat`
    so they can be folded into the artist token pool for scoring.
    """
    if not s:
        return ""
    # NFKD decomposes accented chars into base + combining mark
    # (e.g. "é" -> "e" + combining acute); the ascii filter then
    # drops the combining marks. Result: "björk" -> "bjork".
    out = unicodedata.normalize("NFKD", s)
    out = out.encode("ascii", "ignore").decode("ascii")
    out = out.lower()
    out = _FEAT_BRACKETED_RE.sub(" ", out)
    out = _FEAT_BARE_RE.sub(" ", out)
    out = _TITLE_NOISE_RE.sub(" ", out)
    # Apostrophes deleted (not replaced with space) so contractions
    # collapse into one token: "don't" -> "dont", same as user typed.
    out = out.replace("'", "").replace("’", "")
    # Separator punctuation -> space.
    out = re.sub(r"[,;:.!?/\\]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


_FUZZY_TOKEN_MIN_LEN = 4
_FUZZY_TOKEN_RATIO_THRESHOLD = 85
# Partial-match weight. An exact token match earns the full 1.0 toward
# coverage; substring + fuzzy matches earn this fraction. Without this
# weighting, query "love" gets coverage 1.0 against title "Lover Boy"
# (since "love" is a substring of "lover"), tying real "Love Me" hits
# at the top. 0.5 keeps partial matches relevant without letting them
# outrank actual exact matches.
_PARTIAL_MATCH_WEIGHT = 0.5


def _match_weight(
    needle: str, haystack_tokens: set[str], *, allow_fuzzy: bool = True,
) -> float:
    """Weight a single string earns against the haystack token set:
      * `1.0`  — exact: `needle` is one of the haystack tokens.
      * `0.5`  — `needle` (len ≥ 4) is a substring of some haystack token,
                 OR (when `allow_fuzzy`) fuzzy-matches one at `ratio ≥ 85`.
      * `0.0`  — no match (or too short for the partial tiers).

    The `len >= 4` gate on the partial tiers is what stops a 3-char query
    like `"rap"` from cross-matching `"rapture"` / `"rapid"`.

    `allow_fuzzy=False` is used by the consecutive-merge pass: concatenating
    an already-exact token with a short neighbour ("buckshot"+"fe") produces
    a string that fuzzy-matches the exact token itself (`ratio ~89`), which
    would leak partial credit onto the fragment. Exact + substring only on
    merges sidesteps that — a split word still recovers via its exact join
    ("fe"+"ver" = "fever")."""
    if needle in haystack_tokens:
        return 1.0
    if len(needle) < _FUZZY_TOKEN_MIN_LEN:
        return 0.0
    if any(needle in ht for ht in haystack_tokens):
        return _PARTIAL_MATCH_WEIGHT
    if allow_fuzzy and any(
        fuzz.ratio(needle, ht) >= _FUZZY_TOKEN_RATIO_THRESHOLD
        for ht in haystack_tokens
        if len(ht) >= _FUZZY_TOKEN_MIN_LEN
    ):
        return _PARTIAL_MATCH_WEIGHT
    return 0.0


def _coverage(query_tokens, haystack_tokens: set[str]) -> float:
    """Weighted fraction of query tokens present in the haystack, in [0, 1].

    Each query token earns a weight via `_match_weight` (exact `1.0`,
    substring / fuzzy `0.5`). On top of that, a **consecutive-merge** pass
    handles the case where a user split one word across spaces: the query
    `"fe ver"` should match the haystack token `"fever"`. For every run of
    adjacent query tokens we concatenate them and, when the joined string
    matches a haystack token, upgrade each constituent token's weight to
    that match's weight (taking the max, so a merge can only ever help).

    `"buckshot fe ver fak"` against `"buckshot fakemink fever"`:
      * per-token — `buckshot` exact (`1.0`); `fe` / `ver` / `fak` too
        short to match anything → `0.0` each.
      * merge — `"fe"+"ver"` = `"fever"`, an exact haystack token, so `fe`
        and `ver` are upgraded to `1.0`.
      * result — `(1 + 1 + 1 + 0) / 4 = 0.75`, enough to lift the real
        "Fever" hit above unrelated "Buckshot …" tracks that only cover
        the single `buckshot` token.

    `query_tokens` must be an ordered sequence (the merge pass relies on
    adjacency); `score` passes the de-duplicated query tokens in order.
    The denominator is the token count, so the merge never inflates
    coverage past 1.0.
    """
    qt_list = list(query_tokens)
    n = len(qt_list)
    if n == 0:
        return 0.0

    weights = [_match_weight(qt, haystack_tokens) for qt in qt_list]

    # Consecutive-merge pass — only meaningful for multi-token queries.
    # Window length capped implicitly by the query size (always small).
    for i in range(n):
        for j in range(i + 1, n):
            concat = "".join(qt_list[i : j + 1])
            if len(concat) < _FUZZY_TOKEN_MIN_LEN:
                continue
            w = _match_weight(concat, haystack_tokens, allow_fuzzy=False)
            if w <= 0.0:
                continue
            for k in range(i, j + 1):
                if w > weights[k]:
                    weights[k] = w

    return sum(weights) / n


def score(query: str, track: Track) -> float:
    """0..100 — higher is a better match.

    Two signals combined:
      * **WRatio base** — rapidfuzz's general-purpose match score over
        title, artist, and the combined string. Takes the strongest of
        the three so a hit in any field counts.
      * **Coverage** — fraction of query tokens that appear in the
        haystack (exact or 4+ char substring). Multi-token queries
        demand most of their tokens be present; single-token typos
        survive at half score via the coverage floor.

    Final = `base * (0.5 + 0.5 * coverage)`. Coverage of 1.0 keeps the
    full WRatio; coverage of 0 still leaves 50% so typos like "feever"
    against "Fever" don't get nuked to zero. Junk that has both low
    WRatio AND low coverage naturally falls below the caller's cutoff
    (`FALLBACK_MIN_SCORE` in `core.fallback`).
    """
    if not query:
        return 0.0
    q = _clean(query)
    if not q:
        return 0.0
    title = _clean(track.title)
    # Fold featured artists from the *raw* title into the artist pool,
    # so a Spotify track "Buckshot — Fever (feat. Fakemink)" scores the
    # same on "fakemink buckshot" as a co-credit "Buckshot, Fakemink"
    # would. Without this, providers that bury collaborators in the
    # title get unfairly demoted.
    feat_names = _extract_feat(track.title)
    base_artist = _clean(track.artists_str)
    artist_parts = [p for p in [base_artist] + [_clean(n) for n in feat_names] if p]
    artist = " ".join(artist_parts)
    combined = f"{artist} {title}".strip()

    title_s = fuzz.WRatio(q, title) if title else 0.0
    artist_s = fuzz.WRatio(q, artist) if artist else 0.0
    combined_s = fuzz.WRatio(q, combined) if combined else 0.0
    # Best of the three: per-field averaged vs. the combined string.
    # The combined score usually wins, but per-field protects against
    # cases where one field dominates the query.
    base = max((title_s + artist_s) / 2, combined_s)

    # Coverage floor: scale base by (0.5 + 0.5 * coverage). Multi-token
    # queries that fully match keep the full base; queries that miss
    # tokens get squashed proportionally; a single-token typo with no
    # token overlap (e.g. "feever" vs "Fever") survives at 0.5*base so
    # the obvious target isn't nuked from the result set entirely.
    # Ordered, de-duplicated query tokens — order matters for the
    # consecutive-merge pass in `_coverage` (split-word recovery), and
    # de-duplication keeps the denominator equal to the distinct-token
    # count so repeats ("la la la") don't dilute coverage.
    q_tok = list(dict.fromkeys(_TOKEN_RE.findall(q)))
    coverage = _coverage(q_tok, _tokens(combined))
    return base * (0.5 + 0.5 * coverage)


# Authority order for cross-provider dedupe (which copy of a duplicated
# track survives) AND for who takes the first slot of each interleave
# round. Single source of truth for every pass below.
PROVIDER_PRIORITY = ("spotify", "soundcloud", "youtube_music")


def _provider_rank(prefer: tuple[str, ...]) -> Callable[[str], int]:
    """Rank function over `prefer`: index in the tuple; providers not
    listed rank after all listed ones."""
    idx = {p: i for i, p in enumerate(prefer)}

    def rank(provider: str) -> int:
        return idx.get(provider, len(prefer))

    return rank


def _dedupe_key(t: Track) -> str:
    """Normalised (artists + title) used to detect cross-provider duplicates.
    Lowercased, whitespace collapsed, punctuation stripped — so "Track" /
    "track." / "TRACK" all collapse onto the same bucket."""
    raw = f"{_clean(t.artists_str)} {_clean(t.title)}"
    key = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", raw)).strip()
    if key:
        return key
    # `_clean` ASCII-strips, so fully non-Latin metadata (Cyrillic, CJK,
    # Arabic, ...) cleans to "". Build the key from the raw text instead
    # — casefolded, punctuation stripped, Unicode preserved — so those
    # tracks still dedupe cross-provider instead of being dropped.
    raw = f"{t.artists_str} {t.title}".casefold()
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", raw)).strip()


def dedupe_tracks(
    tracks: Iterable[Track],
    *, prefer: tuple[str, ...] = PROVIDER_PRIORITY,
) -> list[Track]:
    """Drop duplicate (artist, title) hits across providers.

    When the same song surfaces from multiple catalogues, keep the one
    from the highest-priority provider listed in `prefer`. Order of the
    remaining items is preserved (stable). Useful right before
    `interleave_by_provider` so the round-robin doesn't waste a slot on
    a SoundCloud-uploaded copy of a Spotify single."""
    rank = _provider_rank(prefer)
    best: dict[str, tuple[int, int, Track]] = {}
    for idx, t in enumerate(tracks):
        key = _dedupe_key(t)
        if not key:
            continue
        prov_rank = rank(t.provider)
        cur = best.get(key)
        if cur is None or prov_rank < cur[0]:
            best[key] = (prov_rank, idx, t)
    # Restore original input order among the kept entries.
    return [t for _, _, t in sorted(best.values(), key=lambda x: x[1])]


# Tolerances for `dedupe_near_duplicates`. Tuned conservatively: a
# 2-second duration delta covers the typical SC re-upload (slight
# intro/outro trim or transcoding rounding); 80 on `fuzz.ratio` catches
# "march madness" inside "march madness (MV IN DESC)" without folding
# genuinely different tracks together.
_NEAR_DUP_DURATION_TOL_S = 2
_NEAR_DUP_TITLE_RATIO = 80

# Guards for `dedupe_embedded_title`. The embedded-title heuristic is
# strong (it needs the authoritative track's artist AND title to both
# appear verbatim inside the other track's title, at the same duration),
# but a 1-2 char artist or title could coincidentally substring-match an
# unrelated track that happens to share a runtime. Require a little
# substance on both before trusting the match.
_EMBED_MIN_ARTIST_LEN = 2
_EMBED_MIN_TITLE_LEN = 3


def _primary_artist_clean(t: Track) -> str:
    """Normalised name of the first artist. Empty when no artists."""
    if not t.artists:
        return ""
    return _clean(t.artists[0].name)


def _titles_overlap(a: str, b: str) -> bool:
    """True when one cleaned title contains the other OR they fuzzy-
    match closely. Handles the "(MV IN DESC)" / "[OFFICIAL FREE DL]" /
    other uploader-noise tails that we can't enumerate in the noise
    regex — same song, different decoration."""
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return fuzz.ratio(a, b) >= _NEAR_DUP_TITLE_RATIO


def dedupe_near_duplicates(
    tracks: Iterable[Track],
    *, prefer: tuple[str, ...] = PROVIDER_PRIORITY,
) -> list[Track]:
    """Second-pass dedupe for tracks that exact-match missed.

    Two tracks are near-duplicates when ALL of:
      * Same primary artist (cleaned, case-insensitive).
      * Durations differ by ≤ `_NEAR_DUP_DURATION_TOL_S` (both must
        report a non-zero duration — a missing duration disqualifies
        the pair, so we never collapse unrelated tracks just because
        one provider didn't report length).
      * Cleaned titles overlap: one contains the other, OR
        `fuzz.ratio` ≥ `_NEAR_DUP_TITLE_RATIO`.

    Keeps the highest-priority provider per cluster. Preserves input
    order among survivors so `interleave_by_provider` downstream still
    sees the same order it would have seen otherwise.

    Real case this catches: SoundCloud "myspacemark — march madness
    (MV IN DESC)" + Spotify "myspacemark — march madness", same
    duration. After this pass only the Spotify version remains.
    """
    track_list = list(tracks)
    if len(track_list) < 2:
        return track_list

    rank = _provider_rank(prefer)
    keep = [True] * len(track_list)
    artists = [_primary_artist_clean(t) for t in track_list]
    titles = [_clean(t.title) for t in track_list]

    for i in range(len(track_list)):
        if not keep[i]:
            continue
        ti = track_list[i]
        ti_dur = ti.duration_seconds
        if not artists[i] or ti_dur <= 0:
            continue
        for j in range(i + 1, len(track_list)):
            if not keep[j]:
                continue
            tj = track_list[j]
            if not tj.duration_seconds:
                continue
            if abs(ti_dur - tj.duration_seconds) > _NEAR_DUP_DURATION_TOL_S:
                continue
            if artists[j] != artists[i]:
                continue
            if not _titles_overlap(titles[i], titles[j]):
                continue
            # Confirmed near-duplicate — keep the higher-priority
            # provider, drop the other. On tie, keep the earlier
            # input-order entry (so order is deterministic).
            pi = rank(ti.provider)
            pj = rank(tj.provider)
            if pi <= pj:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [t for t, k in zip(track_list, keep) if k]


def _title_embeds_track(reup_title_clean: str, authoritative: Track) -> bool:
    """True when `authoritative`'s primary artist AND title both appear,
    verbatim, inside `reup_title_clean` (an already-`_clean`'d title).

    Models the common SoundCloud re-upload shape: a Spotify single
    "Psychonaut 4 — Suicide Is Legal" re-posted by some random uploader
    as a track titled "Psychonaut 4 - Suicide Is Legal" (uploader name in
    the artist field, original artist + song folded into the title). The
    same-artist near-dup pass misses these because the artist field no
    longer matches; this catches them by the title fingerprint instead.

    Kept dynamic — no hard-coded names. Both fields must clear a minimum
    length so a trivially-short artist/title can't coincidentally match.
    """
    artist = _primary_artist_clean(authoritative)
    title = _clean(authoritative.title)
    if len(artist) < _EMBED_MIN_ARTIST_LEN or len(title) < _EMBED_MIN_TITLE_LEN:
        return False
    return artist in reup_title_clean and title in reup_title_clean


def dedupe_embedded_title(
    tracks: Iterable[Track],
    *, prefer: tuple[str, ...] = PROVIDER_PRIORITY,
) -> list[Track]:
    """Third-pass dedupe for cross-provider re-uploads whose ARTIST field
    differs but whose TITLE embeds the authoritative track's artist+song.

    Two tracks collapse when ALL of:
      * Both report a non-zero duration differing by
        ≤ `_NEAR_DUP_DURATION_TOL_S`.
      * They come from different-priority providers (so one is clearly the
        authoritative source — Spotify over SoundCloud over YT Music).
      * The lower-priority track's cleaned title contains BOTH the
        higher-priority track's cleaned primary-artist name and its
        cleaned title (see `_title_embeds_track`).

    The authoritative (higher-priority) track is always kept; the
    re-upload is dropped. Order is preserved among survivors.

    Real case this catches that `dedupe_near_duplicates` cannot: Spotify
    "Psychonaut 4 — Suicide Is Legal" (388s) and SoundCloud
    "GIORGI SANIKIDZE — Psychonaut 4 - Suicide Is Legal" (388s). The
    artist fields differ, so the same-artist pass keeps both; here the SC
    title fingerprint matches the Spotify artist+title and the SC copy is
    dropped in favour of Spotify.
    """
    track_list = list(tracks)
    if len(track_list) < 2:
        return track_list

    rank = _provider_rank(prefer)
    keep = [True] * len(track_list)

    for i in range(len(track_list)):
        if not keep[i]:
            continue
        ti = track_list[i]
        ti_dur = ti.duration_seconds
        if ti_dur <= 0:
            continue
        pi = rank(ti.provider)
        for j in range(i + 1, len(track_list)):
            if not keep[j]:
                continue
            tj = track_list[j]
            if tj.duration_seconds <= 0:
                continue
            if abs(ti_dur - tj.duration_seconds) > _NEAR_DUP_DURATION_TOL_S:
                continue
            pj = rank(tj.provider)
            if pi == pj:
                # Same provider tier — neither is the authoritative source,
                # so the "prefer Spotify" direction is undefined. Skip.
                continue
            if pi < pj:
                keeper, reup, reup_idx = ti, tj, j
            else:
                keeper, reup, reup_idx = tj, ti, i
            if _title_embeds_track(_clean(reup.title), keeper):
                keep[reup_idx] = False
                if reup_idx == i:
                    break
    return [t for t, k in zip(track_list, keep) if k]


def interleave_by_provider(
    tracks: Iterable[Track],
    limit: int = 20,
    *,
    provider_priority: tuple[str, ...] = PROVIDER_PRIORITY,
) -> list[Track]:
    """Round-robin merge of already-ranked provider results.

    Each provider's search engine has ranked its own list with semantic
    knowledge a lexical matcher lacks (stylised spellings like "Ke$ha",
    aliases, slang, popularity), so that order is trusted as-is — no
    local re-scoring, no dropping. One track per provider per turn:
    Spotify's #1, SoundCloud's #1, Spotify's #2, ... `provider_priority`
    decides who takes the first slot of each round; providers not listed
    take their turn after the named ones (alphabetical, deterministic).
    When one provider's queue runs dry the remaining ones keep rotating.

    Input order doubles as provider rank: `tracks` is the (deduped)
    concatenation of per-provider result lists, and every dedupe pass is
    stable, so each provider's tracks are still in that provider's own
    ranking order here.
    """
    rank = _provider_rank(provider_priority)
    queues: dict[str, list[Track]] = {}
    for t in tracks:
        queues.setdefault(t.provider, []).append(t)
    order = sorted(queues, key=lambda p: (rank(p), p))
    out: list[Track] = []
    while len(out) < limit and any(queues.values()):
        for prov in order:
            queue = queues[prov]
            if not queue:
                continue
            out.append(queue.pop(0))
            if len(out) >= limit:
                break
    return out
