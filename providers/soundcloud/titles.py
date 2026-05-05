"""Best-effort cleaner for SoundCloud track titles.

SoundCloud titles are user-controlled freeform strings — common patterns:
    "Artist - Title"
    "Artist - Title [Free DL]"
    "Title (Artist Remix)"
    "Title | Skip to 0:42"
    "★ Title ★"

This module strips junk markers and can remove redundant artist prefixes
(`Artist - Title`) only when the known artist name matches exactly.
"""

from __future__ import annotations

import re
from typing import Optional

# Markers we always remove (case-insensitive).
_NOISE_PATTERNS = [
    re.compile(r"\[\s*free\s*(?:dl|download)\s*\]", re.I),
    re.compile(r"\(\s*free\s*(?:dl|download)\s*\)", re.I),
    re.compile(r"\bfree\s*(?:dl|download)\b\s*[!]*", re.I),
    re.compile(r"[★☆♪♫➤►▶]+"),
    re.compile(r"\bout\s*now\b\s*[!]*", re.I),
    re.compile(r"\b(?:hq|hd)\b", re.I),
    re.compile(r"[\(\[\{]\s*[.):/\-]*\s*[\)\]\}]"),  # leftover empty bracket
]

# "skip to" appears in many user variants; we only target likely timestamp
# annotations and keep surrounding title text.
_NUM_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten)"
_TIME_EXPR = (
    r"(?:"
    r"\d{1,2}[:/]\d{2}(?::\d{2})?"                                   # 1:23, 00/16, 1:02:33
    r"|:\d{2}"                                                       # :50
    r"|(?:a|an)\s*(?:minutes?|mins?|min|seconds?|secs?|sec|s)\b"     # a minute / a min
    r"|(?:\d+|" + _NUM_WORD + r")\s*(?:minutes?|mins?|min|m)\.?\b"  # 1 min, one minute
    r"(?:\s*(?:and\s*)?(?:\d+|" + _NUM_WORD + r")\s*(?:seconds?|secs?|sec|s)\.?\b)?"
    r"|(?:\d+|" + _NUM_WORD + r")\s*(?:seconds?|secs?|sec|s)\.?\b"   # 45 sec
    r")"
)
_SKIP_TO_PATTERNS = [
    # Bracketed forms:
    #   [skip to 1:23], (AI REMASTER skip2 00:45), {skip to one minute}
    re.compile(
        r"[\[\(\{][^\]\)\}]{0,120}?"
        r"skip\s*(?:to|2)?\s*:?\s*" + _TIME_EXPR +
        r"[^\]\)\}]{0,120}[\]\)\}]",
        re.I,
    ),
    # Inline forms:
    #   | skip to 1:23
    #   - skip to 2m 03s
    #   skip2 45s / skip to one minute
    re.compile(
        r"(?:^|[\s|,;:*()\/\[\]{}._~〜\-–—☆★♪♫➤►▶])"
        r"skip\s*(?:to|2|till?|til)?\s*"
        r":?\s*" + _TIME_EXPR +
        r"(?:\s*(?:\.mp3|\.wav))?",
        re.I,
    ),
    # Reversed order seen in the wild:
    #   (1 min skip), (a minute skip), "2:10 skip"
    re.compile(
        r"[\[\(\{][^\]\)\}]{0,80}?"
        + _TIME_EXPR +
        r"\s*skip(?:\s*(?:to|2))?"
        r"[^\]\)\}]{0,80}[\]\)\}]",
        re.I,
    ),
    re.compile(
        r"(?:^|[\s|,;:*()\/\[\]{}._~〜\-–—☆★♪♫➤►▶])"
        + _TIME_EXPR +
        r"\s*skip(?:\s*(?:to|2))?"
        r"(?:\s*(?:\.mp3|\.wav))?",
        re.I,
    ),
    # "skip to 47" (bare number allowed only with explicit "to")
    re.compile(
        r"(?:^|[\s|,;:*()\/\[\]{}._~〜\-–—☆★♪♫➤►▶])"
        r"skip\s*to\s*\d{1,3}\b"
        r"(?:\s*(?:\.mp3|\.wav))?",
        re.I,
    ),
]

# Splits that look like "Artist - Title" or "Artist – Title".
_DASH_SPLIT = re.compile(r"\s+[-–—]\s+")
_DASH_CHARS = r"\-–—−‒﹘﹣－"


def clean_title(raw: str) -> str:
    """Strip noise markers, collapse whitespace. Doesn't touch the artist."""
    out = raw or ""
    stripped_skip = False
    for pat in _SKIP_TO_PATTERNS:
        new_out = pat.sub(" ", out)
        if new_out != out:
            stripped_skip = True
        out = new_out
    for pat in _NOISE_PATTERNS:
        out = pat.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip(" -–—|·•")
    if out:
        return out
    if stripped_skip:
        return "Untitled"
    return raw


def strip_redundant_artist_prefix(title: str, artist_name: Optional[str]) -> str:
    """Remove leading ``Artist -`` from title when it matches known artist.

    Safety rules:
    - only checks the *start* of the title
    - requires an explicit dash-like separator after the artist
    - never falls back to guessing artist from title
    """
    if not title or not artist_name:
        return title
    artist = (artist_name or "").strip()
    if not artist:
        return title
    # Flexible spaces in artist name, but otherwise exact literal text.
    artist_pat = re.escape(artist).replace(r"\ ", r"\s+")
    pat = re.compile(
        rf"^\s*(?:{artist_pat})\s*[{_DASH_CHARS}]\s*(.+?)\s*$",
        re.I,
    )
    m = pat.match(title)
    if not m:
        return title
    rest = (m.group(1) or "").strip()
    return rest or title


def split_artist_title(raw: str) -> tuple[Optional[str], str]:
    """Try to split 'Artist - Title' patterns. Returns (artist | None, title).
    If no split is found, the input is returned as the title verbatim
    (after cleaning)."""
    cleaned = clean_title(raw)
    parts = _DASH_SPLIT.split(cleaned, maxsplit=1)
    if len(parts) == 2 and 1 <= len(parts[0]) <= 60 and parts[1]:
        return parts[0].strip(), parts[1].strip()
    return None, cleaned
