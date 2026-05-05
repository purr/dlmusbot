"""Spotify ID conversions: base62 (22-char) <-> 16-byte gid."""

import re

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOOKUP = {c: i for i, c in enumerate(_ALPHABET)}

_TRACK_URL_RE = re.compile(r"track[/:]([A-Za-z0-9]{22})")


def parse_track_id(s: str) -> str:
    """Accept either a Spotify track URL/URI or a raw 22-char id."""
    s = s.strip()
    m = _TRACK_URL_RE.search(s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{22}", s):
        return s
    raise ValueError(f"can't parse Spotify track id from: {s!r}")


def base62_to_gid(track_id: str) -> bytes:
    val = 0
    for ch in track_id:
        val = val * 62 + _LOOKUP[ch]
    return val.to_bytes(16, "big")


def gid_to_base62(gid: bytes) -> str:
    val = int.from_bytes(gid, "big")
    chars: list[str] = []
    while val > 0:
        val, r = divmod(val, 62)
        chars.append(_ALPHABET[r])
    s = "".join(reversed(chars))
    return s.rjust(22, "0")
