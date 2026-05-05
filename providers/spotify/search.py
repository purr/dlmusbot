"""Spotify search + album/playlist listing.

Two backends sharing the same sp_dc-derived bearer token:

- Free-text track search uses the **internal mobile-client search-view**
  (`spclient.wg.spotify.com/searchview/km/v4/search/<q>`). The same bearer
  token works there. Crucially, the public `/v1/search` endpoint applies a
  near-zero per-app rate budget to TOTP-derived tokens — it returns
  HTTP 429 with a 24-hour Retry-After after only a handful of requests,
  even when the token is otherwise valid. The internal searchview has its
  own (much larger) budget and is what the Spotify mobile app uses, so it
  Just Works under bot-load patterns.

- Album / playlist fetches keep using the public Web API (`/v1/albums`,
  `/v1/playlists`). They're hit at most once per pasted URL, so the
  search-specific 429 problem doesn't bite there.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

import aiohttp

from core.exceptions import ProviderError, TrackNotFoundError
from core.http_retry import get_with_retry
from core.models import Album, ArtistRef, Playlist, Track

from ._internal.auth import DEFAULT_HEADERS

API_BASE = "https://api.spotify.com/v1"

# Internal search-view endpoint used by Spotify's own mobile clients. Any of
# the regional hosts (gae2-, gew4-, spclient.wg) accepts the same request;
# spclient.wg.spotify.com is the canonical alias.
SEARCHVIEW_BASE = "https://spclient.wg.spotify.com/searchview/km/v4/search"

# Internal playlist endpoint. Returns revision + length + items[].uri but no
# per-track metadata — those have to be hydrated via Mercury. Same host /
# bearer token as searchview, no separate rate budget issues observed.
PLAYLIST_V2_BASE = "https://spclient.wg.spotify.com/playlist/v2/playlist"

# Shared header bundle used by every spclient call: pose as the iOS Music
# app so the endpoint hands us the higher mobile-client rate budget. Album
# / playlist / search calls all reuse this.
SPCLIENT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en",
    "App-Platform": "iOS",
    "User-Agent": "Spotify/8.9.10 iOS/17.4 (iPhone15,2)",
}

# Backwards-compat alias — older imports referenced this private name.
_SEARCHVIEW_HEADERS = SPCLIENT_HEADERS


def _artist_refs(items: list[dict]) -> list[ArtistRef]:
    out: list[ArtistRef] = []
    for a in items or []:
        if not a:
            continue
        name = a.get("name")
        if not name:
            continue
        aid = a.get("id")
        out.append(ArtistRef(
            name=name,
            artist_id=aid,
            url=f"https://open.spotify.com/artist/{aid}" if aid else None,
        ))
    return out


def _track_from_api(item: dict) -> Optional[Track]:
    """Build a core.Track from an /v1/tracks-shaped payload."""
    if not item or not isinstance(item, dict):
        return None
    tid = item.get("id")
    if not tid:
        return None
    artists = _artist_refs(item.get("artists") or [])
    album = item.get("album") or {}
    images = album.get("images") or []
    artwork = images[0].get("url") if images else None
    return Track(
        provider="spotify",
        track_id=tid,
        title=item.get("name") or "<unknown>",
        artists=artists,
        album=album.get("name"),
        duration_seconds=int((item.get("duration_ms") or 0) // 1000),
        artwork_url=artwork,
        url=f"https://open.spotify.com/track/{tid}",
        isrc=(item.get("external_ids") or {}).get("isrc"),
        extra={"album_id": album.get("id")} if album.get("id") else {},
    )


async def _get(
    http: aiohttp.ClientSession, url: str, token: str,
    *, max_attempts: int = 4, **params,
) -> dict:
    headers = {"Authorization": f"Bearer {token}", **DEFAULT_HEADERS}
    try:
        r = await get_with_retry(
            http, url, headers=headers, params=params,
            max_attempts=max_attempts,
        )
    except aiohttp.ClientError as e:
        raise ProviderError(f"spotify network error: {e}") from e
    async with r:
        if r.status == 404:
            raise TrackNotFoundError(f"spotify {url} not found")
        if r.status != 200:
            body = (await r.text())[:200]
            raise ProviderError(f"spotify {url} -> {r.status}: {body}")
        return await r.json(content_type=None)


def _id_from_uri(uri: Optional[str]) -> Optional[str]:
    """`spotify:track:abc...` -> `abc...` (also handles artist/album URIs)."""
    if not uri or ":" not in uri:
        return None
    return uri.rsplit(":", 1)[-1] or None


def _track_from_searchview(hit: dict) -> Optional[Track]:
    """Parse one entry from the internal /searchview/km/v4 response into our
    core.Track. Schema:
        {name, uri, image, duration (ms), popularity,
         artists:[{name, uri}], album:{name, uri}}"""
    if not isinstance(hit, dict):
        return None
    tid = _id_from_uri(hit.get("uri"))
    if not tid:
        return None
    artists: list[ArtistRef] = []
    for a in hit.get("artists") or []:
        name = (a or {}).get("name")
        if not name:
            continue
        aid = _id_from_uri((a or {}).get("uri"))
        artists.append(ArtistRef(
            name=name,
            artist_id=aid,
            url=f"https://open.spotify.com/artist/{aid}" if aid else None,
        ))
    album = hit.get("album") or {}
    album_name = album.get("name") or None
    album_id = _id_from_uri(album.get("uri"))
    return Track(
        provider="spotify",
        track_id=tid,
        title=hit.get("name") or "<unknown>",
        artists=artists,
        album=album_name,
        duration_seconds=int((hit.get("duration") or 0) // 1000),
        artwork_url=hit.get("image") or None,
        url=f"https://open.spotify.com/track/{tid}",
        extra={"album_id": album_id} if album_id else {},
    )


async def search_tracks(
    http: aiohttp.ClientSession, token: str, query: str, limit: int = 25
) -> list[Track]:
    """Free-text track search via the internal mobile search-view endpoint.

    Why not /v1/search? See the module docstring — it's near-immediately
    rate-limited for sp_dc-derived tokens. The internal searchview takes
    the same bearer and returns the same catalogue."""
    q = query.strip()
    if not q:
        return []
    url = f"{SEARCHVIEW_BASE}/{quote(q, safe='')}"
    headers = {
        "Authorization": f"Bearer {token}",
        **_SEARCHVIEW_HEADERS,
    }
    params = {
        "entityVersion": "2",
        "catalogue": "premium",
        "country": "US",
        "locale": "en",
        "platform": "iPhone-iPhone10_5",
        "limit": str(min(max(int(limit), 1), 50)),
        "imageSize": "large",
    }
    try:
        async with http.get(url, headers=headers, params=params) as r:
            if r.status == 401:
                # Token went stale mid-flight; let the caller refresh.
                raise ProviderError("spotify searchview 401 (token expired)")
            if r.status != 200:
                body = (await r.text())[:200]
                raise ProviderError(f"spotify searchview {r.status}: {body}")
            data = await r.json(content_type=None)
    except aiohttp.ClientError as e:
        raise ProviderError(f"spotify searchview network error: {e}") from e

    hits = (((data.get("results") or {}).get("tracks") or {}).get("hits") or [])
    return [t for t in (_track_from_searchview(h) for h in hits) if t]


async def fetch_album(
    http: aiohttp.ClientSession, token: str, album_id: str
) -> Album:
    data = await _get(http, f"{API_BASE}/albums/{album_id}", token, market="from_token")
    aid = data.get("id") or album_id
    artists = _artist_refs(data.get("artists") or [])
    images = data.get("images") or []
    artwork = images[0].get("url") if images else None
    items = (data.get("tracks") or {}).get("items") or []
    tracks: list[Track] = []
    for it in items:
        tid = it.get("id")
        if not tid:
            continue
        t_artists = _artist_refs(it.get("artists") or [])
        tracks.append(Track(
            provider="spotify",
            track_id=tid,
            title=it.get("name") or "<unknown>",
            artists=t_artists,
            album=data.get("name"),
            duration_seconds=int((it.get("duration_ms") or 0) // 1000),
            artwork_url=artwork,
            url=f"https://open.spotify.com/track/{tid}",
            extra={"album_id": aid},
        ))
    return Album(
        provider="spotify",
        album_id=aid,
        title=data.get("name") or "<unknown album>",
        artists=artists,
        artwork_url=artwork,
        url=f"https://open.spotify.com/album/{aid}",
        tracks=tracks,
    )


async def fetch_playlist(
    http: aiohttp.ClientSession, token: str, playlist_id: str
) -> Playlist:
    data = await _get(
        http, f"{API_BASE}/playlists/{playlist_id}", token, market="from_token",
    )
    pid = data.get("id") or playlist_id
    images = data.get("images") or []
    artwork = images[0].get("url") if images else None
    owner = (data.get("owner") or {}).get("display_name")
    items = (data.get("tracks") or {}).get("items") or []

    # Walk paginated `next` until exhausted (capped to keep latency bounded).
    all_items = list(items)
    next_url = (data.get("tracks") or {}).get("next")
    safety = 0
    while next_url and safety < 10 and len(all_items) < 200:
        page = await _get(http, next_url, token)
        all_items.extend(page.get("items") or [])
        next_url = page.get("next")
        safety += 1

    tracks: list[Track] = []
    for entry in all_items:
        it = (entry or {}).get("track") or {}
        t = _track_from_api(it)
        if t:
            tracks.append(t)
    return Playlist(
        provider="spotify",
        playlist_id=pid,
        title=data.get("name") or "<unknown playlist>",
        owner=owner,
        artwork_url=artwork,
        url=f"https://open.spotify.com/playlist/{pid}",
        tracks=tracks,
    )
