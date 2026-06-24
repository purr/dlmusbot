"""Spotify search + album/playlist listing.

Two backends sharing the same sp_dc-derived bearer token:

- Free-text track search uses the Web Player's **pathfinder GraphQL API**
  (`api-partner.spotify.com/pathfinder/v1/query`, `searchDesktop`
  operation). Spotify's Feb-2026 API change retired the old internal
  mobile search-view route (`spclient.../searchview/km/v4`), which now
  answers a bare HTTP 400, so search moved to the same endpoint the
  desktop Web Player itself uses. Pathfinder takes the bearer token PLUS a
  separately-minted `client-token` header (see provider._client_token).
  The public `/v1/search` endpoint stays unusable: it applies a near-zero
  per-app rate budget to TOTP-derived tokens, returning HTTP 429 with a
  24-hour Retry-After after only a handful of requests.

- Album / playlist fetches keep using the public Web API (`/v1/albums`,
  `/v1/playlists`). They're hit at most once per pasted URL, so the
  search-specific 429 problem doesn't bite there.
"""

from __future__ import annotations

import json
from typing import Optional

import aiohttp

from core.exceptions import ProviderError, TrackNotFoundError
from core.http_retry import get_with_retry
from core.models import Album, ArtistRef, Playlist, Track

from ._internal.auth import DEFAULT_HEADERS
from ._internal.exceptions import TokenExpiredError

API_BASE = "https://api.spotify.com/v1"

# Web Player pathfinder GraphQL endpoint — the search backend the desktop
# client uses, and the only free-text search that still works for sp_dc
# tokens after the Feb-2026 API change. Requires both the bearer token and
# a `client-token` header.
PATHFINDER_BASE = "https://api-partner.spotify.com/pathfinder/v1/query"

# Persisted-query hash for the `searchDesktop` operation. Spotify rotates
# these whenever it ships a new Web Player JS bundle; if search starts
# failing with `PersistedQueryNotFound` this is the value to refresh — grab
# it from a logged-in open.spotify.com search request in a browser network
# trace. The `variables` set in `search_tracks` is paired to THIS hash;
# change them together.
SEARCH_OPERATION = "searchDesktop"
SEARCH_QUERY_HASH = "21969b655b795601fb2d2204a4243188e75fdc6d3520e7b9cd3f4db2aff9591e"

# Internal playlist endpoint. Returns revision + length + items[].uri but no
# per-track metadata — those have to be hydrated via Mercury.
PLAYLIST_V2_BASE = "https://spclient.wg.spotify.com/playlist/v2/playlist"

# Shared header bundle for spclient calls (the playlist-v2 fetch): pose as
# the iOS Music app so the endpoint hands us the higher mobile-client rate
# budget.
SPCLIENT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en",
    "App-Platform": "iOS",
    "User-Agent": "Spotify/8.9.10 iOS/17.4 (iPhone15,2)",
}


def _artist_refs(items: list[dict]) -> list[ArtistRef]:
    out: list[ArtistRef] = []
    for a in items or []:
        if not a:
            continue
        name = a.get("name")
        if not name:
            continue
        aid = a.get("id")
        out.append(
            ArtistRef(
                name=name,
                artist_id=aid,
                url=f"https://open.spotify.com/artist/{aid}" if aid else None,
            )
        )
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
    http: aiohttp.ClientSession,
    url: str,
    token: str,
    *,
    max_attempts: int = 4,
    **params,
) -> dict:
    headers = {"Authorization": f"Bearer {token}", **DEFAULT_HEADERS}
    try:
        r = await get_with_retry(
            http,
            url,
            headers=headers,
            params=params,
            max_attempts=max_attempts,
        )
    except aiohttp.ClientError as e:
        raise ProviderError(f"spotify network error: {e}") from e
    async with r:
        if r.status == 401:
            raise TokenExpiredError(f"spotify {url} 401 (token expired)")
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


def _largest_cover(cover_art: dict) -> Optional[str]:
    """Pick the widest image URL from a pathfinder `coverArt.sources` list."""
    best: Optional[str] = None
    best_w = -1
    for s in (cover_art or {}).get("sources") or []:
        if not isinstance(s, dict) or not s.get("url"):
            continue
        w = s.get("width") or 0
        if w >= best_w:
            best_w = w
            best = s["url"]
    return best


def _track_from_pathfinder(node: dict) -> Optional[Track]:
    """Parse one `searchV2.tracksV2.items[]` entry from a pathfinder
    `searchDesktop` response into our core.Track. Each entry wraps the track
    as `{item: {data: {...}}}`; the inner `data` looks like:
        {uri, name, duration:{totalMilliseconds},
         artists:{items:[{uri, profile:{name}}]},
         albumOfTrack:{uri, name, coverArt:{sources:[{url, width}]}}}"""
    if not isinstance(node, dict):
        return None
    data = (node.get("item") or {}).get("data") or node.get("data") or node
    if not isinstance(data, dict):
        return None
    tid = _id_from_uri(data.get("uri"))
    if not tid:
        return None
    artists: list[ArtistRef] = []
    for a in (data.get("artists") or {}).get("items") or []:
        name = ((a or {}).get("profile") or {}).get("name")
        if not name:
            continue
        aid = _id_from_uri((a or {}).get("uri"))
        artists.append(
            ArtistRef(
                name=name,
                artist_id=aid,
                url=f"https://open.spotify.com/artist/{aid}" if aid else None,
            )
        )
    album = data.get("albumOfTrack") or {}
    album_name = album.get("name") or None
    album_id = _id_from_uri(album.get("uri"))
    duration_ms = (data.get("duration") or {}).get("totalMilliseconds") or 0
    return Track(
        provider="spotify",
        track_id=tid,
        title=data.get("name") or "<unknown>",
        artists=artists,
        album=album_name,
        duration_seconds=int(duration_ms // 1000),
        artwork_url=_largest_cover(album.get("coverArt") or {}),
        url=f"https://open.spotify.com/track/{tid}",
        extra={"album_id": album_id} if album_id else {},
    )


async def search_tracks(
    http: aiohttp.ClientSession,
    token: str,
    client_token: str,
    query: str,
    limit: int = 25,
) -> list[Track]:
    """Free-text track search via the Web Player pathfinder GraphQL API.

    Needs the sp_dc-derived bearer token AND a `client-token` (minted
    separately by the provider). See the module docstring for why neither
    the old searchview route nor /v1/search is usable."""
    q = query.strip()
    if not q:
        return []
    variables = {
        "searchTerm": q,
        "offset": 0,
        "limit": min(max(int(limit), 1), 50),
        "numberOfTopResults": 5,
        "includeAudiobooks": False,
    }
    extensions = {"persistedQuery": {"version": 1, "sha256Hash": SEARCH_QUERY_HASH}}
    params = {
        "operationName": SEARCH_OPERATION,
        "variables": json.dumps(variables, separators=(",", ":")),
        "extensions": json.dumps(extensions, separators=(",", ":")),
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "client-token": client_token,
        "Accept": "application/json",
        "App-Platform": "WebPlayer",
    }
    try:
        async with http.get(PATHFINDER_BASE, headers=headers, params=params) as r:
            if r.status == 401:
                # Token went stale mid-flight; let the caller refresh.
                raise TokenExpiredError("spotify pathfinder 401 (token expired)")
            if r.status != 200:
                body = (await r.text())[:200]
                raise ProviderError(f"spotify pathfinder {r.status}: {body}")
            data = await r.json(content_type=None)
    except aiohttp.ClientError as e:
        raise ProviderError(f"spotify pathfinder network error: {e}") from e
    except ValueError as e:
        # Malformed / non-JSON 200 body (CDN or WAF interstitial, maintenance
        # page). json() raises a ValueError subclass — wrap it like the rest.
        raise ProviderError(f"spotify pathfinder bad json: {e}") from e

    if not isinstance(data, dict):
        return []
    errors = data.get("errors")
    if errors:
        msg = "; ".join(str((e or {}).get("message") or e) for e in errors)[:200]
        raise ProviderError(f"spotify pathfinder error: {msg}")

    items = (
        ((data.get("data") or {}).get("searchV2") or {}).get("tracksV2") or {}
    ).get("items") or []
    return [t for t in (_track_from_pathfinder(it) for it in items) if t]


async def fetch_album(http: aiohttp.ClientSession, token: str, album_id: str) -> Album:
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
        tracks.append(
            Track(
                provider="spotify",
                track_id=tid,
                title=it.get("name") or "<unknown>",
                artists=t_artists,
                album=data.get("name"),
                duration_seconds=int((it.get("duration_ms") or 0) // 1000),
                artwork_url=artwork,
                url=f"https://open.spotify.com/track/{tid}",
                extra={"album_id": aid},
            )
        )
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
        http,
        f"{API_BASE}/playlists/{playlist_id}",
        token,
        market="from_token",
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
