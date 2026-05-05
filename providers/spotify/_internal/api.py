"""High-level Spotify API: build pydantic models from Mercury/HTTPS responses."""

from __future__ import annotations

from typing import Optional

import aiohttp

from . import protobuf as pb
from .exceptions import StorageResolveError, TrackUnavailableError
from .ids import gid_to_base62
from .librespot import Session
from .models import Album, Artist, AudioFile, Image, Track


# AudioFile.Format enum (from Spotify Metadata.proto).
FORMAT_NAMES = {
    0: "OGG_VORBIS_96",
    1: "OGG_VORBIS_160",
    2: "OGG_VORBIS_320",
    3: "MP3_256",
    4: "MP3_320",
    5: "MP3_160",
    6: "MP3_96",
    7: "MP3_160_ENC",
    8: "AAC_24",
    9: "AAC_48",
    10: "MP4_128",
    11: "MP4_256",
    12: "MP4_128_DUAL",
    13: "MP4_256_DUAL",
    14: "MP4_128_CBCS",
    15: "MP4_256_CBCS",
    16: "FLAC_FLAC",
    17: "MP4_FLAC",
    18: "AAC_24_NORM",
    19: "FLAC_FLAC_24BIT",
}

IMAGE_SIZE_LABELS = {0: "DEFAULT", 1: "SMALL", 2: "LARGE", 3: "XLARGE"}

STORAGE_RESOLVE_URL = (
    "https://gae2-spclient.spotify.com/storage-resolve/files/audio/interactive/"
    "{file_id}?alt=json"
)

TRACK_PLAYBACK_URL = (
    "https://gue1-spclient.spotify.com/track-playback/v1/media/spotify:track:{track_id}"
)


def _gid_to_url(gid_hex: str, kind: str) -> tuple[str, str]:
    """Returns (spotify_id, https URL)."""
    sid = gid_to_base62(bytes.fromhex(gid_hex))
    return sid, f"https://open.spotify.com/{kind}/{sid}"


def _parse_artist(buf: bytes, role: Optional[str] = None) -> Optional[Artist]:
    m = pb.decode(buf)
    gid = pb.get_bytes(m, 1)
    name = pb.get_str(m, 2)
    if not gid or not name:
        return None
    sid, url = _gid_to_url(gid.hex(), "artist")
    return Artist(gid_hex=gid.hex(), spotify_id=sid, name=name, role=role, spotify_url=url)


def _parse_artist_with_role(buf: bytes) -> Optional[Artist]:
    """Track field 31 (artist_with_role): {artist_gid, artist_name, role}."""
    m = pb.decode(buf)
    gid = pb.get_bytes(m, 1)          # artist_gid
    name = pb.get_str(m, 2)           # artist_name
    # role is wire-encoded as sint32 (zigzag) on this endpoint.
    role_raw = pb.get_sint(m, 3)
    if not gid or not name:
        return None
    role_map = {
        1: "MAIN_ARTIST",
        2: "FEATURED_ARTIST",
        3: "REMIXER",
        4: "ACTOR",
        5: "COMPOSER",
        6: "CONDUCTOR",
        7: "ORCHESTRA",
        8: "PRODUCER",
        9: "WRITER",
    }
    role = role_map.get(role_raw or 0)
    sid, url = _gid_to_url(gid.hex(), "artist")
    return Artist(gid_hex=gid.hex(), spotify_id=sid, name=name, role=role, spotify_url=url)


def _parse_image(buf: bytes) -> Optional[Image]:
    m = pb.decode(buf)
    fid = pb.get_bytes(m, 1)
    if not fid:
        return None
    size = IMAGE_SIZE_LABELS.get(pb.get_int(m, 2) or 0, "DEFAULT")
    return Image(
        url=f"https://i.scdn.co/image/{fid.hex()}",
        width=pb.get_int(m, 3),
        height=pb.get_int(m, 4),
        size_label=size,
    )


def _parse_album(buf: bytes) -> Optional[Album]:
    m = pb.decode(buf)
    gid = pb.get_bytes(m, 1)
    name = pb.get_str(m, 2)
    if not gid or not name:
        return None
    artists = [a for a in (_parse_artist(b) for b in pb.get_list(m, 3)) if a]
    label = pb.get_str(m, 5)
    release_date = None
    if 6 in m:
        d = pb.decode(m[6][0])
        # Date fields are sint32 (zigzag) on this endpoint.
        y = pb.get_sint(d, 1)
        mo = pb.get_sint(d, 2)
        da = pb.get_sint(d, 3)
        if y:
            release_date = (
                f"{y:04d}"
                + (f"-{mo:02d}" if mo else "")
                + (f"-{da:02d}" if mo and da else "")
            )
    images: list[Image] = []
    if 17 in m:  # cover_group (ImageGroup with repeated Image at field 1)
        cg = pb.decode(m[17][0])
        for b in pb.get_list(cg, 1):
            img = _parse_image(b)
            if img:
                images.append(img)
    sid, url = _gid_to_url(gid.hex(), "album")
    return Album(
        gid_hex=gid.hex(),
        spotify_id=sid,
        name=name,
        label=label,
        release_date=release_date,
        artists=artists,
        images=images,
        spotify_url=url,
    )


def parse_album_track_gids(buf: bytes) -> tuple[Optional[Album], list[str]]:
    """Decode a Mercury Album protobuf into (Album-metadata, [gid_hex,...]).

    The Mercury album endpoint embeds tracks under `Album.disc[].track[]`,
    but each track entry only carries field 1 (gid) — name / artists /
    duration are *not* there. So we extract just the GIDs in correct
    play-order and let the caller hydrate per-track via Mercury's track
    endpoint (`hm://metadata/4/track/<gid>`).

    Album.disc lives at field 11; Disc.track at its own field 3."""
    album = _parse_album(buf)
    gids: list[str] = []
    m = pb.decode(buf)
    for disc_buf in m.get(11, []):
        if not isinstance(disc_buf, bytes):
            continue
        disc = pb.decode(disc_buf)
        for t_buf in disc.get(3, []):
            if not isinstance(t_buf, bytes):
                continue
            tm = pb.decode(t_buf)
            gid = pb.get_bytes(tm, 1)
            if gid:
                gids.append(gid.hex())
    return album, gids


async def fetch_album_track_gids(
    session: Session, gid_hex: str,
) -> tuple[Optional[Album], list[str]]:
    """Fetch an album over Mercury — returns (Album-metadata, [gid_hex,...])."""
    body = await session.mercury_get(f"hm://metadata/4/album/{gid_hex}")
    return parse_album_track_gids(body)


async def fetch_track_basic(session: Session, gid_hex: str) -> Optional[Track]:
    """Lightweight track fetch — Mercury metadata only, **no** track-playback
    enrichment for FLAC. Used by album/playlist listing where we just need
    name + artists + duration per entry, not the full file table."""
    body = await session.mercury_get(f"hm://metadata/4/track/{gid_hex}")
    return parse_track(body)


def parse_artist_top_tracks(buf: bytes) -> tuple[Optional[str], list[str]]:
    """Decode a Mercury Artist protobuf into (artist_name, [gid_hex,...]).

    Spotify's `hm://metadata/4/artist/<gid>` returns:
        field 2 = artist name
        field 4 = repeated TopTracks { country (1), track (2) repeated }

    Each track entry has its own field 1 = gid. We pull tracks from the
    first TopTracks bucket (typically US — the "global" hits surface
    here). Album/single discographies live at fields 6 / 8 but those
    require a per-album expansion that's too slow for inline."""
    m = pb.decode(buf)
    name = pb.get_str(m, 2)
    gids: list[str] = []
    seen: set[str] = set()
    for tt_buf in m.get(4, []):
        if not isinstance(tt_buf, bytes):
            continue
        tt = pb.decode(tt_buf)
        for ref in tt.get(2, []):
            if not isinstance(ref, bytes):
                continue
            ref_m = pb.decode(ref)
            gid = pb.get_bytes(ref_m, 1)
            if gid:
                gh = gid.hex()
                if gh not in seen:
                    seen.add(gh)
                    gids.append(gh)
        # First populated bucket wins — usually US/global. Stop early
        # so we don't pile up duplicates from per-country lists.
        if gids:
            break
    return name, gids


async def fetch_artist_top_tracks(
    session: Session, gid_hex: str,
) -> tuple[Optional[str], list[str]]:
    """Fetch artist metadata over Mercury → (name, [track_gid_hex,...])."""
    body = await session.mercury_get(f"hm://metadata/4/artist/{gid_hex}")
    return parse_artist_top_tracks(body)


def _collect_audio_files(buf: bytes, out: list[AudioFile], depth: int = 0) -> None:
    """Recursively pull AudioFile messages out of nested Track containers."""
    if depth > 4:
        return
    try:
        m = pb.decode(buf)
    except Exception:
        return
    fid = m.get(1)
    fmt = m.get(2)
    if (
        fid and fmt
        and isinstance(fid[0], bytes) and len(fid[0]) == 20
        and isinstance(fmt[0], int) and fmt[0] in FORMAT_NAMES
        # Image messages also have field 3/4 (width/height); AudioFile doesn't.
        and 3 not in m and 4 not in m
    ):
        out.append(AudioFile(
            file_id_hex=fid[0].hex(),
            format_id=fmt[0],
            format_name=FORMAT_NAMES[fmt[0]],
        ))
        return
    # 12=file, 21=alternative, 24=original_audio, 39=audio_formats
    for k in (12, 21, 24, 39):
        for v in m.get(k, []):
            if isinstance(v, bytes):
                _collect_audio_files(v, out, depth + 1)


def parse_track(track_proto: bytes) -> Track:
    """Parse a Track protobuf into a Track pydantic model."""
    m = pb.decode(track_proto)
    gid = pb.get_bytes(m, 1)
    name = pb.get_str(m, 2) or "<unknown>"
    if not gid:
        raise TrackUnavailableError("track protobuf has no gid")
    sid, url = _gid_to_url(gid.hex(), "track")

    album = _parse_album(m[3][0]) if 3 in m else None

    main_artists = [a for a in (_parse_artist(b, role="MAIN_ARTIST") for b in pb.get_list(m, 4)) if a]

    # artist_with_role (field 32) gives us featured artists / producers / etc.
    featured: list[Artist] = []
    seen_main_gids = {a.gid_hex for a in main_artists}
    for b in pb.get_list(m, 32):
        a = _parse_artist_with_role(b)
        if a and a.gid_hex not in seen_main_gids and a.role != "MAIN_ARTIST":
            featured.append(a)

    # external_id is at field 10 in the current Mercury Track schema.
    isrc = None
    for b in pb.get_list(m, 10):
        em = pb.decode(b)
        t = pb.get_str(em, 1)
        v = pb.get_str(em, 2)
        if t and v and t.lower() == "isrc":
            isrc = v
            break

    # language_of_performance is at field 22 (was 23 in older proto).
    languages = []
    for b in m.get(22, []):
        if isinstance(b, bytes):
            try:
                languages.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                pass

    files: list[AudioFile] = []
    _collect_audio_files(track_proto, files)
    seen_ids: set[str] = set()
    deduped: list[AudioFile] = []
    for f in files:
        if f.file_id_hex in seen_ids:
            continue
        seen_ids.add(f.file_id_hex)
        deduped.append(f)

    return Track(
        gid_hex=gid.hex(),
        spotify_id=sid,
        name=name,
        artists=main_artists,
        featured_artists=featured,
        album=album,
        duration_ms=pb.get_sint(m, 7) or 0,
        track_number=pb.get_sint(m, 5),
        disc_number=pb.get_sint(m, 6),
        isrc=isrc,
        popularity=pb.get_sint(m, 8),
        languages=languages,
        spotify_url=url,
        files=deduped,
    )


async def fetch_playback_files(
    http: aiohttp.ClientSession,
    access_token: str,
    track_id: str,
    *,
    manifest_format: str = "file_ids_flac",
) -> tuple[Optional[str], list[AudioFile]]:
    """Hit track-playback to get file_ids that Mercury doesn't expose.

    The lossless raw-FLAC variant (`file_ids_flac` -> format 16 = FLAC_FLAC)
    is not advertised by Mercury but lives here. MP4 variants are also
    available but they're Widevine-encrypted, so we ignore them.

    Returns (linked_track_id, files). linked_track_id is the URI Spotify
    actually serves; if the input is itself canonical, this equals input."""
    try:
        async with http.get(
            TRACK_PLAYBACK_URL.format(track_id=track_id),
            headers={"Authorization": f"Bearer {access_token}"},
            params={"manifestFileFormat": manifest_format},
        ) as r:
            if r.status != 200:
                return None, []
            data = await r.json(content_type=None)
    except aiohttp.ClientError:
        return None, []

    files: list[AudioFile] = []
    linked_id: Optional[str] = None
    for entry in (data.get("media") or {}).values():
        item = entry.get("item") or {}
        meta = item.get("metadata") or {}
        uri = meta.get("uri")
        if isinstance(uri, str) and uri.startswith("spotify:track:"):
            linked_id = uri.rsplit(":", 1)[-1]
        manifest = item.get("manifest") or {}
        for mlist in manifest.values():
            if not isinstance(mlist, list):
                continue
            for f in mlist:
                fid = f.get("file_id")
                fmt_raw = f.get("format")
                if not fid or fmt_raw is None:
                    continue
                try:
                    fmt = int(fmt_raw)
                except (TypeError, ValueError):
                    continue
                # Only keep formats we can actually decrypt via librespot AES-CTR.
                # MP4* variants (10–13, 17) are Widevine-encrypted = unsupported.
                if fmt not in {0, 1, 2, 8, 16}:
                    continue
                files.append(AudioFile(
                    file_id_hex=fid,
                    format_id=fmt,
                    format_name=FORMAT_NAMES.get(fmt, f"UNKNOWN_{fmt}"),
                ))
    return linked_id, files


async def _resolve_linked_track_id(
    http: aiohttp.ClientSession, access_token: str, track_id: str
) -> Optional[str]:
    """Use the track-playback endpoint to find the actual playable track URI.
    Spotify "links" old/region-restricted track IDs to their currently-playable
    re-releases; the linked URI shows up as `media[…].item.metadata.uri`.
    Returns the new 22-char id, or None if no link is set up."""
    try:
        async with http.get(
            TRACK_PLAYBACK_URL.format(track_id=track_id),
            headers={"Authorization": f"Bearer {access_token}"},
            params={"manifestFileFormat": "file_ids_mp4"},
        ) as r:
            if r.status != 200:
                return None
            data = await r.json(content_type=None)
    except aiohttp.ClientError:
        return None
    media = data.get("media") or {}
    for entry in media.values():
        meta = (entry.get("item") or {}).get("metadata") or {}
        uri = meta.get("uri")
        if isinstance(uri, str) and uri.startswith("spotify:track:"):
            new_id = uri.rsplit(":", 1)[-1]
            if new_id and new_id != track_id:
                return new_id
    return None


def _is_same_recording(a: Track, b: Track) -> bool:
    """Decide whether two parsed tracks represent the *same recording*.

    Spotify's track-playback `linked_from` redirect normally points an
    old/regional/relisted track ID at the canonical playable copy of the
    *same recording*. But it occasionally points at an entirely
    different song (sometimes when a track is removed and Spotify slots
    in a placeholder, sometimes apparent bugs). Auto-following those
    silently lands the user with the wrong audio under the right
    metadata — which is what we just want to avoid.

    Heuristics, in priority order:
      1. ISRC match (case-insensitive) — a globally-unique recording ID;
         strongest signal.
      2. Title equality + at least one shared artist (lowercased).
    Both fail → treat as a different recording, refuse the redirect."""
    if a.isrc and b.isrc and a.isrc.strip().lower() == b.isrc.strip().lower():
        return True
    a_title = (a.name or "").strip().lower()
    b_title = (b.name or "").strip().lower()
    if not a_title or not b_title:
        return False
    if a_title != b_title:
        return False
    a_artists = {(x.name or "").strip().lower() for x in a.artists if x.name}
    b_artists = {(x.name or "").strip().lower() for x in b.artists if x.name}
    return bool(a_artists & b_artists)


async def fetch_track(session: Session, gid_hex: str) -> Track:
    """Fetch + parse a track via Mercury, transparently following `linked_from`
    redirects, and enriching with the FLAC file_id from the track-playback
    endpoint (Mercury never exposes that one for HiFi tracks)."""
    body = await session.mercury_get(f"hm://metadata/4/track/{gid_hex}")
    track = parse_track(body)

    # If the original gid has no files, try to follow Spotify's
    # `linked_from` to a playable variant — but only when it's actually
    # the same recording. Spotify sometimes redirects removed tracks to
    # an unrelated song (e.g. ID `6lixAnaOUA8304wYeoOSnh` "Shake Sum"
    # by DFB Drako redirects to "Pour Two 4's" by Shoreline Mafia in our
    # tests). We refuse those — better to surface "unavailable" than to
    # silently deliver the wrong audio.
    if not track.files and session.http and session.access_token:
        linked_id = await _resolve_linked_track_id(
            session.http, session.access_token, track.spotify_id
        )
        if linked_id:
            from .ids import base62_to_gid
            new_gid_hex = base62_to_gid(linked_id).hex()
            body2 = await session.mercury_get(f"hm://metadata/4/track/{new_gid_hex}")
            track2 = parse_track(body2)
            if track2.files and _is_same_recording(track, track2):
                track = track2
            # else: leave track with files=[] — the trailing
            # TrackUnavailableError raise handles the "no playable
            # version we trust" case.

    # Enrich with formats the playback API knows about (notably raw FLAC)
    # but Mercury doesn't list. Skip silently on any failure. CRITICAL:
    # if the playback API redirects to a different track (`linked_id !=
    # our spotify_id`), refuse the enrichment — the file_ids we'd merge
    # in would unlock audio for the *other* recording, not ours, and
    # Mercury's audio-key endpoint validates the file_id without
    # checking it actually belongs to the requested track gid. That's
    # how a user pasting "Shake Sum" could end up with Pour Two 4's
    # audio under Shake Sum's metadata.
    if session.http and session.access_token:
        try:
            playback_linked, extra = await fetch_playback_files(
                session.http, session.access_token, track.spotify_id,
                manifest_format="file_ids_flac",
            )
        except Exception:
            playback_linked, extra = None, []
        if extra and (playback_linked is None or playback_linked == track.spotify_id):
            seen = {f.file_id_hex for f in track.files}
            merged = list(track.files)
            for f in extra:
                if f.file_id_hex not in seen:
                    merged.append(f)
                    seen.add(f.file_id_hex)
            track = track.model_copy(update={"files": merged})

    if not track.files:
        raise TrackUnavailableError(
            f"track {track.spotify_id} has no playable files (region-locked or removed)"
        )
    return track


async def resolve_cdn_url(
    http: aiohttp.ClientSession, file_id_hex: str, access_token: str
) -> str:
    """Hit storage-resolve and pick a usable CDN URL."""
    url = STORAGE_RESOLVE_URL.format(file_id=file_id_hex)
    try:
        async with http.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        ) as r:
            if r.status != 200:
                body = (await r.text())[:300]
                raise StorageResolveError(f"storage-resolve {r.status}: {body}")
            data = await r.json(content_type=None)
    except aiohttp.ClientError as e:
        raise StorageResolveError(f"storage-resolve network error: {e}") from e

    urls = data.get("cdnurl") or []
    for u in urls:
        if "audio4-gm-fb" in u or "audio-gm-fb" in u:
            continue
        return u
    if urls:
        return urls[0]
    raise StorageResolveError(f"storage-resolve returned no urls: {data}")


# Default priority, highest quality first; (format_name, file_ext).
DEFAULT_FORMAT_PRIORITY: list[tuple[str, str]] = [
    ("FLAC_FLAC_24BIT", "flac"),
    ("FLAC_FLAC", "flac"),
    ("MP4_FLAC", "mp4"),
    ("OGG_VORBIS_320", "ogg"),
    ("OGG_VORBIS_160", "ogg"),
    ("OGG_VORBIS_96", "ogg"),
    ("MP4_256", "mp4"),
    ("MP4_256_DUAL", "mp4"),
    ("MP4_128", "mp4"),
    ("MP4_128_DUAL", "mp4"),
]


def select_best_file(
    track: Track, preferred: Optional[list[str]] = None
) -> tuple[AudioFile, str]:
    """Pick best AudioFile per priority, restricted to formats this library
    can actually decrypt. Returns (file, file_extension).

    FLAC_FLAC / MP4_FLAC / MP4_* exist for HiFi-tier accounts but are
    PlayPlay/Widevine-encrypted and silently dropped from selection. They
    remain visible on the Track for display purposes."""
    by_name = {
        f.format_name: f for f in track.files if f.is_librespot_decryptable
    }
    if not by_name:
        all_fmts = sorted(f.format_name for f in track.files)
        raise TrackUnavailableError(
            f"track has no librespot-decryptable formats; available "
            f"(but DRM-locked): {all_fmts}"
        )
    default_names = [n for n, _ in DEFAULT_FORMAT_PRIORITY]
    order = (
        list(preferred) + [n for n in default_names if n not in preferred]
        if preferred else default_names
    )
    ext_by_name = {n: e for n, e in DEFAULT_FORMAT_PRIORITY}
    for name in order:
        if name in by_name:
            return by_name[name], ext_by_name.get(name, "ogg")
    raise TrackUnavailableError(
        f"none of the desired formats are available; decryptable: {sorted(by_name)}"
    )
