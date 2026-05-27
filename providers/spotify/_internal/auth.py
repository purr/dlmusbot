"""sp_dc cookie -> Spotify Web Player access token (async).

Flow:
  1. Pull current TOTP secrets dict (disk cache first, remote mirror as
     fallback). Cached for 24h so cold starts don't pay the round-trip.
  2. Pick the highest-versioned secret, deobfuscate it.
  3. Get Spotify's server time.
  4. Generate TOTP code from server time + derived secret.
  5. GET /api/token with sp_dc cookie + totp params -> access token JSON.
"""

import asyncio
import hashlib
import hmac
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp

from core import disk_cache

from .exceptions import AuthError

_log = logging.getLogger(__name__)


# Public mirrors of Spotify's TOTP secret dictionary. Spotify rotates
# the secret every few days; each of these mirrors auto-syncs from
# Spotify's Web Player JS bundle, so picking whichever responds first
# is correct. All three return the identical `{version: [int array]}`
# payload — verified live before commit. Order is "try fastest /
# most reliable first":
#   1. GitHub mirror (high availability, hourly refresh workflow)
#   2. Alternative GitHub mirror (independent uploader, same hourly)
#   3. git.gay forgejo (original — kept as last-resort)
TOTP_SECRETS_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/CycloneAddons/"
    "spotify-token-generator/main/secrets/secretDict.json",
    "https://raw.githubusercontent.com/xyloflake/"
    "spot-secrets-go/main/secrets/secretDict.json",
    "https://git.gay/thereallo/totp-secrets/raw/branch/"
    "main/secrets/secretDict.json",
)
SERVER_TIME_URL = "https://open.spotify.com/api/server-time"
TOKEN_URL = "https://open.spotify.com/api/token"

TOTP_PERIOD = 30
TOTP_DIGITS = 6
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Fast-fail timeout for the secrets fetch specifically. We have a disk
# cache fallback so it's safe to give up quickly and use the cached
# copy when the remote mirror is slow.
SECRETS_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=8)
SECRETS_FETCH_ATTEMPTS = 2

# Cached TOTP secrets are written here after first successful fetch.
# Secrets rotate rarely (weeks to months); 24h fresh-cache window plus
# stale-cache fallback gets the bot warm in ~0ms on every restart
# after the very first one.
# Path is anchored relative to the repo root (this file's grandparent's
# grandparent), NOT cwd — so cron / systemd / pm2 with non-default
# working directory can't silently scatter cache files.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOTP_CACHE_PATH = _REPO_ROOT / "data" / "spotify_totp_secrets.json"
_TOTP_CACHE_TTL_S = 24 * 3600

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US",
    "Origin": "https://open.spotify.com",
    "Referer": "https://open.spotify.com/",
    "Spotify-App-Version": "1.2.87.27.ga2033a72",
    "App-Platform": "WebPlayer",
}


def _derive_secret(ciphertext_ints: list[int]) -> bytes:
    return "".join(
        str(b ^ ((i % 33) + 9)) for i, b in enumerate(ciphertext_ints)
    ).encode("ascii")


def _totp(secret: bytes, server_time_ms: float) -> str:
    counter = int(server_time_ms) // 1000 // TOTP_PERIOD
    digest = hmac.new(secret, counter.to_bytes(8, "big"), hashlib.sha1).digest()
    offset = digest[-1] & 0x0f
    binary = (
        ((digest[offset] & 0x7f) << 24)
        | ((digest[offset + 1] & 0xff) << 16)
        | ((digest[offset + 2] & 0xff) << 8)
        | (digest[offset + 3] & 0xff)
    )
    return str(binary % (10 ** TOTP_DIGITS)).zfill(TOTP_DIGITS)


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    cookies: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
) -> dict:
    try:
        kwargs: dict = {"cookies": cookies, "params": params}
        if timeout is not None:
            kwargs["timeout"] = timeout
        async with session.get(url, **kwargs) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise AuthError(f"GET {url} -> {resp.status}: {body}")
            return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise AuthError(f"network error fetching {url}: {e}") from e


def _is_valid_secrets(payload) -> bool:
    return isinstance(payload, dict) and bool(payload)


def _load_cached_secrets() -> tuple[Optional[dict], bool]:
    """Returns (secrets_or_None, is_fresh) via the shared disk_cache
    helper. `is_fresh` means within TTL; stale secrets are still
    returned so callers can use them as a fallback when the remote
    fetch fails."""
    return disk_cache.load(
        _TOTP_CACHE_PATH,
        value_key="secrets",
        ttl_seconds=_TOTP_CACHE_TTL_S,
        validator=_is_valid_secrets,
    )


def _save_cached_secrets(secrets: dict) -> None:
    disk_cache.save(_TOTP_CACHE_PATH, value_key="secrets", value=secrets)


async def _fetch_secrets_from_mirrors(
    session: aiohttp.ClientSession,
) -> dict:
    """Walk `TOTP_SECRETS_URLS` in order; return the first one that
    answers a valid payload. Each mirror gets `SECRETS_FETCH_ATTEMPTS`
    tries with a tight timeout so a single slow host can't pin the
    whole warmup. Every attempt is logged at DEBUG, every failure at
    WARNING — so when this is genuinely broken the operator sees
    exactly which mirror(s) failed and why."""
    last_err: Optional[Exception] = None
    for mirror_idx, url in enumerate(TOTP_SECRETS_URLS, start=1):
        host = url.split("/", 3)[2] if url.count("/") >= 2 else url
        for attempt in range(1, SECRETS_FETCH_ATTEMPTS + 1):
            _log.debug(
                "TOTP secrets: mirror %d/%d [%s] attempt %d/%d",
                mirror_idx, len(TOTP_SECRETS_URLS), host,
                attempt, SECRETS_FETCH_ATTEMPTS,
            )
            try:
                t0 = time.time()
                payload = await _get_json(
                    session, url, timeout=SECRETS_FETCH_TIMEOUT,
                )
                _log.info(
                    "TOTP secrets fetched from %s in %.2fs (%d versions)",
                    host, time.time() - t0,
                    len(payload) if isinstance(payload, dict) else 0,
                )
                return payload
            except AuthError as e:
                last_err = e
                _log.warning(
                    "TOTP secrets mirror %s attempt %d/%d failed: %s",
                    host, attempt, SECRETS_FETCH_ATTEMPTS, e,
                )
                if attempt < SECRETS_FETCH_ATTEMPTS:
                    await asyncio.sleep(0.5)
    assert last_err is not None
    raise AuthError(
        f"all {len(TOTP_SECRETS_URLS)} TOTP secret mirrors failed; "
        f"last error: {last_err}"
    ) from last_err


async def get_access_token(
    sp_dc: str,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict:
    """Returns: {accessToken, accessTokenExpirationTimestampMs, clientId, isAnonymous}.
    If `session` is provided, it is reused; otherwise a temporary one is created."""
    if not sp_dc:
        raise AuthError("sp_dc cookie is empty")

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT)
    try:
        # Fast path: fresh disk cache — skip remote fetch entirely.
        # Cold path: fetch from remote, persist to disk on success.
        # Last-resort path: remote failed AND cache is stale — use
        # the stale cache anyway. Spotify rotates TOTP secrets so
        # rarely that a several-day-old set usually still works,
        # and shipping an authoritative-but-stale token beats
        # refusing to start.
        cached, cache_fresh = _load_cached_secrets()
        if cached and cache_fresh:
            _log.debug("TOTP secrets: using fresh disk cache")
            secrets = cached
        else:
            if cached:
                _log.debug(
                    "TOTP secrets: disk cache stale, refreshing from mirrors",
                )
            else:
                _log.debug(
                    "TOTP secrets: no disk cache, fetching from mirrors",
                )
            try:
                secrets = await _fetch_secrets_from_mirrors(session)
                _save_cached_secrets(secrets)
            except AuthError as fetch_err:
                if cached is not None:
                    _log.warning(
                        "TOTP secrets fetch failed (%s); using stale cache",
                        fetch_err,
                    )
                    secrets = cached
                else:
                    raise
        if not isinstance(secrets, dict) or not secrets:
            raise AuthError(f"empty TOTP secrets payload: {secrets!r}")
        version = max(secrets.keys(), key=int)
        secret = _derive_secret(secrets[version])
        _log.debug("TOTP secrets: using version %s", version)

        t0 = time.time()
        st = await _get_json(session, SERVER_TIME_URL, cookies={"sp_dc": sp_dc})
        _log.debug(
            "spotify server-time fetched in %.2fs", time.time() - t0,
        )
        try:
            server_time_ms = 1000 * st["serverTime"]
        except (KeyError, TypeError) as e:
            raise AuthError(f"unexpected server-time response: {st!r}") from e

        code = _totp(secret, server_time_ms)
        t0 = time.time()
        token = await _get_json(
            session,
            TOKEN_URL,
            cookies={"sp_dc": sp_dc},
            params={
                "reason": "init",
                "productType": "web-player",
                "totp": code,
                "totpServer": code,
                "totpVer": version,
            },
        )
        _log.debug(
            "spotify access-token minted in %.2fs", time.time() - t0,
        )
        if "accessToken" not in token:
            raise AuthError(f"no accessToken in response: {token!r}")
        return token
    finally:
        if own_session:
            await session.close()
