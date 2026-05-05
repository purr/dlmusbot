"""sp_dc cookie -> Spotify Web Player access token (async).

Flow:
  1. Pull current TOTP secrets dict from a public mirror.
  2. Pick the highest-versioned secret, deobfuscate it.
  3. Get Spotify's server time.
  4. Generate TOTP code from server time + derived secret.
  5. GET /api/token with sp_dc cookie + totp params -> access token JSON.
"""

import hashlib
import hmac
from typing import Optional

import aiohttp

from .exceptions import AuthError


TOTP_SECRETS_URL = (
    "https://git.gay/thereallo/totp-secrets/raw/branch/main/secrets/secretDict.json"
)
SERVER_TIME_URL = "https://open.spotify.com/api/server-time"
TOKEN_URL = "https://open.spotify.com/api/token"

TOTP_PERIOD = 30
TOTP_DIGITS = 6
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

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
) -> dict:
    try:
        async with session.get(url, cookies=cookies, params=params) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise AuthError(f"GET {url} -> {resp.status}: {body}")
            return await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        raise AuthError(f"network error fetching {url}: {e}") from e


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
        secrets = await _get_json(session, TOTP_SECRETS_URL)
        if not isinstance(secrets, dict) or not secrets:
            raise AuthError(f"empty TOTP secrets payload: {secrets!r}")
        version = max(secrets.keys(), key=int)
        secret = _derive_secret(secrets[version])

        st = await _get_json(session, SERVER_TIME_URL, cookies={"sp_dc": sp_dc})
        try:
            server_time_ms = 1000 * st["serverTime"]
        except (KeyError, TypeError) as e:
            raise AuthError(f"unexpected server-time response: {st!r}") from e

        code = _totp(secret, server_time_ms)
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
        if "accessToken" not in token:
            raise AuthError(f"no accessToken in response: {token!r}")
        return token
    finally:
        if own_session:
            await session.close()
