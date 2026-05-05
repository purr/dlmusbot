"""Async minimal librespot client.

Implements the parts of Spotify's binary protocol we need:
  - apresolve.spotify.com discovery
  - TCP connection to AP server
  - Diffie-Hellman handshake -> Shannon cipher pair
  - Login with AUTHENTICATION_SPOTIFY_TOKEN
  - Mercury GET (track metadata, etc.)
  - request_key / aes_key channel (audio decryption keys)

Async via asyncio.open_connection. Pure stdlib + aiohttp + our local
aes / shannon / protobuf modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import struct
from typing import Optional

import aiohttp

from . import protobuf as pb
from .exceptions import HandshakeError, LoginError, MercuryError
from .shannon import Shannon


DH_PRIME = int.from_bytes(
    bytes.fromhex(
        "ffffffffffffffffc90fdaa22168c234c4c6628b80dc1cd1"
        "29024e088a67cc74020bbea63b139b22514a08798e3404dd"
        "ef9519b3cd3a431b302b0a6df25f14374fe1356d6d51c245"
        "e485b576625e7ec6f44c42e9a63a3620ffffffffffffffff"
    ),
    "big",
)
DH_GENERATOR = 2

PACKET_LOGIN = 0xab
PACKET_AP_WELCOME = 0xac
PACKET_AUTH_FAILURE = 0xad
PACKET_REQUEST_KEY = 0x0c
PACKET_AES_KEY = 0x0d
PACKET_AES_KEY_ERROR = 0x0e
PACKET_PING = 0x04
PACKET_PONG = 0x49
PACKET_MERCURY_REQ = 0xb2
PACKET_MERCURY_SUB = 0xb3

# Spotify APLoginFailed error codes (from keyexchange.proto).
AP_ERROR_NAMES = {
    0: "ProtocolError", 2: "TryAnotherAP", 5: "BadConnectionId",
    9: "TravelRestriction", 11: "PremiumAccountRequired",
    12: "BadCredentials", 13: "CouldNotValidateCredentials",
    14: "AccountExists", 15: "ExtraVerificationRequired",
    16: "InvalidAppKey", 17: "ApplicationBanned",
}


async def _resolve_ap(http: aiohttp.ClientSession) -> str:
    try:
        async with http.get(
            "https://apresolve.spotify.com/?type=accesspoint",
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            data = await r.json(content_type=None)
    except aiohttp.ClientError as e:
        raise HandshakeError(f"apresolve network failure: {e}") from e
    aps = data.get("accesspoint") or []
    if not aps:
        raise HandshakeError(f"apresolve returned no accesspoints: {data}")
    return aps[0]


def _build_client_hello(dh_public: int, nonce: bytes) -> bytes:
    build_info = pb.encode([
        (10, pb.VARINT, 1),       # product = LIBSPOTIFY
        (30, pb.VARINT, 0x27),    # platform = WIN32_X86_64
        (40, pb.VARINT, 117800400),
    ])
    gc = dh_public.to_bytes((dh_public.bit_length() + 7) // 8, "big")
    dh_hello = pb.encode([(10, pb.LEN, gc), (20, pb.VARINT, 1)])
    login_crypto_hello = pb.encode([(10, pb.LEN, dh_hello)])
    return pb.encode([
        (10, pb.LEN, build_info),
        (30, pb.VARINT, 0),        # SHANNON
        (50, pb.LEN, login_crypto_hello),
        (60, pb.LEN, nonce),
        (70, pb.LEN, b"\x1e"),
    ])


def _build_client_response_plaintext(challenge_hmac: bytes) -> bytes:
    dh_resp = pb.encode([(10, pb.LEN, challenge_hmac)])
    login_crypto_response = pb.encode([(10, pb.LEN, dh_resp)])
    return pb.encode([
        (10, pb.LEN, login_crypto_response),
        (20, pb.LEN, b""),
        (30, pb.LEN, b""),
    ])


def _build_client_response_encrypted(access_token: str, device_id: str) -> bytes:
    login_credentials = pb.encode([
        (20, pb.VARINT, 3),       # AUTHENTICATION_SPOTIFY_TOKEN
        (30, pb.LEN, access_token.encode("utf-8")),
    ])
    system_info = pb.encode([
        (10, pb.VARINT, 0),       # CPU_UNKNOWN
        (60, pb.VARINT, 0),       # OS_UNKNOWN
        (90, pb.LEN, b"librespot-python-stdlib"),
        (100, pb.LEN, device_id.encode("ascii")),
    ])
    return pb.encode([
        (10, pb.LEN, login_credentials),
        (50, pb.LEN, system_info),
        (70, pb.LEN, b"librespot-stdlib-1.0"),
    ])


class Session:
    """Async Spotify librespot session.

    Use as an async context manager:
        async with Session(access_token) as sess:
            body = await sess.mercury_get(...)
            key  = await sess.request_audio_key(...)
    """

    def __init__(
        self,
        access_token: str,
        *,
        http: Optional[aiohttp.ClientSession] = None,
    ):
        self._access_token = access_token
        self._http = http
        self._owns_http = http is None
        self._device_id = os.urandom(20).hex()

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._send_cipher: Optional[Shannon] = None
        self._recv_cipher: Optional[Shannon] = None
        self._send_nonce = 0
        self._recv_nonce = 0
        self._mercury_seq = 0
        self._lock = asyncio.Lock()

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def access_token(self) -> str:
        return self._access_token

    @access_token.setter
    def access_token(self, value: str) -> None:
        # Update token used by HTTP-side calls (fetch_track playback API,
        # storage-resolve). The AP socket handshake-bound auth is unaffected.
        self._access_token = value

    @property
    def is_connected(self) -> bool:
        return self._writer is not None

    @property
    def http(self) -> Optional[aiohttp.ClientSession]:
        return self._http

    async def __aenter__(self) -> "Session":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession()
        ap = await _resolve_ap(self._http)
        host, _, port_s = ap.partition(":")
        port = int(port_s) if port_s else 4070
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=30,
            )
        except (OSError, asyncio.TimeoutError) as e:
            raise HandshakeError(f"failed connecting to AP {host}:{port}: {e}") from e
        try:
            await self._handshake()
            await self._authenticate()
        except (HandshakeError, LoginError):
            await self.close()
            raise

    async def _read_exact(self, n: int) -> bytes:
        try:
            return await self._reader.readexactly(n)
        except asyncio.IncompleteReadError as e:
            raise HandshakeError(f"connection closed mid-read ({len(e.partial)}/{n})") from e

    async def _handshake(self) -> None:
        priv = int.from_bytes(os.urandom(0x5f), "big")
        pub = pow(DH_GENERATOR, priv, DH_PRIME)
        nonce = os.urandom(16)

        hello = _build_client_hello(pub, nonce)
        sent = b"\x00\x04" + struct.pack(">I", 2 + 4 + len(hello)) + hello
        self._writer.write(sent)
        await self._writer.drain()

        ap_total_len = struct.unpack(">I", await self._read_exact(4))[0]
        ap_body = await self._read_exact(ap_total_len - 4)
        full_recv = struct.pack(">I", ap_total_len) + ap_body

        try:
            ap_msg = pb.decode(ap_body)
            challenge_msg = pb.decode(pb.get_bytes(ap_msg, 10))
            lcu = pb.decode(pb.get_bytes(challenge_msg, 10))
            dh_chall = pb.decode(pb.get_bytes(lcu, 10))
            gs = pb.get_bytes(dh_chall, 10)
        except Exception as e:
            raise HandshakeError(f"failed parsing APResponseMessage: {e}") from e

        shared_key_int = pow(int.from_bytes(gs, "big"), priv, DH_PRIME)
        shared_key = shared_key_int.to_bytes(
            (shared_key_int.bit_length() + 7) // 8, "big"
        )

        acc = sent + full_recv
        keys_buf = b""
        for i in range(1, 6):
            keys_buf += hmac.new(shared_key, acc + bytes([i]), hashlib.sha1).digest()
        challenge_hmac = hmac.new(keys_buf[:20], acc, hashlib.sha1).digest()

        crp = _build_client_response_plaintext(challenge_hmac)
        self._writer.write(struct.pack(">I", 4 + len(crp)) + crp)
        await self._writer.drain()

        self._send_cipher = Shannon()
        self._send_cipher.key(keys_buf[20:52])
        self._recv_cipher = Shannon()
        self._recv_cipher.key(keys_buf[52:84])
        self._send_nonce = 0
        self._recv_nonce = 0

    async def _send_encoded(self, cmd: int, payload: bytes) -> None:
        c = self._send_cipher
        c.nonce(self._send_nonce)
        self._send_nonce += 1
        body = bytes([cmd]) + struct.pack(">H", len(payload)) + payload
        ct = c.encrypt(body)
        mac = c.finish(4)
        self._writer.write(ct + mac)
        await self._writer.drain()

    async def _recv_encoded(self) -> tuple[int, bytes]:
        c = self._recv_cipher
        c.nonce(self._recv_nonce)
        self._recv_nonce += 1
        header = c.decrypt(await self._read_exact(3))
        cmd = header[0]
        payload_len = (header[1] << 8) | header[2]
        payload = c.decrypt(await self._read_exact(payload_len))
        mac = await self._read_exact(4)
        if mac != c.finish(4):
            raise HandshakeError("bad packet MAC (cipher state desynced)")
        return cmd, payload

    async def _authenticate(self) -> None:
        cre = _build_client_response_encrypted(self._access_token, self._device_id)
        await self._send_encoded(PACKET_LOGIN, cre)
        # Drain non-fatal early packets until APWelcome / AuthFailure.
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            if asyncio.get_running_loop().time() > deadline:
                raise LoginError("auth timed out waiting for APWelcome")
            cmd, payload = await self._recv_encoded()
            if cmd == PACKET_AP_WELCOME:
                return
            if cmd == PACKET_AUTH_FAILURE:
                m = pb.decode(payload)
                code = pb.get_int(m, 10)
                desc = pb.get_str(m, 40)
                name = AP_ERROR_NAMES.get(code, str(code))
                raise LoginError(f"login failed: {name}{(' / ' + desc) if desc else ''}")
            # else: server info (country_code, product_info, license_version, etc.) — ignore.

    def _next_mercury_seq(self) -> int:
        s = self._mercury_seq
        self._mercury_seq += 1
        return s

    async def mercury_get(self, uri: str) -> bytes:
        """GET via Mercury. Returns concatenated body bytes (post-Header parts)."""
        async with self._lock:
            seq = self._next_mercury_seq()
            header = pb.encode([
                (1, pb.LEN, uri.encode("utf-8")),
                (3, pb.LEN, b"GET"),
            ])
            body = (
                struct.pack(">H", 4)
                + struct.pack(">I", seq)
                + b"\x01"
                + struct.pack(">H", 1)
                + struct.pack(">H", len(header))
                + header
            )
            await self._send_encoded(PACKET_MERCURY_REQ, body)
            return await self._wait_mercury(seq)

    async def _wait_mercury(self, want_seq: int) -> bytes:
        partials: dict[int, list[bytes]] = {}
        while True:
            cmd, payload = await self._recv_encoded()
            if cmd in (PACKET_MERCURY_REQ, PACKET_MERCURY_SUB):
                pos = 0
                seq_len = struct.unpack(">H", payload[pos:pos + 2])[0]
                pos += 2
                if seq_len == 2:
                    seq = struct.unpack(">H", payload[pos:pos + 2])[0]
                    pos += 2
                elif seq_len == 4:
                    seq = struct.unpack(">I", payload[pos:pos + 4])[0]
                    pos += 4
                elif seq_len == 8:
                    seq = struct.unpack(">Q", payload[pos:pos + 8])[0]
                    pos += 8
                else:
                    raise MercuryError(f"unknown mercury seq length {seq_len}")
                flags = payload[pos]
                pos += 1
                parts = struct.unpack(">H", payload[pos:pos + 2])[0]
                pos += 2
                buf = partials.setdefault(seq, [])
                for _ in range(parts):
                    size = struct.unpack(">H", payload[pos:pos + 2])[0]
                    pos += 2
                    buf.append(payload[pos:pos + size])
                    pos += size
                if flags == 0x01 and seq == want_seq:
                    header = pb.decode(buf[0])
                    raw = pb.get_int(header, 4)
                    # status_code is sint32 (zigzag-encoded).
                    status = 200 if raw is None else (raw >> 1) ^ -(raw & 1)
                    if not (200 <= status < 300):
                        body_preview = b"".join(buf[1:])[:200]
                        raise MercuryError(
                            f"mercury GET status {status}, body={body_preview!r}"
                        )
                    return b"".join(buf[1:])
            elif cmd == PACKET_PING:
                await self._send_encoded(PACKET_PONG, payload)
            # else: ignore other server packets.

    async def request_audio_key(
        self, file_id: bytes, track_gid: bytes
    ) -> bytes:
        """Request 16-byte AES key for (file_id, track_gid)."""
        async with self._lock:
            seq = self._mercury_seq
            self._mercury_seq += 1
            payload = file_id + track_gid + struct.pack(">I", seq) + b"\x00\x00"
            await self._send_encoded(PACKET_REQUEST_KEY, payload)
            while True:
                cmd, body = await self._recv_encoded()
                if cmd == PACKET_AES_KEY:
                    got = struct.unpack(">I", body[:4])[0]
                    if got == seq:
                        return body[4:20]
                elif cmd == PACKET_AES_KEY_ERROR:
                    got = struct.unpack(">I", body[:4])[0]
                    if got == seq:
                        code = struct.unpack(">H", body[4:6])[0]
                        raise MercuryError(f"audio_key error code {code}")
                elif cmd == PACKET_PING:
                    await self._send_encoded(PACKET_PONG, body)

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass
            self._writer = None
            self._reader = None
        if self._owns_http and self._http is not None:
            await self._http.close()
            self._http = None
