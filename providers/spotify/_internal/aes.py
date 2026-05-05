"""Pure Python AES-128 ECB. Used to derive CTR keystream for Spotify audio.

Stdlib has no symmetric ciphers, so AES is implemented here from scratch.
Only encrypt (single block, ECB) is needed: AES-CTR keystream = AES_ECB(key, counter).
"""

# AES S-box (forward only; we never decrypt directly).
_SBOX = bytes((
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
))

_RCON = (0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36)


def _xtime(b: int) -> int:
    return ((b << 1) ^ 0x1b) & 0xff if b & 0x80 else (b << 1)


def _expand_key128(key: bytes) -> list:
    assert len(key) == 16
    w = [bytearray(key[i:i+4]) for i in range(0, 16, 4)]
    for i in range(4, 44):
        t = bytearray(w[i-1])
        if i % 4 == 0:
            t = bytearray((_SBOX[t[1]] ^ _RCON[i//4 - 1], _SBOX[t[2]], _SBOX[t[3]], _SBOX[t[0]]))
        w.append(bytearray(a ^ b for a, b in zip(w[i-4], t)))
    # Round keys: 11 keys of 16 bytes each.
    return [bytes(b for word in w[r*4:(r+1)*4] for b in word) for r in range(11)]


def _sub_bytes(state: bytearray) -> None:
    for i in range(16):
        state[i] = _SBOX[state[i]]


def _shift_rows(state: bytearray) -> None:
    state[1], state[5], state[9], state[13] = state[5], state[9], state[13], state[1]
    state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
    state[3], state[7], state[11], state[15] = state[15], state[3], state[7], state[11]


def _mix_columns(state: bytearray) -> None:
    for c in range(4):
        i = c * 4
        a0, a1, a2, a3 = state[i], state[i+1], state[i+2], state[i+3]
        t = a0 ^ a1 ^ a2 ^ a3
        state[i]   ^= t ^ _xtime(a0 ^ a1)
        state[i+1] ^= t ^ _xtime(a1 ^ a2)
        state[i+2] ^= t ^ _xtime(a2 ^ a3)
        state[i+3] ^= t ^ _xtime(a3 ^ a0)


def _add_round_key(state: bytearray, rk: bytes) -> None:
    for i in range(16):
        state[i] ^= rk[i]


def encrypt_block(key: bytes, block: bytes) -> bytes:
    """Encrypt one 16-byte block with AES-128 ECB."""
    rks = _expand_key128(key)
    state = bytearray(block)
    _add_round_key(state, rks[0])
    for r in range(1, 10):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, rks[r])
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, rks[10])
    return bytes(state)


# Try the C-backed `cryptography` lib first — its AES implementation
# releases the GIL during the actual cipher work, so multiple concurrent
# decrypts (one per parallel download) don't starve the asyncio event
# loop. Pure-Python `encrypt_block` above stays as a fallback for
# environments without the lib (and is still used by the self-test).
try:
    from cryptography.hazmat.primitives.ciphers import (  # noqa: I001
        Cipher, algorithms, modes,
    )
    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False


def ctr_xor(key: bytes, iv_int: int, data: bytes, block_offset: int = 0) -> bytes:
    """AES-128 CTR. iv_int is the 128-bit big-endian counter start.
    block_offset shifts the counter by N blocks (for resuming/seeking).

    Uses OpenSSL via `cryptography` when available — orders of magnitude
    faster than pure Python and, more importantly, releases the GIL so
    parallel download workers don't starve the asyncio loop. Falls back
    to the pure-Python loop below if the lib's missing."""
    counter = iv_int + block_offset
    if _HAS_CRYPTOGRAPHY:
        nonce = counter.to_bytes(16, "big")
        cipher = Cipher(algorithms.AES(key), modes.CTR(nonce))
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    out = bytearray(len(data))
    for off in range(0, len(data), 16):
        ks = encrypt_block(key, counter.to_bytes(16, "big"))
        chunk = data[off:off+16]
        for i in range(len(chunk)):
            out[off+i] = chunk[i] ^ ks[i]
        counter = (counter + 1) & ((1 << 128) - 1)
    return bytes(out)


# Spotify uses this fixed nonce base for OGG audio file CTR mode.
SPOTIFY_AUDIO_IV = int.from_bytes(
    bytes.fromhex("72e067fbddcbcf77ebe8bc643f630d93"), "big"
)


def _self_test() -> None:
    # FIPS 197 Appendix B test vector.
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt  = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    ct  = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")
    got = encrypt_block(key, pt)
    assert got == ct, f"AES FIPS-197 vector failed: {got.hex()}"

    # NIST SP 800-38A AES-128 ECB vector 1.
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt  = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    ct  = bytes.fromhex("3ad77bb40d7a3660a89ecaf32466ef97")
    assert encrypt_block(key, pt) == ct

    # NIST SP 800-38A AES-128 CTR vector 1: F.5.1.
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    iv  = int.from_bytes(bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"), "big")
    pt  = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    ct  = bytes.fromhex("874d6191b620e3261bef6864990db6ce")
    assert ctr_xor(key, iv, pt) == ct
    print("AES self-test OK")


if __name__ == "__main__":
    _self_test()
