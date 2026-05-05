"""Shannon stream cipher used by Spotify.

This is a custom MAC-and-stream cipher (essentially a 16-word LFSR variant).
Adapted from librespot-python (which is itself adapted from librespot-org).
Pure stdlib.
"""

import struct


class Shannon:
    n = 16
    fold = n
    initkonst = 0x6996c53a
    keyp = 13

    def __init__(self):
        self.r = [0] * self.n
        self.crc = [0] * self.n
        self.init_r = [0] * self.n
        self.konst = 0
        self.sbuf = 0
        self.mbuf = 0
        self.nbuf = 0

    @staticmethod
    def rotl(i: int, distance: int) -> int:
        return ((i << distance) | (i >> (32 - distance))) & 0xffffffff

    def sbox(self, i: int) -> int:
        i ^= self.rotl(i, 5) | self.rotl(i, 7)
        i ^= self.rotl(i, 19) | self.rotl(i, 22)
        return i & 0xffffffff

    def sbox2(self, i: int) -> int:
        i ^= self.rotl(i, 7) | self.rotl(i, 22)
        i ^= self.rotl(i, 5) | self.rotl(i, 19)
        return i & 0xffffffff

    def cycle(self) -> None:
        t = self.r[12] ^ self.r[13] ^ self.konst
        t = self.sbox(t) ^ self.rotl(self.r[0], 1)
        for i in range(1, self.n):
            self.r[i - 1] = self.r[i]
        self.r[self.n - 1] = t & 0xffffffff
        t = self.sbox2(self.r[2] ^ self.r[15])
        self.r[0] = (self.r[0] ^ t) & 0xffffffff
        self.sbuf = (t ^ self.r[8] ^ self.r[12]) & 0xffffffff

    def crc_func(self, i: int) -> None:
        t = self.crc[0] ^ self.crc[2] ^ self.crc[15] ^ i
        for j in range(1, self.n):
            self.crc[j - 1] = self.crc[j]
        self.crc[self.n - 1] = t & 0xffffffff

    def mac_func(self, i: int) -> None:
        self.crc_func(i)
        self.r[self.keyp] = (self.r[self.keyp] ^ i) & 0xffffffff

    def init_state(self) -> None:
        self.r[0] = 1
        self.r[1] = 1
        for i in range(2, self.n):
            self.r[i] = (self.r[i - 1] + self.r[i - 2]) & 0xffffffff
        self.konst = self.initkonst

    def save_state(self) -> None:
        self.init_r = list(self.r)

    def reload_state(self) -> None:
        self.r = list(self.init_r)

    def gen_konst(self) -> None:
        self.konst = self.r[0]

    def add_key(self, k: int) -> None:
        self.r[self.keyp] = (self.r[self.keyp] ^ k) & 0xffffffff

    def diffuse(self) -> None:
        for _ in range(self.fold):
            self.cycle()

    def load_key(self, key: bytes) -> None:
        padding_size = (4 - (len(key) % 4)) % 4
        key = key + (b"\x00" * padding_size) + struct.pack("<I", len(key))
        for i in range(0, len(key), 4):
            self.r[self.keyp] = (
                self.r[self.keyp] ^ struct.unpack("<I", key[i:i + 4])[0]
            ) & 0xffffffff
            self.cycle()
        for i in range(self.n):
            self.crc[i] = self.r[i]
        self.diffuse()
        for i in range(self.n):
            self.r[i] = (self.r[i] ^ self.crc[i]) & 0xffffffff

    def key(self, key: bytes) -> None:
        self.init_state()
        self.load_key(key)
        self.gen_konst()
        self.save_state()
        self.nbuf = 0

    def nonce(self, nonce) -> None:
        if isinstance(nonce, int):
            nonce = struct.pack(">I", nonce)
        self.reload_state()
        self.konst = self.initkonst
        self.load_key(nonce)
        self.gen_konst()
        self.nbuf = 0

    def encrypt(self, buffer: bytes, n: int = None) -> bytes:
        if n is None:
            n = len(buffer)
        buffer = bytearray(buffer)
        i = 0
        if self.nbuf != 0:
            while self.nbuf != 0 and n != 0:
                self.mbuf ^= (buffer[i] & 0xff) << (32 - self.nbuf)
                buffer[i] ^= (self.sbuf >> (32 - self.nbuf)) & 0xff
                i += 1
                self.nbuf -= 8
                n -= 1
            if self.nbuf != 0:
                return b""
            self.mac_func(self.mbuf)
        j = n & ~0x03
        while i < j:
            self.cycle()
            t = (
                ((buffer[i + 3] & 0xff) << 24)
                | ((buffer[i + 2] & 0xff) << 16)
                | ((buffer[i + 1] & 0xff) << 8)
                | (buffer[i] & 0xff)
            )
            self.mac_func(t)
            t ^= self.sbuf
            buffer[i + 3] = (t >> 24) & 0xff
            buffer[i + 2] = (t >> 16) & 0xff
            buffer[i + 1] = (t >> 8) & 0xff
            buffer[i] = t & 0xff
            i += 4
        n &= 0x03
        if n != 0:
            self.cycle()
            self.mbuf = 0
            self.nbuf = 32
            while self.nbuf != 0 and n != 0:
                self.mbuf ^= (buffer[i] & 0xff) << (32 - self.nbuf)
                buffer[i] ^= (self.sbuf >> (32 - self.nbuf)) & 0xff
                i += 1
                self.nbuf -= 8
                n -= 1
        return bytes(buffer)

    def decrypt(self, buffer: bytes, n: int = None) -> bytes:
        if n is None:
            n = len(buffer)
        buffer = bytearray(buffer)
        i = 0
        if self.nbuf != 0:
            while self.nbuf != 0 and n != 0:
                buffer[i] ^= (self.sbuf >> (32 - self.nbuf)) & 0xff
                self.mbuf ^= (buffer[i] & 0xff) << (32 - self.nbuf)
                i += 1
                self.nbuf -= 8
                n -= 1
            if self.nbuf != 0:
                return b""
            self.mac_func(self.mbuf)
        j = n & ~0x03
        while i < j:
            self.cycle()
            t = (
                ((buffer[i + 3] & 0xff) << 24)
                | ((buffer[i + 2] & 0xff) << 16)
                | ((buffer[i + 1] & 0xff) << 8)
                | (buffer[i] & 0xff)
            )
            t ^= self.sbuf
            self.mac_func(t)
            buffer[i + 3] = (t >> 24) & 0xff
            buffer[i + 2] = (t >> 16) & 0xff
            buffer[i + 1] = (t >> 8) & 0xff
            buffer[i] = t & 0xff
            i += 4
        n &= 0x03
        if n != 0:
            self.cycle()
            self.mbuf = 0
            self.nbuf = 32
            while self.nbuf != 0 and n != 0:
                buffer[i] ^= (self.sbuf >> (32 - self.nbuf)) & 0xff
                self.mbuf ^= (buffer[i] & 0xff) << (32 - self.nbuf)
                i += 1
                self.nbuf -= 8
                n -= 1
        return bytes(buffer)

    def finish(self, n: int) -> bytes:
        buffer = bytearray(4)
        i = 0
        if self.nbuf != 0:
            self.mac_func(self.mbuf)
        self.cycle()
        self.add_key(self.initkonst ^ (self.nbuf << 3))
        self.nbuf = 0
        for j in range(self.n):
            self.r[j] = (self.r[j] ^ self.crc[j]) & 0xffffffff
        self.diffuse()
        while n > 0:
            self.cycle()
            if n >= 4:
                buffer[i + 3] = (self.sbuf >> 24) & 0xff
                buffer[i + 2] = (self.sbuf >> 16) & 0xff
                buffer[i + 1] = (self.sbuf >> 8) & 0xff
                buffer[i] = self.sbuf & 0xff
                n -= 4
                i += 4
            else:
                for j in range(n):
                    buffer[i + j] = (self.sbuf >> (i * 8)) & 0xff
                break
        return bytes(buffer)


def _self_test():
    # Round-trip: encrypt, then decrypt with fresh state.
    key = b"test-key-1234567"
    nonce_val = 0
    plaintext = b"The quick brown fox jumps over the lazy dog!!!!"

    enc = Shannon()
    enc.key(key)
    enc.nonce(nonce_val)
    ciphertext = enc.encrypt(plaintext)
    enc_mac = enc.finish(4)

    dec = Shannon()
    dec.key(key)
    dec.nonce(nonce_val)
    decrypted = dec.decrypt(ciphertext)
    dec_mac = dec.finish(4)

    assert decrypted == plaintext, decrypted
    assert dec_mac == enc_mac, (dec_mac.hex(), enc_mac.hex())
    print("Shannon self-test OK")


if __name__ == "__main__":
    _self_test()
