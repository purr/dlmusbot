"""Minimal Protocol Buffers wire-format decoder.

Returns a dict keyed by field number whose values are lists (because protobuf
fields can repeat). Each value is one of: int (varint / fixed32 / fixed64),
bytes (length-delimited), depending on wire type. Nested messages stay raw
bytes; the caller calls decode() again on them.
"""

VARINT = 0
FIXED64 = 1
LEN = 2
FIXED32 = 5


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")


def decode(buf: bytes) -> dict:
    """Decode a protobuf message. Returns {field_no: [values...]}."""
    out: dict = {}
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_no = tag >> 3
        wire = tag & 7
        if wire == VARINT:
            val, pos = _read_varint(buf, pos)
        elif wire == FIXED64:
            val = int.from_bytes(buf[pos : pos + 8], "little")
            pos += 8
        elif wire == LEN:
            length, pos = _read_varint(buf, pos)
            val = buf[pos : pos + length]
            pos += length
        elif wire == FIXED32:
            val = int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        else:
            raise ValueError(f"unknown wire type {wire}")
        out.setdefault(field_no, []).append(val)
    return out


def get_str(msg: dict, field_no: int):
    v = msg.get(field_no)
    return v[0].decode("utf-8") if v else None


def get_bytes(msg: dict, field_no: int):
    v = msg.get(field_no)
    return v[0] if v else None


def get_int(msg: dict, field_no: int):
    v = msg.get(field_no)
    return v[0] if v else None


def get_sint(msg: dict, field_no: int):
    """Decode a sint32/sint64 (zigzag-encoded varint).
    Spotify's Track-level numeric fields (number, popularity, duration,
    date.year/month/day, ...) are wire-encoded this way even though declared
    int32 in older proto definitions."""
    v = msg.get(field_no)
    if not v:
        return None
    raw = v[0]
    return (raw >> 1) ^ -(raw & 1)


def get_list(msg: dict, field_no: int):
    return msg.get(field_no, [])


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_field(field_no: int, wire_type: int, value) -> bytes:
    """Encode one field. value type depends on wire_type:
    VARINT: int
    LEN: bytes
    FIXED32: int
    FIXED64: int
    """
    tag = _encode_varint((field_no << 3) | wire_type)
    if wire_type == VARINT:
        return tag + _encode_varint(value)
    if wire_type == LEN:
        return tag + _encode_varint(len(value)) + value
    if wire_type == FIXED32:
        return tag + value.to_bytes(4, "little")
    if wire_type == FIXED64:
        return tag + value.to_bytes(8, "little")
    raise ValueError(f"unsupported wire type {wire_type}")


def encode(fields) -> bytes:
    """fields = list of (field_no, wire_type, value)."""
    out = bytearray()
    for fn, wt, v in fields:
        out += encode_field(fn, wt, v)
    return bytes(out)


def _self_test():
    # Hand-encoded message: field 1 varint = 150, field 2 string = "test".
    raw = bytes(
        [
            0x08,
            0x96,
            0x01,  # field 1, varint 150
            0x12,
            0x04,
            0x74,
            0x65,
            0x73,
            0x74,  # field 2, length-delim "test"
        ]
    )
    m = decode(raw)
    assert m[1] == [150], m
    assert m[2] == [b"test"], m
    assert get_str(m, 2) == "test"
    print("protobuf self-test OK")


if __name__ == "__main__":
    _self_test()
