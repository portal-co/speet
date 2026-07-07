"""Minimal, hand-rolled implementation of the frame protocol documented in
``speet_plugin_host_subprocess::frame`` and the wire codec documented in
``speet_plugin_api::wire`` — just enough for these test fixtures to speak
the protocol, not a general-purpose plugin SDK. Deliberately not Rust, to
demonstrate the subprocess host's language-neutrality (see
docs/guides/plugin-api.md §5.3's verification plan).

Frame: [1 byte version][1 byte kind][4 bytes LE request_id][4 bytes LE
payload_len][payload]. kind: 0=MainRequest 1=MainResponse 2=ImportRequest
3=ImportResponse.
"""
import struct
import sys

VERSION = 1
KIND_MAIN_REQUEST = 0
KIND_MAIN_RESPONSE = 1
KIND_IMPORT_REQUEST = 2
KIND_IMPORT_RESPONSE = 3

_stdin = sys.stdin.buffer
_stdout = sys.stdout.buffer


def read_frame():
    header = _stdin.read(2)
    if len(header) < 2:
        return None
    version, kind = header[0], header[1]
    request_id = struct.unpack("<I", _stdin.read(4))[0]
    length = struct.unpack("<I", _stdin.read(4))[0]
    payload = _stdin.read(length) if length else b""
    return (kind, request_id, payload)


def write_frame(kind, request_id, payload):
    _stdout.write(bytes([VERSION, kind]))
    _stdout.write(struct.pack("<I", request_id))
    _stdout.write(struct.pack("<I", len(payload)))
    _stdout.write(payload)
    _stdout.flush()


# ── Wire primitives (subset of speet_plugin_api::wire used by these fixtures) ──

def encode_u8(v):
    return bytes([v & 0xFF])


def encode_u32(v):
    return struct.pack("<I", v)


def encode_bytes(b):
    return encode_u32(len(b)) + b


def encode_string(s):
    return encode_bytes(s.encode("utf-8"))


def encode_option_none():
    return bytes([0])


def decode_u8(buf, off):
    return buf[off], off + 1


def decode_u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0], off + 4


def decode_bytes(buf, off):
    n, off = decode_u32(buf, off)
    return buf[off:off + n], off + n


def decode_string(buf, off):
    b, off = decode_bytes(buf, off)
    return b.decode("utf-8"), off
