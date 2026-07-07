#!/usr/bin/env python3
"""AddressMapperPlugin::translate with a dynamic argument: reads `addr_local`
out of the request and emits `local.get $addr_local` (opcode 0x20). Only
correct for `addr_local < 128` (single-byte LEB128) — fine for a fixture
proving the marshalling convention, not a general guest SDK.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from protocol import (  # noqa: E402
    KIND_MAIN_REQUEST,
    KIND_MAIN_RESPONSE,
    decode_u32,
    encode_bytes,
    encode_u8,
    read_frame,
    write_frame,
)


def main():
    while True:
        frame = read_frame()
        if frame is None:
            return
        kind, request_id, payload = frame
        if kind != KIND_MAIN_REQUEST:
            continue
        tag = payload[0]
        if tag == 0:
            addr_local, _ = decode_u32(payload, 1)
            wasm_bytes = bytes([0x20, addr_local & 0x7F])
            snippet = encode_bytes(wasm_bytes)
            presult_ok = encode_u8(0) + snippet
            resp = encode_u8(0) + presult_ok
        else:
            resp = b""
        write_frame(KIND_MAIN_RESPONSE, request_id, resp)


if __name__ == "__main__":
    main()
