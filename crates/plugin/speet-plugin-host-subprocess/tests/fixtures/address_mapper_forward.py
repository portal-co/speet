#!/usr/bin/env python3
"""A forwarding AddressMapperPlugin that delegates `translate` to a
host-granted import named "host-mmu" (§2.7's subprocess realization).
Forwards the request verbatim — the imported plugin's response shape
(`AddressMapperResponse::Translate`) is exactly what this fixture's own
`MainResponse` needs to be. If the import resolves to nothing (denied —
an empty `ImportResponse` payload, the documented sentinel), returns a
pre-built `Err(PluginError)` response instead of forwarding garbage —
what a real, well-behaved plugin should do.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from protocol import (  # noqa: E402
    KIND_IMPORT_REQUEST,
    KIND_MAIN_REQUEST,
    KIND_MAIN_RESPONSE,
    encode_string,
    encode_u32,
    encode_u8,
    read_frame,
    write_frame,
)

ROLE_ADDRESS_MAPPER = 1


def denied_response():
    err = encode_u32(1) + encode_string("import denied")
    presult_err = encode_u8(1) + err
    return encode_u8(0) + presult_err


def main():
    next_import_id = 1000
    while True:
        frame = read_frame()
        if frame is None:
            return
        kind, request_id, payload = frame
        if kind != KIND_MAIN_REQUEST:
            continue

        import_payload = encode_u8(ROLE_ADDRESS_MAPPER) + encode_string("host-mmu") + payload
        write_frame(KIND_IMPORT_REQUEST, next_import_id, import_payload)
        next_import_id += 1

        resp_frame = read_frame()
        if resp_frame is None:
            return
        _, _, resp_payload = resp_frame

        resp = resp_payload if len(resp_payload) > 0 else denied_response()
        write_frame(KIND_MAIN_RESPONSE, request_id, resp)


if __name__ == "__main__":
    main()
