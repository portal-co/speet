#!/usr/bin/env python3
"""A constant-data TargetPlugin (the headline cross-host artifact — see
verification plan item 3): a reduced LinuxToWasi-equivalent declaring just
``proc_exit``/``fd_write`` and an empty syscall table. Mirrors the same
sample data the WASM and in-process legs of this test use, byte-for-byte.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from protocol import (  # noqa: E402
    KIND_MAIN_REQUEST,
    KIND_MAIN_RESPONSE,
    encode_string,
    encode_u32,
    encode_u8,
    read_frame,
    write_frame,
)


def encode_func_import(module, field, params, results):
    return (
        encode_string(module)
        + encode_string(field)
        + encode_u32(len(params))
        + b"".join(encode_u8(p) for p in params)
        + encode_u32(len(results))
        + b"".join(encode_u8(r) for r in results)
    )


def encode_module_manifest():
    out = encode_u32(2)
    out += encode_func_import("wasi_snapshot_preview1", "proc_exit", [0], [])
    out += encode_func_import("wasi_snapshot_preview1", "fd_write", [0, 0, 0, 0], [0])
    out += encode_u32(0) * 8  # globals, memories, tables, tags, memory_data,
    # passive_memory_data, element_segments, passive_element_segments — all empty
    return out


def encode_syscall_table():
    return encode_u32(0)  # entries: vec![]


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
            resp = encode_u8(0) + encode_module_manifest()
        elif tag == 1:
            resp = encode_u8(1) + encode_syscall_table()
        else:
            resp = b""
        write_frame(KIND_MAIN_RESPONSE, request_id, resp)


if __name__ == "__main__":
    main()
