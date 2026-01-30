Binary Trailer Format and Manifest Schema

Goals
- Define a stable on-disk format for appending WASM/AOT blobs and manifest metadata to binaries.
- Ensure format supports versioning, signature verification, and tool inspection.

Trailer layout (appended trailer option)
- Footer structure placed after EOF:
  - 8 bytes: MAGIC "RCOMPv1\0" (null-terminated)
  - 4 bytes: trailer_version (uint32)
  - 8 bytes: manifest_len (uint64)
  - 8 bytes: blob_len (uint64)
  - manifest_len bytes: UTF-8 JSON manifest
  - blob_len bytes: WASM or AOT blob
  - 4 bytes: sig_key_id_len (uint32)
  - sig_key_id_len bytes: signer key id (UTF-8)
  - 4 bytes: sig_len (uint32)
  - sig_len bytes: signature (raw bytes, e.g., Ed25519)

Notes
- All multi-byte integers are little-endian.
- The loader scans the end of file for MAGIC and then reads trailer fields by offset. If MAGIC not found, treat binary as unmodified.
- Signatures cover the manifest and blob. The manifest must include the checksum of the original binary to bind the artifact.

Manifest JSON schema (example)
{
  "name": "example-service",
  "version": "0.1.0",
  "arch": "x86_64",
  "wasi_version": "wasi_snapshot_preview1",
  "vkernel_version": ">=0.1.0",
  "allowed_syscalls": ["read","write","open","close","socket","connect","accept","epoll_wait","futex_wait","futex_wake"],
  "entrypoint": "_start",
  "original_checksum": "sha256:...",
  "build_info": {
    "toolchain_hash": "...",
    "build_date": "2026-01-30T...Z"
  }
}

Versioning
- Increase trailer_version on incompatible trailer format changes.
- Include both wasi_version and vkernel_version for compatibility checks.

Signature
- Use Ed25519 or similar modern signature scheme for simplicity and small size.
- Signer keys should be managed in CI/KMS and only public keys kept on host nodes.

Tooling
- Provide CLI tools to append, extract, and verify trailers.
- Allow option to embed trailer as ELF section instead of appended trailer for advanced use.

Next steps
- Finalize trailer binary layout and implement loader parsing for appended trailers in prototype.