WASM Appending Technology (Per-Binary)

Goals
- Produce a self-contained artifact per binary that includes the original binary (optional), a signed WASM or AOT blob, and a manifest describing runtime requirements.
- Provide a loader stub that verifies artifact integrity and launches the WASM/AOT module using the host vkernel.
- Use a stable, auditable on-disk format that can be parsed by loaders and tooling.

Artifact format
- Option A: ELF integrated section
  - Add a custom ELF section (.recompiler or .wasm_blob) containing a trailer with magic, manifest JSON, WASM/AOT blob, and signature.
  - Pros: clean integration with existing ELF semantics; can be inspected with readelf and tools.
  - Cons: requires ELF modification tooling and careful handling of dynamic loaders.

- Option B: Appended trailer
  - Append a trailer after EOF with a magic header and offsets: MAGIC | manifest_len | manifest | blob_len | blob | signature_len | signature
  - Pros: simpler to implement, works across formats.
  - Cons: some tools may not expect appended data; must ensure loader uses rpath or interpreter correctly.

Manifest contents (JSON)
- name: human readable name
- version: toolchain version that produced artifact
- arch: target architecture (x86_64, aarch64)
- wasi_version: required WASI version
- vkernel_version: minimum vkernel version required
- allowed_syscalls: optional explicit allow list
- entrypoint: main symbol or WASM _start symbol
- original_checksum: checksum of original binary (optional)
- signatures: signer key id and signature blob
- build_info: reproducible build metadata (toolchain hash, build date)

Loader stub
- Small native ELF that runs as the container's process entrypoint.
- Responsibilities:
  - Detect appended artifact and parse manifest
  - Verify signature against trusted keys (configured on host)
  - Connect to vkernel and negotiate capabilities
  - Initialize WASM runtime (interpreter or AOT loader) with vkernel-backed WASI
  - Start the module and manage process lifecycle and signals

Loader behavior details
- If original native code is present and a signed AOT artifact exists, loader may run AOT code directly (subject to policy). Otherwise, run WASM module.
- Refuse to run unsigned or malformed artifacts.
- Support debug mode to dump manifest and blob for dev workflows.

Versioning and backwards compatibility
- Include a format version in the trailer header to allow future changes.
- Ensure loader gracefully rejects newer-format artifacts if unsupported.

Tooling
- Provide CLI utilities to: inspect artifacts, append blobs to binaries, extract blobs, verify signatures, and replace loader stubs.
- Integrate with CI to produce appended artifacts automatically during image build.

Security considerations
- Strong signatures and key management on hosts are critical; vkernel must map signer keys to allowed capabilities.
- Avoid carrying private keys in images; only sign artifacts in build pipeline and verify in runtime.

Next steps
- Define exact binary trailer structure and manifest JSON schema (see 14-formats.md).
- Prototype loader stub in C that reads trailing trailer and launches a tiny WASM runtime.