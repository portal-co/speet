# Direct Linking

**Status: Planned** — see [future-features.md](../future-features.md).

## Purpose

Link recompiled guest `.o` + `speet-rt` + HostApi shim **in-process** without spawning `clang`/`lld` as external commands. Enables embedding the thin runtime inside a long-lived supervisor.

## Scope

- Rust linker driver wrapping `lld` APIs or an in-tree reloc resolver.
- No temp executable on disk for the link step (output may still be written for `execve` spawn in Phase 0 style).

## Non-goals

- Replacing `binary-io` object emission (guest `.o` still comes from wasm-blitz).
- Container megabinary linking (that path stays WASM-native inside vkernel).
