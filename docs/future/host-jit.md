# Host JIT (asm→asm)

**Status: Planned** — see [future-features.md](../future-features.md).

## Purpose

Fast path for the [thin runtime](../thin-runtime-plan.md) when the guest is already native machine code: skip the WASM round-trip and translate **asm→asm** directly (speet frontends → wasm-blitz SysV or a future direct emitter).

## When to use

- Hot / on-the-fly recompilation where WASM validation and megabinary linking overhead is unnecessary.
- Same-platform binaries where register-file threading can stay in native form end-to-end.

## When not to use

- Container megabinary builds ([container-plan.md](../container-plan.md)) — pure WASM, no host JIT.
- Cross-ISA translation (still needs speet → WASM IR).

## Relationship to wasm-blitz v1

Phase 0 thin runtime uses wasm-blitz as the host JIT through WASM as an intermediate representation. This doc covers the **shortcut** that elides WASM when correctness preconditions are met.
