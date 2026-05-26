# speet-interp

**Crate:** `crates/helper/speet-interp`  
**Status: Active implementation.**

Generates out-of-bounds (OOB) jump dispatch tables and Thompson-threaded interpreter stubs. Used when a computed jump target at runtime is not covered by the statically translated slot range.

---

## Purpose

The one-function-per-slot model (see [docs/guides/yecta.md §1](guides/yecta.md)) translates a fixed range of guest instructions. If a computed branch at runtime jumps to an address outside that range, the fallback is:

1. A **binary-search dispatch table** that maps out-of-range PCs to translated slot indices (for common indirect targets that the recompiler can identify statically).
2. A **Thompson-threaded interpreter stub** for truly unknown targets — interprets one instruction at a time, then chains back into the translated binary when it reaches a known slot.

---

## Modules

- `builder` — constructs the binary-search dispatch table as a WASM `br_table` + comparison sequence
- `context` — holds the dispatch table and interpreter state shared across the translated binary
- `lib` — public API: `emit_oob_dispatch`, `emit_interpreter_stub`

---

## Integration

Architecture frontends call `speet-interp` when emitting the indirect-jump fallback path. The interpreter stub is wired as the default branch in the `br_table` dispatch. `context` is stored in the linker context and consulted at the end of translation when all static targets are known.
