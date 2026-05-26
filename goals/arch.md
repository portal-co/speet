# Architecture Frontends

← [goals.md](../goals.md)

See [docs/guides/arch-recompilers.md](../docs/guides/arch-recompilers.md) for invariants when touching these crates.

---

## In progress

- [ ] **speet-x86_64** — FP and SIMD instructions currently stubbed as `unreachable`. Implement using `speet-wasm-helpers` for wide multiply variants; map SSE/AVX to WASM SIMD where possible.
- [ ] **speet-dex** — bring to feature parity with `speet-riscv`; structured control flow means slot granularity can be larger than 2 bytes.

## Planned

- [ ] **speet-aarch64** — decoder (`disarm64`) is already in workspace dependencies but unused. Start with the register file and branch instructions.
- [ ] **WASM-GC frontend** — urgent for Claude Code / jsaw + speet integration. Translates WASM GC bytecode to the linear-memory megabinary model.
- [ ] **speet-powerpc** — crate exists as a stub (`crates/native/speet-powerpc/src/lib.rs`). Classic Mac PPC and some embedded targets. No translation logic yet.

## Completed

- [x] speet-riscv — RV32I/RV64I + compressed (RVC); weak-memory ordering; speculative call lowering
- [x] speet-mips — MIPS32/64; delay slots; weak-memory ordering
