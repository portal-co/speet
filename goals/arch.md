# Architecture Frontends

← [goals.md](../goals.md)

See [docs/guides/arch-recompilers.md](../docs/guides/arch-recompilers.md) for invariants when touching these crates.

---

## In progress

- [ ] **speet-x86_64** — packed/vector SIMD (non-scalar XMM lanes) still `unreachable`; scalar SSE FP is done. Map remaining SSE/AVX to WASM SIMD where possible.
- [ ] **speet-dex** — bring to feature parity with `speet-riscv`; structured control flow means slot granularity can be larger than 2 bytes.

## Planned

- [ ] **WASM-GC frontend** — urgent for Claude Code / jsaw + speet integration. Translates WASM GC bytecode to the linear-memory megabinary model.
- [ ] **speet-powerpc** — crate exists as a stub (`crates/native/speet-powerpc/src/lib.rs`). Classic Mac PPC and some embedded targets. No translation logic yet.

## Completed

- [x] speet-riscv — RV32I/RV64I + compressed (RVC); weak-memory ordering; speculative Flag/Exception; thin-runtime `bind_memory_layout` / stub_for_pc / `plt_calling_convention` (`BinArch::RiscV64`)
- [x] speet-aarch64 — register file + branches; speculative Flag/Exception on BL/BLR/RET; SMULH/UMULH; thin-runtime PLT/layout surface
- [x] speet-mips — MIPS32/64; delay slots; weak-memory ordering
- [x] speet-x86_64 — integer + scalar SSE FP; speculative Flag/Exception; thin-runtime PLT/layout
