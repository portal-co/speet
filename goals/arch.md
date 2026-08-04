# Architecture Frontends

← [goals.md](../goals.md)

See [docs/guides/arch-recompilers.md](../docs/guides/arch-recompilers.md) for invariants when touching these crates.

---

## Main four (implemented)

These native frontends are the supported ISA set. They share the register-file WASM ABI, `return_call` control flow, and (where noted) speculative Flag/Exception + thin-runtime surfaces. Residual gaps are listed under In progress / Planned — not “frontend missing.”

| Frontend | Baseline | Speculative Flag/Exception | Thin-runtime (`bind_memory_layout` / stub_for_pc / plt_cc) |
|----------|----------|----------------------------|-------------------------------------------------------------|
| **speet-aarch64** | Register file, branches, DP_3SRC incl. SMULH/UMULH | Yes (BL/BLR/RET) | Yes |
| **speet-riscv** | RV32I/RV64I + RVC; weak-memory | Yes (jal/jalr/ret) | Yes (`BinArch::RiscV64`) |
| **speet-x86_64** | Integer + scalar SSE FP | Yes (call/ret) | Yes |
| **speet-mips** | MIPS32/64; weak-memory; delay slots | Yes (jal/jalr/jr $ra) | Yes (`bind_memory_layout` / stub_for_pc / `plt_calling_convention`) |

## PowerPC — not started

**speet-powerpc** is a stub crate (`crates/native/speet-powerpc/src/lib.rs`) with no translation logic. Once started, it should aim for the same surface as the main four: register-file ABI `(regs) -> (regs)`, speculative Flag/Exception, thin-runtime bind/stub/plt, and trap hooks. (wasm-blitz `blitz-ppc64` is a separate WIP backend and does not imply speet frontend progress.)

## 32-bit expansion

RV32 guest **input** is already covered by `speet-riscv` (`Xlen::Rv32`). The following expand both **input** (speet frontends) and **output** (asm-arch writers + wasm-blitz ILP32 backends + `BinArch`):

| Target | Speet input | asm-arch | blitz | BinArch |
|--------|-------------|----------|-------|---------|
| **riscv32** | Done (`speet-riscv`) | `asm-riscv32` | `blitz-riscv32` | `RiscV32` |
| **arm32** (AArch32 ARMv7-A) | Done thin (`speet-arm`, A32 primary; Thumb stub) | `asm-arm` (A32 emit) | `blitz-arm` | `Arm` |
| **x86-32** (i686) | Done thin (`speet-x86`) | `asm-x86` | `blitz-i686` | `X86` |

WASM value slots stay **8 bytes** on ILP32 hosts; host pointers use 4 bytes. Thin-runtime execute on 32-bit hosts is Linux-only when toolchains exist.

**Phase 4 smoke:** `speet-arm` / `speet-x86` wired through `BinArch::{Arm,X86}` in `speet-recompile::frontend`; FlagSpec e2e (`arm_flag_smoke` / `x86_flag_smoke`) + crate unit tests translate handcrafted integer/control sequences. Thumb-2 decode and full ILP32 thin-runtime parity remain follow-ups.

---

## In progress

- [ ] **speet-x86_64** — packed/vector SIMD (non-scalar XMM lanes) still `unreachable`; scalar SSE FP is done. Map remaining SSE/AVX to WASM SIMD where possible.
- [ ] **speet-dex** — bring to feature parity with `speet-riscv`; structured control flow means slot granularity can be larger than 2 bytes. (Managed bytecode — separate from the main four.)
- [ ] **speet-arm** — expand beyond A32 smoke (Thumb-2 decode, fuller DP/LDR/STR, thin-runtime PLT parity).
- [ ] **speet-x86** — expand beyond i686 smoke (flags/Jcc, fuller addressing, thin-runtime PLT parity).

## Planned

- [ ] **WASM-GC frontend** — urgent for Claude Code / jsaw + speet integration. Translates WASM GC bytecode to the linear-memory megabinary model.
- [ ] **speet-powerpc** — start translation with main-four parity goals (see above).

## Completed (baseline)

- [x] speet-riscv — RV32I/RV64I + RVC; weak-memory; speculative Flag/Exception; thin-runtime PLT/layout
- [x] speet-aarch64 — register file + branches; speculative Flag/Exception; SMULH/UMULH; thin-runtime PLT/layout
- [x] speet-mips — MIPS32/64; delay slots; weak-memory; speculative Flag/Exception; thin-runtime bind/stub/plt
- [x] speet-x86_64 — integer + scalar SSE FP; speculative Flag/Exception; thin-runtime PLT/layout
- [x] speet-arm — thin A32 frontend + FlagSpec smoke (`BinArch::Arm`)
- [x] speet-x86 — thin i686 frontend + FlagSpec smoke (`BinArch::X86`)
- [x] `BinArch::RiscV32` frontend path via `speet-riscv` (`Xlen::Rv32`)
