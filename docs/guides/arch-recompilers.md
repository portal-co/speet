# Architecture Recompilers Component Guide

**Crates:** `crates/native/speet-x86_64`, `crates/native/speet-aarch64`, `crates/native/speet-riscv`, `crates/native/speet-mips`, `crates/native/speet-powerpc`, `crates/managed/speet-dex`  
**Design doc:** [recompiler-guide.md](../recompiler-guide.md)  
**Status:** [goals/arch.md](../../goals/arch.md) — **main four** (aarch64, riscv, x86_64, mips) are implemented; **powerpc** is not started (stub only; should gain main-four parity once started). DEX is managed bytecode, tracked separately.

---

## 1. `Ctx` parameter passed down to raw targets

**Code:** `setup_traps` in all recompiler crates

All recompiler `setup_traps` and translate methods accept a `ctx: &mut Context` parameter. Even though default or vanilla targets (like `wasm_encoder::Function`) might pass a dummy `()` or ignore this parameter, it is a strict requirement for other `wax-core` recompile targets. These custom targets use the context to manage parallel instruction emission and coordinate state across multiple threads.

**Do not** remove the `ctx: &mut Context` parameter from `setup_traps` or any other recompiler API signatures. Ensure `Ctx` is passed all the way down to the raw target.

---

## 2. `return_call` for every control-flow edge

Each control-flow edge in the translated output is a `return_call` to the target slot's function (see [yecta.md](yecta.md) §1b). Architecture recompilers must use `TrapContext::jump` / `TrapContext::jump_if` (or `Reactor::jmp` directly) for all jumps — not inline branch sequences or a single monolithic loop.

**Do not** restructure multiple instructions into a single loop-based translation function. The one-function-per-slot model and O(1) stack depth depend on `return_call` chains.

---

## 3. Trap integration (in progress)

The `TrapConfig` field and `setup_traps` call sites are present in all architecture recompilers as stubs. Per-instruction and per-jump trap firing is specified in [trap-hooks.md](../trap-hooks.md) §8 and tracked in [goals/active.md](../../goals/active.md).

**Do not** remove the `TrapConfig` stubs — they are load-bearing placeholders for the pending integration.

---

## 4. speet-riscv — RISC-V specifics

**Code:** `crates/native/speet-riscv/src/`

- **Compressed instructions (RVC)**: slots exist at every 2-byte offset, including inside 4-byte instructions. See [yecta.md](yecta.md) §1a.
- **Weak memory model**: loads and stores use `speet-ordering` with `MemOrder::Relaxed` for ordinary memory accesses and `MemOrder::SeqCst` for fence instructions. Deferred stores use `feed_lazy`. See [yecta.md](yecta.md) §2.
- **Direct calls**: ABI-compliant `jal x1`/`jalr x1` are lowered to speculative WASM `call`s (`CallEscape::Flag` / Exception). See [yecta.md](yecta.md) §3.
- **Thin-runtime surface**: `bind_memory_layout`, `set_stub_for_pc_import_idx`, and `plt_calling_convention` mirror x86/aarch64 so RV guests share HostOffset/ZeroOffset + PLT shims (`BinArch::RiscV64`).

---

## 4b. speet-aarch64 — AArch64 specifics

**Code:** `crates/native/speet-aarch64/src/`

- **Fixed-width slots**: 4-byte granularity; thin frontend uses `AArch64CfgDecoder` (not a fallthrough-only stub) for `PcSlotMap`.
- **Speculative calls**: ABI `BL`/`BLR` + `RET` support `CallEscape::Flag` (primary; wasmi-safe) and Exception when a tag is wired. Expected RA lives in guest `LR` (x30) plus a hidden `expected_ra` param. Conditional branches stay Jump-only.
- **DP_3SRC**: `MADD`/`MSUB`/`SMADDL`/`UMADDL`/`SMSUBL`/`UMSUBL` plus `SMULH`/`UMULH` via `speet-wasm-helpers` mulh sequences.

---

## 5. speet-mips — MIPS specifics

**Code:** `crates/native/speet-mips/src/`

MIPS uses the same weak-memory ordering as RISC-V (see §4 above). Branch/jump targets use `pc+8` (skipping the architectural delay slot); the delay-slot instruction is a separate decode slot when reachable — do not invent a second emission of that instruction at the branch site.

**Speculative / thin-runtime:** `CallEscape::Flag` (+ Exception) on ABI `jal`/`jalr`/`jr $ra`; `bind_memory_layout` / stub_for_pc / `plt_calling_convention` match the other main-four frontends.

---

## 6. speet-x86_64 — x86-64 specifics

**Code:** `crates/native/speet-x86_64/src/`

Scalar SSE FP is implemented: XMM0–15 are modeled as raw-bits `i64` locals (`xmm_slot`,
`BASE_PARAMS = 42`), and FP handlers reinterpret i64→f64/f32 around native WASM FP ops and back.
This covers the scalar arith/move/compare/convert families asm-arch's backend can emit — see
[asm-arch-instruction-sync.md](asm-arch-instruction-sync.md) for the full per-instruction
matrix. Packed/vector SIMD (the `xmm` registers' non-scalar lanes) is not modeled and still
falls through to `unreachable`.

**Do not** remove the `xmm_slot` register file or the i64 bit-reinterpret pattern — other FP
handlers depend on the exact-bits round-trip it provides (NaN payloads, f32 low half).

---

## 7. speet-dex — Dalvik/DEX specifics

**Code:** `crates/managed/speet-dex/src/`

DEX is a register-based managed bytecode. Unlike native ISAs, DEX has structured control flow (no computed gotos at the bytecode level), which means the one-slot-per-possible-address model can use larger slot granularity. However, `return_call` is still used for all edges to maintain the consistent O(1) stack-depth contract.

---

## 8. speet-powerpc — PowerPC (not started)

`crates/native/speet-powerpc/src/lib.rs` is a stub with no translation logic. Once started, target the same surface as the main four (register-file ABI, speculative Flag/Exception, thin-runtime bind/stub/plt, trap hooks). See [goals/arch.md](../../goals/arch.md).
