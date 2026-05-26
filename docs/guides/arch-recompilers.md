# Architecture Recompilers Component Guide

**Crates:** `crates/native/speet-x86_64`, `crates/native/speet-riscv`, `crates/native/speet-mips`, `crates/native/speet-powerpc`, `crates/managed/speet-dex`  
**Design doc:** [recompiler-guide.md](../recompiler-guide.md)

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
- **Direct calls**: ABI-compliant `jal x1`/`jalr x1` are lowered to speculative WASM `call`s. See [yecta.md](yecta.md) §3.

---

## 5. speet-mips — MIPS specifics

**Code:** `crates/native/speet-mips/src/`

MIPS uses the same weak-memory ordering as RISC-V (see §4 above). MIPS has delay slots — the instruction after a branch always executes before the branch takes effect. The recompiler handles this by emitting the delay-slot instruction inline before the `return_call`.

**Do not** skip delay-slot emission or move the `return_call` before the delay-slot instruction.

---

## 6. speet-x86_64 — x86-64 specifics

**Code:** `crates/native/speet-x86_64/src/`

FP and SIMD instructions are currently stubbed as `unreachable`. This is intentional — the stubs mark what remains to be implemented. See [goals/arch.md](../../goals/arch.md).

**Do not** remove the `unreachable` stubs silently — replace them with implementations when adding FP/SIMD support.

---

## 7. speet-dex — Dalvik/DEX specifics

**Code:** `crates/managed/speet-dex/src/`

DEX is a register-based managed bytecode. Unlike native ISAs, DEX has structured control flow (no computed gotos at the bytecode level), which means the one-slot-per-possible-address model can use larger slot granularity. However, `return_call` is still used for all edges to maintain the consistent O(1) stack-depth contract.

---

## 8. speet-powerpc — PowerPC (stub)

`crates/native/speet-powerpc/src/lib.rs` is a stub crate with no translation logic. It is a name reservation for future implementation. See [goals/arch.md](../../goals/arch.md).
