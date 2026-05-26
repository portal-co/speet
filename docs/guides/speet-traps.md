# speet-traps Component Guide

**Crates:** `crates/helper/speet-traps`  
**Design doc:** [trap-hooks.md](../trap-hooks.md) §3.1, §3.5, §9  
**Related:** [parallel-api.md](parallel-api.md) (hooks and interior mutability)

---

## 1. Trap state in parameters, not locals

**Code:** `crates/helper/speet-traps/src/hardening.rs` (`RopDetectTrap`), `crates/helper/speet-traps/src/security.rs` (`CfiReturnTrap`)

WASM non-parameter locals reset to zero at every function boundary. Because each guest instruction slot is its own WASM function (see [yecta.md](yecta.md) §1), state stored in a non-param local is silently lost when the `return_call` chain advances.

`RopDetectTrap` stores its call/return depth counter in a WASM **parameter** so it survives across the chain. `CfiReturnTrap` uses a **local** for its bitmap-index scratch because that scratch is only needed within the body of a single function.

The two-phase `declare_params` / `declare_locals` protocol in `TrapConfig` exists because WASM function types (which include parameters) must be declared before any function body is emitted, while locals are declared per-function.

**Do not** move `RopDetectTrap`'s depth counter to a local or a global. A local would reset to zero on every `return_call`; a global would require external coordination.

---

## 2. `EmitSink` trait for jump emission in `TrapContext`

**Code:** `crates/helper/yecta/src/lib.rs` (`EmitSink`), `crates/helper/speet-traps/src/context.rs` (`TrapContext`)  
**Design doc:** [trap-hooks.md](../trap-hooks.md) §3.5

`TrapContext::jump` and `TrapContext::jump_if` call `EmitSink::emit_jmp` on the underlying (type-erased) sink. When the sink is a `Reactor`, this delegates to `Reactor::jmp` with full predecessor-graph bookkeeping. The trap traits (`InstructionTrap`, `JumpTrap`, `TrapConfig`) carry no `F` type parameter; `TrapContext` holds `&mut dyn EmitSink<Context, E>` instead.

**Do not** add `F` back to the trap traits. The `dyn EmitSink` indirection is intentional — it allows traps to be used as `dyn InstructionTrap<Context, E>` trait objects without caring about the concrete sink type.

---

## 3. `FuncSignature`: injected params mirror as returns

**Code:** `crates/helper/wasm-layout/src/lib.rs` (`FuncSignature`)  
**Design doc:** [func-signature.md](../func-signature.md) §2–§5

`FuncSignature` pairs a `LocalLayout` (params) with a `Vec<ValType>` of return types. The returns are exactly the injected/trap params — those declared after the `injected_start` mark — mirrored back. Arch params are not returned; they travel forward via `return_call`.

This gives every translated function the type `(arch_params + injected) -> (injected)`. At `call` sites the caller pops the returned injected values back into its own locals, preserving trap state (e.g. `RopDetectTrap` depth counter) across speculative calls and WASM-frontend direct calls without exception-based unwinding.

**Do not** move injected params back to `()` returns — that breaks call-site trap-state preservation for both native speculative calls and the WASM frontend.

---

## 4. `&mut self` on hook traits and interior mutability

The `InstructionTrap`, `JumpTrap`, and `TrapConfig` traits currently use `&mut self` receivers. This is a **known limitation** — it makes emit closures non-`Send` and blocks parallel schedule emission. The direction for new and refactored trap implementations is to use interior mutability instead. See [parallel-api.md](parallel-api.md) for the migration plan and guidance on choosing between `Mutex`, `RwLock`, and atomics.

**Do not** paper over the `&mut self` issue by removing the parallel emission goal. **Do** replace `&mut self` with interior mutability as each trap implementation is touched.

---

## 5. Integration into architecture recompilers

**Code:** `crates/native/speet-x86_64/src/lib.rs`, `crates/native/speet-riscv/src/lib.rs`, etc.  
**Design doc:** [trap-hooks.md](../trap-hooks.md) §8 (integration plan)

The trait infrastructure is complete, but integration into architecture recompilers (`setup_traps`, per-instruction trap firing, jump-site trap firing, `classify_insn`) is still in progress. The integration plan is specified in `trap-hooks.md §8`. This is tracked as a goal in [goals/active.md](../../goals/active.md).

**Do not** remove the `TrapConfig` field or `setup_traps` stubs from architecture recompilers — they are placeholders for the pending integration.
