# Agent Guide — Speet Recompiler

This file documents design decisions in the speet codebase that are intentional
but may look wrong or over-engineered at first glance.  **Do not "fix" these
patterns without reading the linked documentation first.**

---

## 1. One WASM function per guest instruction (yecta)

**Code:** `crates/helper/yecta/src/lib.rs`, `Reactor` struct
**Doc:** `docs/recompiler-guide.md` §1

The `Reactor` emits one WASM function for every guest instruction.  This is not
a mistake.  Guest ISAs (x86-64, RISC-V, MIPS, DEX) have arbitrary,
unstructured control-flow graphs — computed gotos, loops entered from the
middle, fall-through between switch arms — that cannot be straightforwardly
mapped to WASM's structured `block`/`loop`/`if` nesting.

The solution is to represent each control-flow edge as a `return_call` tail
call to the next function.  Because WASM tail calls forward parameters without
growing the call stack, the entire translated binary runs at **O(1) stack
depth** no matter how many instruction-function hops it takes.  The guest
register file lives in the WASM *parameters* and is forwarded unchanged on
every `return_call`.

Do not collapse functions, eliminate `return_call` chains, or try to
restructure the CFG into a single function — this would require a general
CFG-to-structured-control-flow conversion and is explicitly avoided.

---

## 2. Trap state in parameters, not locals

**Code:** `crates/helper/speet-traps/src/hardening.rs` (`RopDetectTrap`),
`crates/helper/speet-traps/src/security.rs` (`CfiReturnTrap`)
**Doc:** `docs/trap-hooks.md` §3.1, §9

WASM non-parameter locals reset to zero at every function boundary.  Because
each guest instruction is its own WASM function (see §1 above), state stored
in a non-param local is silently lost when the `return_call` chain advances.

`RopDetectTrap` stores its call/return depth counter in a WASM **parameter**
so it survives across the chain.  `CfiReturnTrap` uses a **local** for its
bitmap-index scratch because that scratch is only needed within the body of a
single function.

The two-phase `declare_params` / `declare_locals` protocol in `TrapConfig`
exists because WASM function types (which include parameters) must be declared
before any function body is emitted, while locals are declared per-function.

Do not move `RopDetectTrap`'s depth counter to a local or a global.  A local
would reset to zero on every `return_call`; a global would require external
coordination.

---

## 3. Lazy store deferral and runtime alias checking

**Code:** `crates/helper/yecta/src/lib.rs` (`LazyStore`, `LocalPool`),
`crates/helper/speet-ordering/src/lib.rs`
**Doc:** `docs/lazy-store-alias-checking.md`

For weak-memory ISAs (RISC-V, MIPS), `MemOrder::Relaxed` defers stores via
`Reactor::feed_lazy` rather than emitting them immediately.  This lets the
reactor sink stores toward control-flow join points and deduplicate stores that
appear in all predecessors.

The hazard is store-to-load forwarding: a deferred store followed by a load
from the same address would give the load a stale value.  The fix is not to
flush all pending stores before every load — that would destroy the
optimisation.  Instead, before emitting each load, the recompiler emits a
runtime alias check: a WASM `if` block that compares the load address against
each pending store's address and flushes only the matching stores.

The `emitted_local` field in `LazyStore` is an i32 runtime flag set to 1
inside the alias-check `if`.  The unconditional barrier flush wraps each store
in `i32.eqz(emitted_local)` to avoid double-storing.

Float stores (`F32Store`, `F64Store`) are always emitted eagerly: float values
cannot be saved in the i32/i64 `LocalPool`, so deferral is not possible.

Do not remove the `emitted_local` flag, remove the alias-check `if` blocks, or
flush all stores before every load.

---

## 4. Free functions for Reactor jumps in `TrapContext`

**Code:** `crates/helper/speet-traps/src/context.rs` (`reactor_jump`,
`reactor_jump_if`)
**Doc:** `docs/trap-hooks.md` §3.5

`TrapContext::jump` and `TrapContext::jump_if` emit `unreachable` as a fallback
for non-Reactor sinks.  To emit a real `return_call` via `Reactor::jmp`, use
the free functions `reactor_jump` and `reactor_jump_if` instead.

These are free functions rather than specialised inherent methods because Rust
stable does not support inherent-method specialisation.  The free-function
pattern with an `F = Reactor<…>` bound is the correct stable-Rust approach.

Do not attempt to make them inherent methods via a blanket impl — this will
not compile on stable Rust.

---

## 5. Two-pass `FuncSchedule` in the linker

**Code:** `crates/os/speet-link/src/linker.rs`
**Doc:** `docs/recompiler-guide.md` §3d

The linker separates function-index declaration from function-body emission in
two phases via `FuncSchedule`:

1. **Registration** (`push`): each binary declares how many functions it will
   produce.  After all `push` calls the full function-index layout is final.
2. **Emit** (`execute`): each binary translates its code with
   `base_func_offset` already set, so cross-binary `call` / `return_call`
   targets can be computed before any body is written.

`execute` panics if an emit closure produces a different function count than
declared; this catches mismatches at the boundary rather than producing a
silently corrupt module.

Do not merge the two passes into one or lazily compute `base_func_offset`
during emission — cross-binary index resolution requires the layout to be final
before any body is emitted.
