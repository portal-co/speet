# Agent Guide — Speet Recompiler

This file documents design decisions in the speet codebase that are intentional
but may look wrong or over-engineered at first glance.  **Do not "fix" these
patterns without reading the linked documentation first.**

---

## 1. One WASM function per possible instruction slot (yecta)

**Code:** `crates/helper/yecta/src/lib.rs`, `Reactor` struct
**Doc:** `docs/recompiler-guide.md` §1

### 1a. Slot granularity

The `Reactor` emits one WASM function for every **possible instruction slot**,
not every actual instruction.  On RISC-V (which has 2-byte-aligned compressed
instructions), a slot exists at every 2-byte offset — even inside a 4-byte
instruction, and even at addresses that are never the start of a real
instruction.  The PC→function-index formula is `(pc - base_pc) / 2`.

This is deliberate: when a jump target is computed at runtime (indirect branch,
computed goto), the recompiler cannot know at translation time which offsets
are valid instruction starts.  Allocating a slot for every possible alignment
means any valid jump target already has a function index, avoiding the need for
a runtime lookup table.

Do not assume "one function = one decoded instruction."  Do not try to reduce
the function count by only allocating slots for known instruction starts — that
breaks indirect branches.

### 1b. Why `return_call` (not a single function)

Guest ISAs (x86-64, RISC-V, MIPS, DEX) have arbitrary, unstructured
control-flow graphs — computed gotos, loops entered from the middle,
fall-through between switch arms — that cannot be straightforwardly mapped to
WASM's structured `block`/`loop`/`if` nesting.

Each control-flow edge is represented as a `return_call` tail call to the
target slot's function.  Because WASM tail calls forward parameters without
growing the call stack, the entire translated binary runs at **O(1) stack
depth** no matter how many hops it takes.  The guest register file lives in
the WASM *parameters* and is forwarded unchanged on every `return_call`.

Do not collapse functions, eliminate `return_call` chains, or try to
restructure the CFG into a single function — this would require a general
CFG-to-structured-control-flow conversion and is explicitly avoided.

### 1c. Function merging (inline fall-through)

`return_call` to an immediately-sequential slot is expensive as a cross-call.
The `Reactor` merges functions that unconditionally fall through to one
successor: it inlines the successor's instruction body directly into the
predecessor's WASM function body, emitting the successor's instructions
immediately after the predecessor's without a `return_call`.

This means **a single WASM function can contain the code for many consecutive
guest instructions**.  When you read a generated WASM function and see
instructions that "don't belong" to the guest instruction you expect, they have
been inlined by function merging.

Do not assume that the WASM function at index N contains only the code for
guest instruction N.  Do not try to reconstruct the guest instruction stream by
reading one WASM function at a time — merged functions span multiple guest
instructions.

### 1d. Constant folding across merged functions (and its hazards)

The `Reactor` performs local constant folding inside each WASM function body as
it emits instructions.  Two mechanisms work together:

- **`const_stack`**: deferred `I32Const`/`I64Const` values on the WASM value
  stack.  Subsequent operations that consume them are folded at emit time.
- **`locals_const` / `locals_virtual`**: when a `local.set N` stores a known
  constant, the store is *elided* (not emitted) and the value is recorded in
  `locals_const[N]`.  Subsequent `local.get N` are replaced inline by the
  constant without emitting a real `local.get`.  The local is marked
  `locals_virtual` — meaning its WASM storage cell has not been physically
  written yet.

**The hazard**: code emitted sequentially inside the same Entry (e.g. multiple
arms of a `br_table` dispatch) shares this folding state.  If arm A stores an
*unknown* value into local N (clearing `locals_const[N]`), arm B's `local.get
N` falls back to a real `local.get` and reads the WASM default `0`, not the
constant that was virtually stored before the dispatch.

**The fix** (`materialize_for` in `yecta/src/lib.rs`): before emitting *any*
instruction that was not constant-folded away, all pending `locals_virtual`
stores are flushed — the deferred `const` + `local.set` pair is emitted so
that the WASM local physically holds the correct value.  This is intentionally
conservative (less optimized, but correct under sequential multi-arm emission).

Do not remove the `locals_virtual` flush in `materialize_for`.  Do not assume
that a local which was "virtually" set actually contains the right value in its
WASM storage cell — it may not until the flush runs.

---

## 2. Trap state in parameters, not locals

**Code:** `crates/helper/speet-traps/src/hardening.rs` (`RopDetectTrap`),
`crates/helper/speet-traps/src/security.rs` (`CfiReturnTrap`)
**Doc:** `docs/trap-hooks.md` §3.1, §9

WASM non-parameter locals reset to zero at every function boundary.  Because
each guest instruction slot is its own WASM function (see §1 above), state
stored in a non-param local is silently lost when the `return_call` chain
advances.

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

## 4. `EmitSink` trait for jump emission in `TrapContext`

**Code:** `crates/helper/yecta/src/lib.rs` (`EmitSink`),
`crates/helper/speet-traps/src/context.rs` (`TrapContext`)
**Doc:** `docs/trap-hooks.md` §3.5

`TrapContext::jump` and `TrapContext::jump_if` call `EmitSink::emit_jmp` on the
underlying (type-erased) sink.  When the sink is a `Reactor`, this delegates to
`Reactor::jmp` with full predecessor-graph bookkeeping.  The trap traits
(`InstructionTrap`, `JumpTrap`, `TrapConfig`) carry no `F` type parameter;
`TrapContext` holds `&mut dyn EmitSink<Context, E>` instead.

Do not add `F` back to the trap traits.  The `dyn EmitSink` indirection is
intentional — it allows traps to be used as `dyn InstructionTrap<Context, E>`
trait objects without caring about the concrete sink type.

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

---

## 6. Unified entity index pre-declaration (`EntityIndexSpace`)

**Code:** `crates/os/speet-link-core/src/layout.rs` (`IndexSpace`, `EntityIndexSpace`)
**Doc:** `docs/entity-index-space.md` §1–§4

`FuncLayout` has been replaced by `EntityIndexSpace`, which applies the same two-pass
discipline to all five WASM entity kinds: types, functions, memories, tables, and tags.
Pass 1 (registration) freezes absolute indices for every entity kind before any body is
emitted.  Pass 2 (emission) reads those indices directly.

`MegabinaryBuilder` no longer self-assigns indices; it consumes them from the frozen
`EntityIndexSpace`.  `FuncSchedule` carries an `EntityIndexSpace` instead of a bare
`FuncLayout`.

Do not add entity declarations inside emit closures — that would make indices unknown
during cross-binary reference resolution.

---

## 7. `FuncSignature`: injected params mirror as returns

**Code:** `crates/helper/wasm-layout/src/lib.rs` (`FuncSignature`)
**Doc:** `docs/func-signature.md` §2–§5

`FuncSignature` pairs a `LocalLayout` (params) with a `Vec<ValType>` of return types.
The returns are exactly the injected/trap params — those declared after the
`injected_start` mark — mirrored back.  Arch params are not returned; they travel forward
via `return_call`.

This gives every translated function the type `(arch_params + injected) -> (injected)`.
At `call` sites the caller pops the returned injected values back into its own locals,
preserving trap state (e.g. `RopDetectTrap` depth counter) across speculative calls and
WASM-frontend direct calls without exception-based unwinding.

`LinkerInner` holds a `FuncSignature` instead of a bare `LocalLayout`.
`TrapConfig::declare_params` receives `&mut FuncSignature`.

Do not move injected params back to `()` returns — that breaks call-site trap-state
preservation for both native speculative calls and the WASM frontend.

---

## 8. Per-emit-closure `Reactor` creation (base-reactor context split)

**Code:** `crates/os/speet-linker/src/lib.rs` (`LinkerInner`),
`crates/os/speet-link-core/src/context.rs` (`ReactorContext`)
**Doc:** `docs/reactor-context-split.md` §1–§4

`LinkerInner` no longer owns a `Reactor`.  Each native-recompiler emit closure creates
a `Reactor` on the stack, wraps it in a `ReactorContext` alongside a borrow of
`LinkerInner`, and drops it via `drain_unit` at the closure's end.  WASM-frontend emit
closures use `LinkerInner` directly (as `BaseContext`) without constructing a reactor.

The dichotomy between native and WASM frontends is now value-level (reactor constructed
or not) rather than type-level (two different context implementations).

Do not add a `Reactor` field back to `LinkerInner` — that reintroduces the hard
native-vs-WASM dichotomy and prevents per-recompile reactor lifecycle management.

---

## 9. `Ctx` parameter passed down to raw targets

**Code:** `crates/native/speet-x86_64/src/lib.rs` (`setup_traps`), `crates/native/speet-riscv/src/lib.rs` (`setup_traps`), `crates/native/speet-mips/src/lib.rs` (`setup_traps`), `crates/managed/speet-dex/src/lib.rs` (`setup_traps`)
**Doc:** `docs/recompiler-guide.md`

All recompiler `setup_traps` and translate methods accept a `ctx: &mut Context` parameter. This is intentional. Even though the default or vanilla targets (like `wasm_encoder::Function`) might pass a dummy `()` or ignore this parameter, it is a strict requirement for other `wax-core` recompile targets. These custom targets use the context to manage parallel instruction emission and coordinate state across multiple threads.

Do not remove the `ctx: &mut Context` parameter from `setup_traps` or any other recompiler API signatures. Ensure `Ctx` is passed all the way down to the raw target.

