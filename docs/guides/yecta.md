# Yecta Component Guide

**Crates:** `crates/helper/yecta`, `crates/helper/speet-ordering`, `crates/helper/speet-wasm-helpers`  
**Design docs:** [recompiler-guide.md](../recompiler-guide.md) §1, [lazy-store-alias-checking.md](../lazy-store-alias-checking.md), [SPECULATIVE_CALLS.md](../../crates/helper/yecta/SPECULATIVE_CALLS.md)

---

## 1. One WASM function per possible instruction slot

**Code:** `crates/helper/yecta/src/lib.rs`, `Reactor` struct

### 1a. Slot granularity

The `Reactor` emits one WASM function for every **possible instruction slot**, not every actual instruction. On RISC-V (which has 2-byte-aligned compressed instructions), a slot exists at every 2-byte offset — even inside a 4-byte instruction, and even at addresses that are never the start of a real instruction. The PC→function-index formula is `(pc - base_pc) / 2`.

This is deliberate: when a jump target is computed at runtime (indirect branch, computed goto), the recompiler cannot know at translation time which offsets are valid instruction starts. Allocating a slot for every possible alignment means any valid jump target already has a function index, avoiding the need for a runtime lookup table.

**Do not** assume "one function = one decoded instruction." **Do not** try to reduce the function count by only allocating slots for known instruction starts — that breaks indirect branches.

### 1b. Why `return_call` (not a single function)

Guest ISAs (x86-64, RISC-V, MIPS, DEX) have arbitrary, unstructured control-flow graphs — computed gotos, loops entered from the middle, fall-through between switch arms — that cannot be straightforwardly mapped to WASM's structured `block`/`loop`/`if` nesting.

Each control-flow edge is represented as a `return_call` tail call to the target slot's function. Because WASM tail calls forward parameters without growing the call stack, the entire translated binary runs at **O(1) stack depth** no matter how many hops it takes. The guest register file lives in the WASM *parameters* and is forwarded unchanged on every `return_call`.

**Do not** collapse functions, eliminate `return_call` chains, or try to restructure the CFG into a single function — this would require a general CFG-to-structured-control-flow conversion and is explicitly avoided.

### 1c. Function merging (inline fall-through)

`return_call` to an immediately-sequential slot is expensive as a cross-call. The `Reactor` merges functions that unconditionally fall through to one successor: it inlines the successor's instruction body directly into the predecessor's WASM function body, emitting the successor's instructions immediately after the predecessor's without a `return_call`.

This means **a single WASM function can contain the code for many consecutive guest instructions**. When you read a generated WASM function and see instructions that "don't belong" to the guest instruction you expect, they have been inlined by function merging.

**Do not** assume that the WASM function at index N contains only the code for guest instruction N. **Do not** try to reconstruct the guest instruction stream by reading one WASM function at a time — merged functions span multiple guest instructions.

### 1d. Constant folding across merged functions (and its hazards)

The `Reactor` performs local constant folding inside each WASM function body as it emits instructions. Two mechanisms work together:

- **`const_stack`**: deferred `I32Const`/`I64Const` values on the WASM value stack. Subsequent operations that consume them are folded at emit time.
- **`locals_const` / `locals_virtual`**: when a `local.set N` stores a known constant, the store is *elided* (not emitted) and the value is recorded in `locals_const[N]`. Subsequent `local.get N` are replaced inline by the constant without emitting a real `local.get`. The local is marked `locals_virtual` — meaning its WASM storage cell has not been physically written yet.

**The hazard**: code emitted sequentially inside the same Entry (e.g. multiple arms of a `br_table` dispatch) shares this folding state. If arm A stores an *unknown* value into local N (clearing `locals_const[N]`), arm B's `local.get N` falls back to a real `local.get` and reads the WASM default `0`, not the constant that was virtually stored before the dispatch.

**The fix** (`ConstFoldState::materialize` / `commit_virtual_locals` in `yecta/src/lib.rs`): before emitting *any* instruction that was not constant-folded away, all pending `locals_virtual` stores are flushed — the deferred `const` + `local.set` pair is emitted so that the WASM local physically holds the correct value. This is intentionally conservative (less optimized, but correct under sequential multi-arm emission).

**Do not** remove the `locals_virtual` flush in `ConstFoldState::commit_virtual_locals`. **Do not** assume that a local which was "virtually" set actually contains the right value in its WASM storage cell — it may not until the flush runs.

### 1e. Optimizer seam, the Reactor↔Entry handshake, and exit-point edges

The constant folding above is one *optimizer strategy*. It lives behind a one-variant enum:

```rust
enum Optimizer { ConstFold(ConstFoldState) }
```

attached to each `Entry`. This is deliberate groundwork: the enum is the seam for future optimizer variants (e.g. native condition emission) without disturbing the faithful default.

Emission is split into two halves:

- **The `Reactor` owns the predecessor graph** — it computes the reachable set, records edges, detects cycles, and forces splits. It decides *which* functions receive an instruction.
- **Each `Entry` decides *how* it emits** — `Entry::feed_one` runs the optimizer pipeline (fold / defer / skip / emit) for one function body. Conditional branches go through `Entry::emit_conditional_arm`, the per-entry half of the handshake: the Reactor loops the reachable set and asks each `Entry` to emit its own branch arm. The default arm emits the faithful `if <cond> { <params>; return_call target } else …`.

Predecessor edges carry an `ExitId` (`Entry.preds: BTreeMap<FuncIdx, ExitId>`). Today every edge is `SOLE_EXIT` because the faithful lowering gives each function a single tail exit. The id is the hook a future optimizer variant uses to emit native control flow — where the taken branch and the fall-through are *distinct* exits of one function and must be told apart in the graph. See [perf.md](../../goals/perf.md) for the long-term goal.

**Do not** assume "one function = one exit" is a structural invariant — it is the *default* lowering, and `ExitId` exists precisely so optimizer variants can break it. **Do not** move the predecessor-graph / cycle / split logic onto `Entry`, or the per-entry emission decision onto the `Reactor`: that split is the handshake.

---

## 2. Lazy store deferral and runtime alias checking

**Code:** `crates/helper/yecta/src/lib.rs` (`LazyStore`, `LocalPool`), `crates/helper/speet-ordering/src/lib.rs`  
**Design doc:** [lazy-store-alias-checking.md](../lazy-store-alias-checking.md)

For weak-memory ISAs (RISC-V, MIPS), `MemOrder::Relaxed` defers stores via `Reactor::feed_lazy` rather than emitting them immediately. This lets the reactor sink stores toward control-flow join points and deduplicate stores that appear in all predecessors.

The hazard is store-to-load forwarding: a deferred store followed by a load from the same address would give the load a stale value. The fix is not to flush all pending stores before every load — that would destroy the optimisation. Instead, before emitting each load, the recompiler emits a runtime alias check: a WASM `if` block that compares the load address against each pending store's address and flushes only the matching stores.

The `emitted_local` field in `LazyStore` is an i32 runtime flag set to 1 inside the alias-check `if`. The unconditional barrier flush wraps each store in `i32.eqz(emitted_local)` to avoid double-storing.

Float stores (`F32Store`, `F64Store`) are always emitted eagerly: float values cannot be saved in the i32/i64 `LocalPool`, so deferral is not possible.

**Do not** remove the `emitted_local` flag, remove the alias-check `if` blocks, or flush all stores before every load.

---

## 3. Speculative call lowering

**Code:** `crates/native/speet-riscv/src/direct.rs`, `crates/native/speet-x86_64/src/direct.rs`  
**Design doc:** [SPECULATIVE_CALLS.md](../../crates/helper/yecta/SPECULATIVE_CALLS.md)

When the recompiler detects an ABI-compliant call instruction (RISC-V `jal x1`/`jalr x1`, x86-64 `call`, AArch64 `BL`/`BLR`) and speculative calls are enabled, it lowers to a native WASM `call` rather than a `return_call`. The expected return address is stored in a hidden `expected_ra` local (via fixups) *and* in the guest ABI location (`ra` / stack / `LR`).

Mismatch signaling is selected by `CallEscape` (see design doc):

- `Exception(EscapeTag)` — `TryTable`/`Throw` (requires `TagSection` + EH runtime)
- `Flag` — trailing `i32` result (`0` match / `1` mismatch); no exception opcodes

On match, emit `Return` (Flag: plus `i32.const 0`). On mismatch, Exception throws; Flag returns with `i32.const 1`. The caller's catch / flag `if` restores the register file and falls through — same shape for both modes.

**Do not** remove or short-circuit the `expected_ra` comparison — it distinguishes a legitimate ABI return from a computed jump that lands on a return instruction.

**Do not** require exception handling for speculative calls — Flag mode is a supported escape path and must keep the post-call flag check (empty `if`/`else` then restore). Do not open `HoistedCallRegion` / `TryTable` on the Flag path.

---

## 4. `speet-wasm-helpers` — WASM arithmetic helpers

**Code:** `crates/helper/speet-wasm-helpers/src/lib.rs`

This crate provides inline WASM instruction sequences for operations that WASM lacks natively, specifically 64×64→128-bit multiplication variants (high-bits extraction for both signed and unsigned). The helpers emit sequences of `i64.mul`, shifts, and `extend` instructions documented inline in the source.

These are pure instruction emitters with no architectural decisions. If you need a new arithmetic helper for a guest ISA instruction that WASM can't express directly, add it here.
