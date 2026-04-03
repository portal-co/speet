# Parallel / Multi-Target API Migration Plan

## Background

The three most recent commits refactored `yecta`'s `Reactor` internals so that
the function list (`fns`) and related state are internally locked with a
`spin::Mutex`-guarded `LockCfg`, `UnsafeCell`, and per-entry read/write lock
protocol.  The goal is to allow multiple threads to emit instructions into
**different** entries of the same reactor concurrently (one entry per thread,
with no contention on distinct entries).

The refactor introduced two granular lock helpers:

| Helper | Semantics |
|---|---|
| `lock_global()` | Exclusive access to the whole `Vec<Entry<F>>` — used for structural operations (push, seal-for-split, drain). |
| `lock_entry(idx, ro: bool)` | Per-entry access. `ro=true` increments a reader count; `ro=false` is a single exclusive writer.  Multiple concurrent readers on the same entry are fine; a writer waits for all readers to finish. |

The `local_pool` and `peephole` fields are now `spin::Mutex<P>` and
`spin::Mutex<Option<Instruction<'static>>>` respectively.

However, the **public API surface that consumers call** has not been updated to
match.  As a result the build currently has ~96 errors across the four
downstream crates.  This document describes exactly what needs to change, crate
by crate, and what the end-state API looks like.

---

## Status quo: what broke and why

### 1. `feed` / `seal` no longer exist on `Reactor`

The old single-threaded API had:
```rust
reactor.feed(ctx, &instr)     // → feed_to(tail_idx, …)
reactor.seal(ctx, &instr)     // → seal_to(tail_idx, …)
```
Both implicitly used `self.fns.len() - 1` as the tail index.  That implicit
tail-index lookup is now done through `lock_global()`, which is not free
(it spin-waits for all per-entry locks to drain).  The old wrappers were
removed.  Callers must supply `tail_idx` explicitly, which they can capture
once and reuse for the duration of a single instruction's emission sequence.

### 2. `jmp` arity changed

```rust
// old
reactor.jmp(ctx, target, params)
// new – takes tail_idx first
reactor.jmp(tail_idx, ctx, target, params)
```

### 3. `Pool` acquired lifetime / generic parameters

`Pool` now carries `&'a (dyn IndirectJumpHandler<Context, E>)`, so every
reference to `Pool` as a bare type needs to become `Pool<'_, Context, E>` (or
an explicit lifetime).  The `Linker` and `ReactorAdapter` store a `Pool` field
and expose it through `ReactorContext::pool()`, which must be updated.

### 4. `local_pool` is now behind a `Mutex`

```rust
// old
self.reactor.local_pool.seed_i32(…)
// new
self.reactor.local_pool.lock().seed_i32(…)
```

### 5. `barrier()` was removed (or never landed)

`speet-ordering` calls `reactor.barrier()`.  There is no such method on
the current reactor.  The correct replacement is `reactor.flush_bundles(ctx,
tail_idx)`.

---

## Affected crates

| Crate | Error categories |
|---|---|
| `speet-ordering` | `feed` (×74), `feed_lazy` arity, `flush_bundles_for_load` arity, `barrier` |
| `speet-link` (context + linker) | `feed`, `seal`, `jmp` arity, `Pool` lifetime/generics |
| `speet-x86_64` | inherits from `speet-link` via `ReactorContext`; also calls `reactor.feed` directly |
| `speet-riscv` | `reactor.local_pool.seed_*` (no `.lock()`), inherits reactor calls |
| `speet-mips` | same shape as `speet-riscv` |
| `speet-powerpc` | same shape as `speet-x86_64` |

---

## Proposed API changes

### A. Add `tail_idx`-free convenience wrappers back to `yecta` (minimal churn path)

Rather than touching every call site in every consumer, add thin wrappers to
`Reactor` that recompute the tail index from the global lock once and delegate:

```rust
impl<Context, E, F, P> Reactor<Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// Emit `instr` into the current tail function.
    /// Convenience wrapper around `feed_to(tail_idx, …)`.
    pub fn feed(&self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self.lock_global().len().checked_sub(1)
            .expect("feed called on empty reactor");
        self.feed_to(tail_idx, ctx, instr)
    }

    /// Seal the current tail function.
    /// Convenience wrapper around `seal_to(tail_idx, …)`.
    pub fn seal(&self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self.lock_global().len().checked_sub(1)
            .expect("seal called on empty reactor");
        self.seal_to(tail_idx, ctx, instr)
    }

    /// Unconditional jump from the current tail.
    /// `jmp(ctx, target, params)` — old 3-arg form.
    pub fn jmp_tail(&self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        let tail_idx = self.lock_global().len().checked_sub(1)
            .expect("jmp_tail called on empty reactor");
        self.jmp(tail_idx, ctx, target, params)
    }

    /// Flush all deferred stores for the current tail.
    /// Replaces the removed `barrier()`.
    pub fn barrier(&self, ctx: &mut Context) -> Result<(), E> {
        let tail_idx = self.lock_global().len().checked_sub(1)
            .expect("barrier called on empty reactor");
        self.flush_bundles(ctx, tail_idx)
    }
}
```

These wrappers pay a `lock_global` spin per call but are single-threaded safe
and restore the old call-site ergonomics everywhere.  Once consumers are
migrated to the explicit-`tail_idx` forms the wrappers can be deprecated.

### B. `Pool` lifetime — add type aliases and update `Linker` / `ReactorContext`

The `Pool<'a, Context, E>` lifetime comes from the embedded
`&'a dyn IndirectJumpHandler` reference.  Consumers that store `Pool` as a
field (e.g., `Linker`, `ReactorAdapter`, `X86Recompiler`, `RiscVRecompiler`)
need to either:

1. **Store the handler separately** and construct `Pool<'_, …>` on the fly when
   calling APIs that need it; or
2. **Add lifetime parameters** to the structs themselves (significant churn).

Option 1 is recommended:

```rust
// In Linker, replace:
pub pool: Pool,

// With:
pub pool_table: TableIdx,
pub pool_ty: TypeIdx,
// (and a method that constructs Pool<'_, …> when needed)
```

Or more precisely, since `IndirectJumpHandler` is typically implemented by the
recompiler itself (which already lives adjacent to the `Linker`), a common
pattern will be to pass the handler at each call site rather than storing it in
`Pool`.

For the near-term migration the simplest fix is to add a type alias in
`ReactorContext`:

```rust
// In context.rs
pub type OwnedPool = (TableIdx, TypeIdx);
// and a helper that lifts to Pool<'static, _, _> with a no-op handler
```

This is addressed concretely per-crate below.

### C. `local_pool` access — add lock accessor

Add to `Reactor`:

```rust
pub fn with_local_pool<R>(&self, f: impl FnOnce(&mut P) -> R) -> R {
    f(&mut *self.local_pool.lock())
}
```

Call sites change from:
```rust
self.reactor.local_pool.seed_i32(start, count);
```
to:
```rust
self.reactor.with_local_pool(|p| p.seed_i32(start, count));
```

---

## Per-crate migration steps

### Step 1 — `yecta` (source of truth)

1. Add `feed`, `seal`, `barrier`, `jmp_tail` convenience wrappers (§A above).
2. Add `with_local_pool` accessor (§C above).
3. Keep `feed_to`, `seal_to`, `jmp(tail_idx, …)` as the primary multi-target
   entry points.  Document the single-target wrappers as "sequential
   convenience, not safe to call from multiple threads concurrently".
4. Fix the `Pool` struct: add a `NoopHandler` unit struct that implements
   `IndirectJumpHandler` as a no-op, and provide:
   ```rust
   pub type StaticPool = (TableIdx, TypeIdx);
   impl StaticPool {
       pub fn as_pool<Context, E>(&self) -> Pool<'static, Context, E> {
           static NOOP: NoopHandler = NoopHandler;
           Pool { table: self.0, ty: self.1, handler: &NOOP }
       }
   }
   ```
   This lets structs store `(TableIdx, TypeIdx)` without lifetime parameters
   and materialise `Pool<'_, …>` on the fly.

**Estimated diff size:** ~100 lines added, 0 removed.

---

### Step 2 — `speet-ordering`

All errors are:
- `reactor.feed(…)` → add `tail_idx` parameter to each public function, or
  use the `feed` convenience wrapper.
- `reactor.feed_lazy(ctx, addr_type, val_type, instr, tail_idx)` — `feed_lazy`
  already takes `tail_idx`; callers need to pass it through.
- `reactor.flush_bundles_for_load(ctx, local, ty, tail_idx)` — same.
- `reactor.barrier()` → `reactor.barrier(ctx)` (new wrapper).

**Recommended approach:**  Add `tail_idx: usize` as the **last** parameter to
each public function in `speet-ordering` (`emit_store`, `emit_load`,
`emit_fence`, `emit_rmw`, etc.).  Each function passes it through to the
reactor calls it makes.

```rust
// Before
pub fn emit_store<Context, E, F, P>(
    ctx: &mut Context,
    reactor: &mut Reactor<Context, E, F, P>,
    order: MemOrder,
    instr: &Instruction<'static>,
) -> Result<(), E> { … }

// After
pub fn emit_store<Context, E, F, P>(
    ctx: &mut Context,
    reactor: &mut Reactor<Context, E, F, P>,
    order: MemOrder,
    instr: &Instruction<'static>,
    tail_idx: usize,
) -> Result<(), E> { … }
```

Because `speet-ordering` is only called from `speet-riscv` and `speet-mips`
(and their tests), the ripple is contained.

**Estimated diff size:** ~150 lines changed.

---

### Step 3 — `speet-link` (`context.rs` + `linker.rs`)

**`Pool` lifetime errors:** Replace `pub pool: Pool` with `pub pool_table:
TableIdx, pub pool_ty: TypeIdx` in both `Linker` and `ReactorAdapter`.  Update
`ReactorContext::pool()` to return `(TableIdx, TypeIdx)` (or add a new
`pool_config()` method) and let callers construct `Pool<'_, …>` with
`as_pool()` from Step 1.

**`feed` / `seal` / `jmp` errors:** The `ReactorContext` trait currently
wraps these:

```rust
fn feed(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E>;
fn jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E>;
fn seal_fn(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E>;
```

These are sequential one-at-a-time wrappers suitable for single-threaded
recompilers.  They can be implemented against the `feed` / `seal` / `jmp_tail`
wrappers added in Step 1:

```rust
// In Linker's ReactorContext impl:
fn feed(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
    self.reactor.feed(ctx, insn)   // uses new single-target wrapper
}
fn jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
    self.reactor.jmp_tail(ctx, target, params)
}
fn seal_fn(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
    self.reactor.seal(ctx, insn)
}
```

`FuncSchedule::execute` is sequential (processes one slot at a time) so the
single-target wrappers are correct for it.

**Estimated diff size:** ~60 lines changed.

---

### Step 4 — `speet-x86_64`

All `reactor.feed` calls go through `self.reactor.feed(ctx, &instr)`, which
will work once the `feed` wrapper is added in Step 1.  No signature changes
needed in `x86_64` itself.

`Pool`-field changes in `Linker` (Step 3) need corresponding updates in
`X86Recompiler::drain_unit` which accesses `ctx.pool()`.  If `pool()` now
returns `(TableIdx, TypeIdx)` instead of `Pool`, update the call accordingly.

**Estimated diff size:** ~5 lines changed.

---

### Step 5 — `speet-riscv`

Two categories of change:

1. **`local_pool` access:**
   ```rust
   // Before (3 occurrences in init_function)
   self.reactor.local_pool.seed_i32(start, count)
   self.reactor.local_pool.seed_i64(start, count)
   // After
   self.reactor.with_local_pool(|p| p.seed_i32(start, count))
   self.reactor.with_local_pool(|p| p.seed_i64(start, count))
   ```

2. **`speet-ordering` arity changes** (Step 2 ripple): every call to
   `emit_store`, `emit_load`, `emit_fence`, `emit_rmw` in `speet-riscv/src/direct.rs`
   (~542 call sites with `feed`) needs a `tail_idx` appended.  The tail index
   can be captured once per instruction at the top of `translate_instruction`
   and threaded through.

**Recommended pattern for `translate_instruction` (and other entry points):**
```rust
pub fn translate_instruction(&mut self, ctx: &mut Context, inst: &Inst, pc: u64, …) -> Result<(), E> {
    let tail_idx = self.reactor.fn_count().checked_sub(1)
        .expect("translate called before next_with");
    // … all reactor calls use tail_idx …
}
```

> **Note:** `fn_count()` calls `lock_global().len()` internally, so capturing
> it once up front avoids repeated lock acquisition during translation of a
> single instruction.

**Estimated diff size:** ~600 lines changed (mostly mechanical `tail_idx` threading).

---

### Step 6 — `speet-mips` and `speet-powerpc`

Both follow the same shape as `speet-riscv` and `speet-x86_64` respectively.
The steps are mechanical once Steps 1–5 are done.

**Estimated diff size:** ~200 lines each.

---

## Multi-target parallel API (the end goal)

Once the single-threaded path compiles cleanly via the convenience wrappers,
the parallel path is unlocked.  The pattern for a parallelising driver is:

```rust
use rayon::prelude::*;

// 1. Allocate a Reactor shared across threads (Arc or scoped borrow).
let reactor = Arc::new(Reactor::<Ctx, E, Function>::default());

// 2. Each worker captures a tail_idx from a pre-allocated slot.
//    next_with() is still single-threaded (structural lock), so pre-allocate:
let slots: Vec<usize> = {
    let mut r = Arc::get_mut(&reactor).unwrap();
    instructions.iter().map(|insn| {
        let idx = r.fn_count();
        r.next_with(&mut ctx, Function::new([…]), 0).unwrap();
        idx
    }).collect()
};

// 3. Emit in parallel — each thread holds a distinct tail_idx.
slots.par_iter().zip(instructions.par_iter()).for_each(|(&tail_idx, insn)| {
    // feed_to, flush_bundles_for_load, etc. are safe on distinct tail_idxs
    reactor.feed_to(tail_idx, &mut thread_ctx, &Instruction::Nop).unwrap();
    // …
});

// 4. Drain sequentially.
let fns = Arc::try_unwrap(reactor).unwrap().into_fns();
```

The lock protocol guarantees:
- Two threads calling `feed_to` on **different** `tail_idx` values can proceed
  concurrently (each acquires `lock_entry(idx, false)` independently).
- A thread calling `lock_global()` will block until all `lock_entry` guards
  are dropped, and vice versa.

`FuncSchedule` remains sequential but is only the orchestrator; the emission
closures it calls could internally parallelise per-instruction emission using
the above pattern.

---

## Parallel `FuncSchedule` (stretch goal)

The two-pass invariant (all `push` calls before any `execute`) is still
required because function-index layout must be final before emission.  However,
the **emission phase** of `FuncSchedule::execute` can be parallelised:

```rust
pub fn execute_parallel<…>(self, linker: &mut Linker<…>, ctx: &mut Context) {
    let Self { layout, items } = self;
    // Phase 1: set base offsets (sequential, cheap)
    let bases: Vec<u32> = items.iter().map(|item| layout.base(item.slot)).collect();
    // Phase 2: emit in parallel (each item owns a disjoint tail_idx range)
    items.into_par_iter().zip(bases).for_each(|(item, base)| {
        linker.reactor.set_base_func_offset(base);  // needs per-slot reactor or Arc<Reactor>
        // …
    });
}
```

This requires `Linker` to own an `Arc<Reactor>` or for `execute` to take a
`&Arc<Reactor>`.  Defer this to after the single-threaded migration is green.

---

## Migration order and checkpoints

| Step | Action | Checkpoint |
|------|--------|-----------|
| 1 | Add `feed`, `seal`, `barrier`, `jmp_tail`, `with_local_pool` to `yecta` | `cargo check -p yecta` clean |
| 2 | Add `NoopHandler` + `StaticPool`/`as_pool()` to `yecta` | no new errors in `speet-link` |
| 3 | Update `speet-ordering`: add `tail_idx` param to all public fns | `cargo check -p speet-ordering` clean |
| 4 | Update `speet-link`: fix `Pool` fields, wire wrappers | `cargo check -p speet-link` clean |
| 5 | Update `speet-x86_64` and `speet-powerpc` | tests pass |
| 6 | Update `speet-riscv` and `speet-mips` | full test suite passes |
| 7 | (Optional) Deprecate `feed`/`seal`/`jmp_tail` wrappers; replace call sites with explicit `tail_idx` | all crates use explicit tail_idx |
| 8 | Implement parallel `FuncSchedule::execute_parallel` | parallelism benchmark |

---

## Invariants that must not change

Refer to `AGENTS.md` for full rationale; summarised here:

- **One WASM function per guest instruction** (`AGENTS.md §1`).  The
  `tail_idx` threading in Steps 2–6 must never collapse multiple instructions
  into the same entry.
- **Trap-state parameters, not locals** (`AGENTS.md §2`).  The `tail_idx`
  forwarded to `jmp` / `feed_to` must match the entry that holds the live
  trap parameters.
- **Lazy store deferral and alias checking** (`AGENTS.md §3`).  Every
  `feed_lazy` call must be matched by a corresponding `flush_bundles_for_load`
  before any load from a potentially-aliasing address, and by
  `flush_bundles` / `barrier` at any control-flow boundary.  The `tail_idx`
  passed to `flush_bundles_for_load` must be the same one used for the
  surrounding `feed_lazy` calls.
- **Two-pass `FuncSchedule`** (`AGENTS.md §5`).  The registration / emit
  separation must be preserved even in the parallel variant.
