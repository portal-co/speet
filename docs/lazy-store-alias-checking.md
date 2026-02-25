# Lazy Store Alias Checking

**Status:** Implemented  
**Crates affected:** `yecta`, `speet-ordering`, `speet-riscv`, `speet-mips`

---

## Problem

Guest ISAs with weak memory models (RISC-V, MIPS) allow ordinary stores to be
observed out of program order.  The `speet-ordering` crate exploits this by
deferring stores via `Reactor::feed_lazy`, letting the yecta reactor sink them
toward control-flow join points or deduplicate them across predecessors.

The optimisation is unsound if a deferred store is followed by a load that
reads from the same address: the load would see stale memory instead of the
value the preceding store wrote.  This is the classic *store-to-load
forwarding* hazard, and it exists here at the recompiler output level, not at
the hardware level.

The naive fix — flushing all pending stores before every load — is correct but
destroys the reordering benefit.  The goal is to preserve it as much as
possible while remaining correct.

---

## Solution: Runtime Alias Checking

Before emitting a load instruction, the recompiler emits a **runtime alias
check** for each pending lazy store.  If the store address equals the load
address at runtime, the store is executed immediately so the load sees the
fresh value.  The check is a wasm `if` block gated on an integer equality
comparison; if the addresses do not match the `if` body is skipped at zero
cost (no branch misprediction in a wasm JIT — the body is simply not entered).

This means:

- **Non-aliasing loads** incur only a comparison + branch per pending store.
- **Aliasing loads** get the store flushed early, exactly when needed.
- The batch flush at `barrier()` / control-flow boundaries is still required
  to drain any stores that were not triggered by a load alias check.

---

## Key Abstractions

### `LazyStore` (in `yecta`)

Each deferred store is represented by a `LazyStore` struct:

```rust
pub struct LazyStore {
    pub addr_local:    u32,      // holds the effective store address
    pub addr_type:     ValType,  // I32 (default) or I64 (memory64)
    pub val_local:     u32,      // holds the value to store
    pub val_type:      ValType,  // I32 or I64 (never F32/F64)
    pub emitted_local: u32,      // i32 runtime flag: 1 = already emitted
    pub instr:         Instruction<'static>,
}
```

`addr_type` tracks whether the address space is 32-bit or 64-bit, enabling
correct alias comparison (`i32.eq` vs `i64.eq`) and correct pool bucket
selection for recycling locals.

`emitted_local` is a boolean `i32` local initialised to 0 at deferral time.
It is set to 1 at runtime by the `local.tee` in the alias-check `if`
condition.  The unconditional flush at `barrier()` checks this flag and skips
stores that were already emitted.  This prevents double-stores when a load
alias fires.

### `LocalPool<N>` (in `yecta`)

A stack-allocated, `no_std`, fixed-capacity pool of recyclable wasm local
indices.  It has two typed free-lists: one for `i32` locals and one for `i64`
locals.  The const capacity `N` bounds each bucket independently.

Each deferred store consumes **three** pool locals:

| Local          | Type       | Purpose                        |
|----------------|------------|--------------------------------|
| `addr_local`   | `addr_type`| Store address (I32 or I64)     |
| `val_local`    | `val_type` | Store value  (I32 or I64)      |
| `emitted_local`| `I32`      | Already-emitted flag           |

If any of the three allocations fails (pool exhausted), all existing pending
stores are flushed unconditionally, their locals freed, and the new store is
emitted eagerly.  Correctness is preserved; only the optimisation is
temporarily disabled.

### `Reactor::feed_lazy` (updated signature)

```rust
pub fn feed_lazy(
    &mut self,
    ctx: &mut Context,
    addr_type: ValType,   // NEW: address width
    val_type:  ValType,   // value width
    instruction: &Instruction<'static>,
) -> Result<(), E>
```

At call time `[addr, value]` must be on the wasm value stack.  `feed_lazy`:

1. Allocates `addr_local`, `val_local`, `emitted_local` from the pool.
2. Emits `LocalSet(val_local)` then `LocalSet(addr_local)` (consuming the
   stack), then `I32Const(0)` + `LocalSet(emitted_local)`.
3. Pushes a `LazyStore` onto each reachable predecessor's `bundles` vec.

If any allocation fails, it falls back to eager emission.

Float stores (`F32Store`, `F64Store`) are **never** deferred because `f32`/
`f64` values cannot be saved in the i32/i64 local pool.  `store_val_type` in
`speet-ordering` returns `None` for float stores; `emit_store` emits them
eagerly regardless of `MemOrder`.

### `Reactor::flush_bundles_for_load` (new method)

```rust
pub fn flush_bundles_for_load(
    &mut self,
    ctx: &mut Context,
    load_addr_local: u32,
    load_addr_type:  ValType,
) -> Result<(), E>
```

Called by `emit_load` and `emit_lr` in `speet-ordering` before the actual
load instruction.  For every pending `LazyStore` in every reachable
predecessor function:

1. Skips stores whose `addr_type != load_addr_type` (different address spaces
   cannot alias).
2. Emits:

```wasm
local.get  store.addr_local
local.get  load_addr_local
i32.eq                          ;; or i64.eq for memory64
local.tee  store.emitted_local  ;; record alias result; keep value on stack
if                              ;; if alias: emit the store now
  local.get  store.addr_local
  local.get  store.val_local
  [store instr]
end
```

The stores are **not** drained from `bundles` here.  They remain pending and
will be drained (and their locals freed) by the next `flush_bundles` call at a
`barrier()` or control-flow boundary.  The `emitted_local` flag prevents
double-stores in `flush_bundles`.

### `Reactor::flush_bundles` (unconditional flush)

Emits each pending store wrapped in a flag check:

```wasm
local.get  store.emitted_local
i32.eqz
if
  local.get  store.addr_local
  local.get  store.val_local
  [store instr]
end
```

Then frees all three locals back to the pool.

---

## `speet-ordering` API Changes

### `emit_store` — added `addr_type` parameter

```rust
pub fn emit_store<Context, E, F: InstructionSink<Context, E>>(
    ctx:       &mut Context,
    reactor:   &mut Reactor<Context, E, F>,
    order:     MemOrder,
    atomic:    AtomicOpts,
    addr_type: ValType,           // NEW
    instr:     Instruction<'static>,
) -> Result<(), E>
```

`addr_type` is passed through to `Reactor::feed_lazy` so the `LazyStore` can
record the correct address width.  Callers obtain it from the recompiler's
`addr_val_type()` helper method.

Float stores (`F32Store`, `F64Store`) are emitted eagerly regardless of
`MemOrder` because they have no pool-compatible value type.

### `emit_load` — added `addr_type` parameter

```rust
pub fn emit_load<Context, E, F: InstructionSink<Context, E>>(
    ctx:       &mut Context,
    reactor:   &mut Reactor<Context, E, F>,
    addr_local: u32,
    addr_type:  ValType,          // NEW
    atomic:     AtomicOpts,
    instr:      Instruction<'static>,
) -> Result<(), E>
```

Calls `reactor.flush_bundles_for_load(ctx, addr_local, addr_type)` before
emitting the load.

### `emit_lr` — added `addr_type` parameter

Same pattern as `emit_load`.

### Internal `emit_rmw` load sites

The three load sites inside `emit_rmw` (non-atomic direct ops, non-atomic
min/max, atomic min/max cmpxchg loop) call
`reactor.flush_bundles_for_load(ctx, addr_local, ValType::I32)`.  AMO
addresses are always 32-bit (`i32`); there is no memory64 path for atomics.

---

## Recompiler Integration

### `addr_val_type()` helper

Each recompiler exposes:

```rust
fn addr_val_type(&self) -> ValType {
    // RISC-V:
    if self.use_memory64 { ValType::I64 } else { ValType::I32 }
    // MIPS (always i32 — no memory64 mode):
    ValType::I32
}
```

This is called once at the start of each `translate_load` / `translate_store`
/ `translate_fload` / `translate_fstore` function body and stored in a local
variable `addr_type`.  This is necessary because `self.addr_val_type()` is an
immutable borrow of `self`, and it must not be live at the same time as the
`&mut self.reactor` mutable borrow required by `emit_load`/`emit_store`.

### `load_addr_scratch_local`

A dedicated wasm local is declared in every generated function to hold the
effective load address for alias comparisons:

| Recompiler | Local index | Type                          |
|------------|-------------|-------------------------------|
| RISC-V     | 74          | `addr_type` (i32 or i64)      |
| MIPS       | 43          | `i32` (always)                |

The type is determined at `init_function` time and matches `addr_val_type()`.

In every load translation:

```rust
let load_addr = Self::load_addr_scratch_local();
let addr_type = self.addr_val_type();
self.reactor.feed(ctx, &Instruction::LocalTee(load_addr))?;
emit_load(ctx, &mut self.reactor, load_addr, addr_type, …)?;
```

The `LocalTee` keeps the address on the wasm stack (consumed by the load
instruction) while simultaneously saving it into `load_addr_scratch_local` for
the alias comparison.

### Local pool sizing

`init_function` declares the pool locals after the load-addr scratch:

**RISC-V:**

| Group              | Count         | Type        | Purpose                    |
|--------------------|---------------|-------------|----------------------------|
| Load-addr scratch  | 1             | `addr_type` | Alias check input          |
| Pool addr slots    | `N_POOL_ADDR` (4) | `addr_type` | `LazyStore.addr_local` |
| Pool i64 slots     | `N_POOL_I64`  (4) | `I64`   | `LazyStore.val_local`  |

The `i32` flag locals (`emitted_local`) are allocated from the addr pool when
`addr_type = I32` (the common 32-bit case) or from a separate flag allocation.
In memory64 mode (`addr_type = I64`) the addr pool holds i64 locals; flag
locals come from the i64 bucket too (since there are no dedicated i32 slots
beyond the fixed layout).

> **Note for memory64:** In memory64 mode (`use_memory64 = true`, `enable_rv64
> = true`) all pool addr slots are i64.  There are no dedicated i32 flag-local
> slots — `emitted_local` must also come from the i64 bucket.  This reduces
> the effective capacity for flag locals.  A future improvement could declare a
> small separate i32 flag-local group regardless of memory model.

**MIPS:**

| Group              | Count           | Type  | Purpose                  |
|--------------------|-----------------|-------|--------------------------|
| Load-addr scratch  | 1               | `I32` | Alias check input        |
| Pool i32 slots     | `N_POOL_I32` (8)| `I32` | addr + val for stores    |
| Pool i64 slots     | `N_POOL_I64` (4)| `I64` | val for 64-bit stores    |

MIPS always uses 32-bit addresses.

---

## Float Stores and Loads

Float stores (`F32Store`, `F64Store`) are **never** deferred via `feed_lazy`.
`store_val_type` returns `None` for them; `emit_store` emits them eagerly.

Float loads (`F32Load`, `F64Load`) go through `emit_load` normally and
participate in the alias check.  The alias check correctly guards any pending
integer stores that might write to the same address that a float load reads
from (or vice versa).  The loaded *value* is a float, but the *address*
comparison is integer — this is always correct.

---

## Memory64 Support

When `use_memory64 = true` on the RISC-V recompiler:

- `addr_val_type()` returns `ValType::I64`.
- `load_addr_scratch_local` is declared as `I64` in `init_function`.
- `LocalTee` saves an i64 address — correct for the wasm memory64 proposal.
- `emit_load`/`emit_store` pass `ValType::I64` as `addr_type`.
- `flush_bundles_for_load` emits `I64Eq` for the alias comparison.
- Pool addr slots are seeded as i64 via `seed_i64`.
- Stores with `addr_type = I32` (from a non-memory64 predecessor) are skipped
  in `flush_bundles_for_load` — different address spaces cannot alias.

MIPS has no memory64 mode; all addresses are always i32.

---

## Correctness Argument

**No double-stores:** The `emitted_local` flag is set to 1 at runtime inside
the alias-check `if` by the `local.tee`.  The `flush_bundles` path wraps each
store in `i32.eqz(emitted_local)` — so a store that was already emitted before
a load is not emitted again.

**No missed stores:** A store whose alias check did not fire (addresses did not
match at runtime) remains in `bundles` and will be emitted by the next
`flush_bundles` call (at `barrier()` or a control-flow boundary).

**Correct load ordering:** `flush_bundles_for_load` is called before the load
instruction is emitted, so the store is always visible to the load if they
alias.

**Pool exhaustion:** When the pool runs out, all existing pending stores are
flushed unconditionally (freeing their locals), then the new store is emitted
eagerly.  This is correct: all pending stores are committed before the current
store, and the current store is committed before any subsequent instruction.

---

## Limitations and Future Work

1. **Flag locals in memory64 mode share the i64 pool.** In memory64 mode there
   are no dedicated i32 locals for `emitted_local`; the i64 bucket must cover
   both addresses and flags.  A dedicated i32 flag-local group (e.g. 4 slots)
   declared unconditionally would fix this without structural changes.

2. **MIPS memory64.** If MIPS ever gains a 64-bit address mode, `addr_val_type`
   should check a `use_memory64` field and the pool layout updated accordingly,
   following the same pattern as RISC-V.

3. **Alias precision.** The check is exact address equality only — no range
   aliasing (e.g. a byte store to address `A` that overlaps a 4-byte load from
   `A-1`).  In practice this is sufficient for the scalar, naturally-aligned
   memory accesses that make up the vast majority of RISC-V/MIPS code.  Range
   aliasing could be added by comparing address ranges at the cost of more
   complex generated code.

4. **Pool capacity tuning.** Currently `N_POOL_ADDR = 4` and `N_POOL_I64 = 4`
   for RISC-V (allowing up to 4 concurrent deferred stores in the common case).
   Empirical measurement of real workloads may suggest different values.

5. **AMO memory64.** `emit_rmw` always passes `ValType::I32` to
   `flush_bundles_for_load`.  If a future RISC-V memory64 target exposes
   atomic memory operations over 64-bit addresses, this should be updated.
