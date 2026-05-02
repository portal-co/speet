# Entity Index Space

This document describes the `EntityIndexSpace` pre-declaration system, which replaces
`FuncLayout` with a unified two-pass index resolver for all five WASM entity kinds.

---

## §1  Motivation

`FuncSchedule` used `FuncLayout` to pre-declare function index ranges in pass 1 so that
cross-binary `call` / `return_call` targets could be computed before any function body was
written.  No equivalent mechanism existed for the other entity kinds — memories, types,
tables, and tags.  Their indices were assigned by `MegabinaryBuilder` at assembly time,
making them unknowable during emission.

`EntityIndexSpace` extends the same two-pass discipline to all five WASM entity kinds.

---

## §2  Types

```rust
/// A single slot-based index range (one per entity kind).
pub struct IndexSpace {
    counts: Vec<u32>,
    bases:  Vec<u32>,
    total:  u32,
}

/// Opaque handle to one range within an `IndexSpace`.
pub struct IndexSlot(usize);

/// Unified pre-declaration covering all five WASM entity kinds.
pub struct EntityIndexSpace {
    pub types:     IndexSpace,
    pub functions: IndexSpace,
    pub memories:  IndexSpace,
    pub tables:    IndexSpace,
    pub tags:      IndexSpace,
}
```

`IndexSpace` mirrors the old `FuncLayout` API:

| Method | Description |
|--------|-------------|
| `append(count) -> IndexSlot` | Declare a contiguous range of `count` indices |
| `base(slot) -> u32` | Absolute index of the first element in `slot` |
| `count(slot) -> u32` | Number of indices in `slot` |
| `total() -> u32` | Grand total of declared indices across all slots |

---

## §3  Two-Pass Protocol

### Pass 1 — Registration

Each binary calls `entity_space.functions.append(n)` (and `types.append(m)`, etc.) to
declare how many entities of each kind it will produce.  After all registrations are
complete the `EntityIndexSpace` is frozen; no new slots may be added.

Cross-binary references (function indices, type indices, memory indices …) are resolved
during this pass via `base(slot)`.

### Pass 2 — Emission

Each binary emits its bodies with the absolute indices already known.  The linker passes
the frozen `EntityIndexSpace` into each emit closure so that index expressions can be
computed directly.

`FuncSchedule::execute` enforces that every emit closure produces exactly the declared
function count; a mismatch panics at the boundary rather than producing a silently corrupt
module.

---

## §4  Relation to Other Systems

- **`FuncSchedule`** now carries an `EntityIndexSpace` instead of a bare `FuncLayout`.
  The `push` step populates all five spaces; the `execute` step passes the frozen space
  to each emit closure.

- **`MegabinaryBuilder`** reads absolute indices from `EntityIndexSpace` instead of
  assigning them internally.  The builder's `declare_memory` / `declare_table` /
  `declare_tag` methods become index consumers rather than allocators.

- **`FuncSignature`** (see `docs/func-signature.md`) uses `EntityIndexSpace.types` to
  pre-register the canonical function type index for each recompiler before any function
  body is emitted.

- **`AGENTS.md` §6** contains a short anchor summary and links back to this document.

---

## §5  Concrete Usage Example

```rust
// Phase 1 — Registration: declare all entity counts up front.
let mut schedule: FuncSchedule<(), Infallible, Function> = FuncSchedule::new();

let n_wasm   = wasm_frontend.count_fns(&wasm_bytes);
let n_native = native_rc.count_fns(&native_bytes);

// Declare function slots (updates entity_space.functions).
let wasm_slot   = schedule.push(n_wasm,   |rctx, ctx| { /* … */ });
let native_slot = schedule.push(n_native, |rctx, ctx| { /* … */ });

// Declare memories, tables, tags on the same space.
let guest_mem_slot = schedule.entity_space_mut().memories.append(1);
let aux_table_slot = schedule.entity_space_mut().tables.append(1);

// Cross-binary index resolution is now available — Phase 1 is complete.
let func_base   = schedule.entity_space().functions.base(native_slot);
let memory_base = schedule.entity_space().memories.base(guest_mem_slot);
let index_offsets = IndexOffsets { func: func_base, table: 0 };

// Phase 2 — Emission: closures use captured bases, linker creates a fresh Reactor.
linker.execute_schedule(schedule, &mut ctx);
```

### Migration from `FuncLayout`

| Old API | New API |
|---------|---------|
| `schedule.layout().base(slot)` | `schedule.entity_space().functions.base(slot)` |
| `FuncSlot` | `IndexSlot` (alias `FuncSlot` retained for compat) |
| `FuncLayout` | `IndexSpace` (alias `FuncLayout` retained for compat) |
| No equivalent | `entity_space_mut().memories.append(n)` |
| No equivalent | `entity_space_mut().tables.append(n)` |
| No equivalent | `entity_space_mut().tags.append(n)` |
