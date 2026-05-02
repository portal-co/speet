# Base-Reactor Context Split

This document describes the base-reactor context split, which moves `Reactor` ownership
out of `LinkerInner` and into per-emit-closure scope so that native and WASM recompilers
share a single base-context type without a hard native-vs-WASM dichotomy.

---

## §1  Motivation

`LinkerInner` previously owned the `Reactor`, making it a mandatory part of every
recompile invocation.  The WASM frontend had to deliberately bypass reactor machinery by
using `BaseContext` (which excluded reactor methods) rather than `ReactorContext`.  There
was no path for a WASM frontend to optionally use a reactor, or for a native frontend to
opt out.

After this split:
- `LinkerInner` holds trap config, signature, pool, escape tag, and index spaces — the
  state that persists across the whole translation unit.
- A `Reactor` is created per emit-closure invocation when the recompiler needs one,
  and `ReactorContext` borrows both `LinkerInner` and that transient `Reactor`.
- WASM-frontend emit closures use `LinkerInner` directly (as `BaseContext`) with no
  reactor construction at all.

The dichotomy shifts from a *type-level* distinction (two different context types) to a
*value-level* distinction (whether the emit closure creates a reactor).

---

## §2  Struct Definitions

```rust
/// Base: persistent translation-unit state, no reactor.
pub struct LinkerInner<'cb, 'ctx, Context, E, P = LocalPool> {
    pub traps:        TrapConfig<'cb, 'ctx, Context, E>,
    pub layout:       FuncSignature,        // replaces bare LocalLayout
    pub pool:         Pool<'cb, Context, E>,
    pub escape_tag:   Option<EscapeTag>,
    pub cell_registry: CellRegistry,
    pub entity_space: EntityIndexSpace,
}

/// Reactor context: base + a transiently-created reactor.
pub struct ReactorContext<'a, Context, E, F, P = LocalPool> {
    pub base:    &'a mut LinkerInner<'a, 'a, Context, E, P>,
    pub reactor: &'a mut Reactor<Context, E, F, P>,
}
```

`BaseContext` is now a thin trait that `LinkerInner` implements directly.  `ReactorContext`
adds the `ReactorContext` trait methods (feed, jmp, next_with, seal_fn) by delegating to
its borrowed `reactor`.

---

## §3  Reactor Lifecycle in FuncSchedule

```rust
// Emit closure for a native recompiler:
schedule.push(fn_count, |base, ctx| {
    let mut reactor = Reactor::new(base.entity_space.functions.base(slot), ...);
    let mut rctx = ReactorContext::new(base, &mut reactor);
    recompiler.translate_bytes(&mut rctx, bytes, ctx)?;
    Ok(reactor.drain_unit())
});

// Emit closure for the WASM frontend:
schedule.push(fn_count, |base, ctx| {
    frontend.translate_module(base, ctx, &wasm_bytes)?;
    Ok(frontend.drain_unit(base, entry_points))
});
```

Each native emit closure constructs a `Reactor` at the start and drops it (via
`drain_unit`) at the end.  The `Reactor` never outlives the closure.

---

## §4  Invariants

- `LinkerInner` must not store a `Reactor`.  A compile-time check (no `Reactor` field)
  enforces this.
- `ReactorContext` must not be stored across function calls; it is always a stack-local
  borrow.
- The `Reactor::base_func_offset` is set from `entity_space.functions.base(slot)` at
  construction time — the same value the old linker computed from `FuncLayout`.
- WASM frontends that wish to use a reactor in the future simply construct one in their
  emit closure; no API change is needed.

---

## §5  Relation to Other Systems

- **`FuncSignature`** (see `docs/func-signature.md`) lives in `LinkerInner`, making it
  available to both `BaseContext` and `ReactorContext` paths.

- **`EntityIndexSpace`** (see `docs/entity-index-space.md`) also lives in `LinkerInner`
  and is readable by both paths during emission.

- **`AGENTS.md` §8** contains a short anchor summary and links back to this document.

---

## §6  Concrete Usage Example

### Native recompiler emit closure (creates its own Reactor)

```rust
// Inside execute_schedule the linker creates ONE Reactor per schedule call
// and wraps it in a ReactorHandle that is passed to all emit closures.
linker.execute_schedule(schedule, &mut ctx);
// Internally this does:
//   let mut reactor: Reactor<Context, E, Function, LocalPool> = Reactor::default();
//   let mut handle = ReactorHandle::new(&mut linker.inner, &mut reactor);
//   schedule.execute(&mut handle, &mut linker.plugin, &mut ctx);
```

If you need a per-item reactor (e.g. for test isolation), create `ReactorHandle` directly:

```rust
let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
reactor.set_base_func_offset(base_offset);
let mut handle = ReactorHandle::new(&mut linker.inner, &mut reactor);
recompiler.translate_bytes(&mut handle, bytes, &mut ctx)?;
handle.seal_remaining(&mut ctx)?;
let unit = BinaryUnit {
    fns: reactor.drain_fns(),
    base_func_offset: base_offset,
    ..
};
```

### WASM frontend emit closure (no Reactor created)

```rust
schedule.push(n_fns, |rctx, ctx| {
    // rctx is a ReactorHandle, but the WASM frontend uses only BaseContext methods.
    frontend.translate_module(rctx, ctx, &wasm_bytes)?;
    frontend.drain_unit(rctx, entry_points)
});
```

### Setting up traps (uses BaseContext via ReactorHandle)

```rust
// Setup phase — uses BaseContext methods; no reactor emission happens here.
let mut linker: Linker<'_, '_, Context, E> = Linker::new();
linker.inner.traps.set_instruction_trap(&mut my_trap);
recompiler.setup_traps(&mut linker.inner);  // calls declare_trap_params + set_locals_mark
// FuncSignature is now sealed in linker.inner.signature.
```
