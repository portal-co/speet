# Linker Component Guide

**Crates:** `crates/os/speet-link-core`, `crates/os/speet-linker`, `crates/helper/wasm-layout`, `crates/os/speet-schedule`, `crates/os/speet-module-target`, `crates/os/speet-module-builder`  
**Design docs:** [recompiler-guide.md](../recompiler-guide.md) §3d, [entity-index-space.md](../entity-index-space.md) §1–§4, [func-signature.md](../func-signature.md) §2–§5, [reactor-context-split.md](../reactor-context-split.md) §1–§4

---

## 1. Two-pass `FuncSchedule`

**Code:** `crates/os/speet-link/src/linker.rs`  
**Design doc:** [recompiler-guide.md](../recompiler-guide.md) §3d

The linker separates function-index declaration from function-body emission in two phases via `FuncSchedule`:

1. **Registration** (`push`): each binary declares how many functions it will produce. After all `push` calls the full function-index layout is final.
2. **Emit** (`execute`): each binary translates its code with `base_func_offset` already set, so cross-binary `call` / `return_call` targets can be computed before any body is written.

`execute` panics if an emit closure produces a different function count than declared; this catches mismatches at the boundary rather than producing a silently corrupt module.

**Do not** merge the two passes into one or lazily compute `base_func_offset` during emission — cross-binary index resolution requires the layout to be final before any body is emitted.

---

## 2. Unified entity index pre-declaration (`EntityIndexSpace`)

**Code:** `crates/os/speet-link-core/src/layout.rs` (`IndexSpace`, `EntityIndexSpace`)  
**Design doc:** [entity-index-space.md](../entity-index-space.md) §1–§4

`FuncLayout` has been replaced by `EntityIndexSpace`, which applies the same two-pass discipline to all five WASM entity kinds: types, functions, memories, tables, and tags. Pass 1 (registration) freezes absolute indices for every entity kind before any body is emitted. Pass 2 (emission) reads those indices directly.

`MegabinaryBuilder` no longer self-assigns indices; it consumes them from the frozen `EntityIndexSpace`. `FuncSchedule` carries an `EntityIndexSpace` instead of a bare `FuncLayout`.

**Do not** add entity declarations inside emit closures — that would make indices unknown during cross-binary reference resolution.

---

## 3. `FuncSignature`: injected params mirror as returns

**Code:** `crates/helper/wasm-layout/src/lib.rs` (`FuncSignature`)  
**Design doc:** [func-signature.md](../func-signature.md) §2–§5

`FuncSignature` pairs a `LocalLayout` (params) with a `Vec<ValType>` of return types. The returns are exactly the injected/trap params — those declared after the `injected_start` mark — mirrored back. Arch params are not returned; they travel forward via `return_call`.

This gives every translated function the type `(arch_params + injected) -> (injected)`. At `call` sites the caller pops the returned injected values back into its own locals, preserving trap state across speculative calls and WASM-frontend direct calls without exception-based unwinding.

`LinkerInner` holds a `FuncSignature` instead of a bare `LocalLayout`. `TrapConfig::declare_params` receives `&mut FuncSignature`.

**Do not** move injected params back to `()` returns — that breaks call-site trap-state preservation.

---

## 4. Per-emit-closure `Reactor` creation (base-reactor context split)

**Code:** `crates/os/speet-linker/src/lib.rs` (`LinkerInner`), `crates/os/speet-link-core/src/context.rs` (`ReactorContext`)  
**Design doc:** [reactor-context-split.md](../reactor-context-split.md) §1–§4

`LinkerInner` no longer owns a `Reactor`. Each native-recompiler emit closure creates a `Reactor` on the stack, wraps it in a `ReactorContext` alongside a borrow of `LinkerInner`, and drops it via `drain_unit` at the closure's end. WASM-frontend emit closures use `LinkerInner` directly (as `BaseContext`) without constructing a reactor.

The dichotomy between native and WASM frontends is now value-level (reactor constructed or not) rather than type-level (two different context implementations).

**Do not** add a `Reactor` field back to `LinkerInner` — that reintroduces the hard native-vs-WASM dichotomy and prevents per-recompile reactor lifecycle management.

---

## 5. `speet-schedule` — two-pass multi-binary coordinator

**Code:** `crates/os/speet-schedule/src/lib.rs`

`speet-schedule` is the coordination layer above `FuncSchedule`. Phase 1 registers all binaries and freezes the `EntityIndexSpace`. Phase 2 dispatches emit closures in order with correct `base_func_offset` values. The panic-on-count-mismatch invariant from `FuncSchedule::execute` (§1 above) applies here too.

`FuncSchedule::execute_selected` is the demand-driven counterpart for callers that
need only a subset of registered binary units. It preserves Phase 1's complete,
frozen index layout, but does not invoke unselected emit closures. Keep expensive
source parsing and lowering inside an emit closure so an unselected unit stays
unmaterialized. It is deliberately coarse: selecting a binary can still emit all of
that binary's functions until its frontend supplies a finer per-function sketch.

**Do not** bypass Phase 1 by computing `base_func_offset` inside a Phase 2 closure — the entire point is that all offsets are known before any emission begins.

---

## 6. `speet-module-target` and `speet-module-builder`

**Code:** `crates/os/speet-module-target/src/lib.rs`, `crates/os/speet-module-builder/src/lib.rs`  
**Design doc:** [module-builder.md](../module-builder.md)

`ModuleTarget` is the abstract interface for WASM module assembly (imports, globals, memory, tables, data segments). `MegabinaryBuilder` is the concrete implementation with type deduplication and complete-module assembly.

Type deduplication is important: a megabinary containing many recompiled binaries would otherwise produce a large duplicate type section. `MegabinaryBuilder` interns types and returns canonical indices.

**Do not** remove type deduplication from `MegabinaryBuilder` — the type section would grow proportionally with the number of recompiled binaries.
