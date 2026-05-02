# FuncSignature: LocalLayout + Returns

This document describes `FuncSignature`, which pairs a `LocalLayout` (params) with an
explicit return type list.  Injected params (trap/call-optimization state) are mirrored as
returns, giving both native recompilers and the WASM frontend a single, consistent function
type protocol.

---

## §1  Motivation

Function types have historically been `(arch_params + injected_params) -> ()`.  Two
problems arise:

1. **Speculative calls** — when a `call` (not `return_call`) crosses a boundary, any
   mutable injected-param state (e.g. the `RopDetectTrap` depth counter) is lost from
   the caller's perspective unless explicitly returned.  Native frontends worked around this
   with `EscapeTag` exception-based unwinding; the WASM frontend threaded
   `injected_params: Vec<ValType>` manually.  Neither expressed the convention in the
   function *type*.

2. **Protocol asymmetry** — native recompilers used the `declare_params → mark →
   declare_locals → alloc_cell` protocol; the WASM frontend bypassed it entirely with a
   private `fn_type_param_layouts` BTreeMap.  Trap integration was therefore impossible
   in the WASM path.

`FuncSignature` unifies both: the function type is `(arch_params + injected) ->
(injected)`, declared once and shared by all frontends.

---

## §2  Type Definition

```rust
pub struct FuncSignature {
    /// Full param layout: arch params first, then injected/trap params.
    pub params: LocalLayout,
    /// Return types = the injected/trap param types, in the same order they
    /// appear at the tail of `params`.  Arch params are not returned.
    pub returns: Vec<ValType>,
    /// Mark at the boundary between arch params and injected params in `params`.
    pub injected_start: Mark,
}

impl FuncSignature {
    /// Number of injected params (= number of return values).
    pub fn injected_count(&self) -> u32;
    /// Derive the canonical WASM FuncType for this signature.
    pub fn to_func_type(&self) -> FuncType;
    /// Seal: place `injected_start`, let traps append via `declare_params`,
    /// then derive `returns` from the appended types.
    pub fn seal(layout: LocalLayout, injected_start: Mark) -> Self;
}
```

`FuncSignature` replaces the bare `LocalLayout` in `LinkerInner` and is the single
source of truth for a recompiler's function type.

---

## §3  Declaration Protocol

```
1. Append arch params to a fresh LocalLayout.
2. Place injected_start mark:  sig.injected_start = layout.mark()
3. Each trap calls declare_params(&mut sig) — appends its injected params.
4. FuncSignature::seal(layout, injected_start) derives returns.
5. sig.to_func_type() is registered in EntityIndexSpace.types (pass 1).
```

This replaces the old sequence `declare_params → mark → declare_locals → alloc_cell`
for the parameter half.  The locals half (`mark → rewind → declare_locals → iter_since`)
is unchanged.

---

## §4  Per-Function Emission

```
for each function:
    layout.rewind(&sig.injected_start)       // truncate to injected_start
    layout.rewind(&locals_mark)              // then rewind further to locals mark
    traps.declare_locals(&mut layout)        // append per-function scratch locals
    cell = cell_registry.alloc(...)          // deduplicate (params, locals) signature
    f = reactor.next_with(ctx, sig.to_func_type(), ...)
```

The function type is constant (derived from `sig`); only the locals change per function.

---

## §5  Call-Site Convention

At every `call f` site (speculative native call or WASM frontend direct call):

```wasm
;; push arch params + injected params
local.get $r0 ... local.get $rN
local.get $depth_counter ...        ;; injected params
call $f                             ;; returns (injected)
;; pop injected params back into locals
local.set $depth_counter ...
```

The multi-value return puts the injected params back in the caller's locals.  No
`EscapeTag` unwinding is needed for the injected-param restoration path; `EscapeTag`
is retained only for speculative-call *return-address mismatch* detection.

The WASM frontend's `injected_params: Vec<ValType>` field is removed; `FuncSignature`
is the authoritative source.

---

## §6  Relation to Other Systems

- **`TrapConfig::declare_params`** now receives `&mut FuncSignature` instead of
  `&mut LocalLayout`.  Traps append to `sig.params` as before; the `injected_start`
  mark delimits their contribution.

- **`EntityIndexSpace`** (see `docs/entity-index-space.md`) receives the pre-registered
  type index from `sig.to_func_type()` during pass 1.

- **`AGENTS.md` §7** contains a short anchor summary and links back to this document.

---

## §7  Concrete Usage Example

### Recompiler setup (once per recompiler instance)

```rust
fn setup_traps(&mut self, rctx: &mut impl BaseContext<Context, E>) {
    // Append arch params (e.g. 32 int regs + PC + expected_ra = 34 params).
    let regs = rctx.layout_mut().append(32, ValType::I64);
    let pc   = rctx.layout_mut().append(1,  ValType::I64);
    let era  = rctx.layout_mut().append(1,  ValType::I64);
    // `declare_trap_params` records injected_start and lets traps append params.
    // (RopDetectTrap would add 1 × I32 here.)
    rctx.declare_trap_params();
    // Place the locals mark — this ALSO seals the FuncSignature in LinkerInner.
    rctx.set_locals_mark(rctx.layout().mark());
}
```

After `set_locals_mark`:
- `linker.inner.signature.params` = `[I64×32, I64×1, I64×1, I32×1]` (34 arch + 1 trap)
- `linker.inner.signature.returns` = `[I32]` (injected trap param mirrored back)
- `linker.inner.signature.injected_count()` = `1`

### Per-function emission (inside init_function)

```rust
fn init_function(&mut self, rctx: &mut impl ReactorContext<Context, E>) {
    rctx.layout_mut().rewind(&rctx.locals_mark());
    // Append arch non-param locals (temps, pool slots).
    let temps = rctx.layout_mut().append(4, ValType::I64);
    rctx.declare_trap_locals();
    let cell = rctx.alloc_cell();
    let f = Function::new(rctx.layout().iter_since(&rctx.locals_mark()).collect());
    self.tail_idx = rctx.next_with(ctx, f, depth)?;
}
```

### At a speculative `call` site

```rust
// Push all params (arch + injected) onto the stack, then call.
// The multi-value return puts injected params back in the caller's locals.
for i in 0..sig.injected_count() {
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(sig.params.local(injected_slot, i)))?;
}
// ... emit arch params ...
rctx.feed(ctx, tail_idx, &Instruction::Call(target_fn_idx))?;
// Pop injected returns back:
for i in (0..sig.injected_count()).rev() {
    rctx.feed(ctx, tail_idx, &Instruction::LocalSet(sig.params.local(injected_slot, i)))?;
}
```
