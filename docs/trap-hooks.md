# speet-traps Design Document

**Crate:** `crates/arch/speet-traps`  
**Status:** Traits + standard implementations complete; integration into architecture recompilers is a separate task (see §Integration Plan below).

---

## 1. Problem statement

Speet recompilers translate guest ISA instructions into wasm functions one instruction per function (yecta's model).  There are two recurring needs that must not be hard-coded into any single recompiler:

1. **Per-instruction hooks** — fire a user-supplied callback before every translated instruction.  Use cases: context-switch preemption checks, static debugger breakpoints, per-instruction profiling counters, instruction filtering.

2. **Per-jump hooks** — fire a callback before every control-flow transfer (branch, call, return, indirect jump).  Use cases: ROP canary detection, CFI shadow-stack validation, control-flow tracing, dynamic CFI policy enforcement.

Both categories must be:

- **Architecture-neutral** — the same trait and the same standard implementations work with RISC-V, MIPS, x86-64, and future architectures.
- **Composable** — multiple independent hooks can be stacked without structural changes to the recompiler.
- **Able to emit wasm code** — hooks may need to increment a global, read/write per-function locals, or redirect control flow.
- **Able to add per-function locals** — hooks that carry per-function state (e.g. a CFI scratch index) must be able to declare additional wasm locals beyond those the architecture defines.
- **Able to add cross-function parameters** — hooks that carry state across `return_call` chains (e.g. a ROP depth counter) must be able to contribute wasm **parameters**, not just locals.
- **Able to perform conditional and unconditional jumps** — a hook redirecting to a violation handler needs a real `Reactor::jmp`, not just an `unreachable`.
- **`no_std`** — the crate must compile without the standard library.

---

## 2. Crate structure

```
crates/arch/speet-traps/
  Cargo.toml
  src/
    lib.rs        — crate root, re-exports
    layout.rs     — FunctionLayout, ExtraParams (cross-function parameter protocol)
    locals.rs     — ExtraLocals (per-function non-param locals)
    context.rs    — TrapContext + reactor_jump / reactor_jump_if
    insn.rs       — InstructionTrap, InsnClass, InstructionInfo, TrapAction, ArchTag
    jump.rs       — JumpTrap, JumpKind, JumpInfo
    config.rs     — TrapConfig (two-phase protocol implementation)
    impls.rs      — NullTrap, ChainedTrap, CounterTrap, CfiReturnTrap,
                    RopDetectTrap, TraceLogTrap
```

Dependencies (all workspace):

```toml
[dependencies]
wasm-encoder.workspace = true
wax-core.workspace = true
yecta = { path = "../yecta" }
```

No dependency on `speet-ordering`, `speet-memory`, or any architecture crate.

---

## 3. Core types

### 3.1 Parameter vs. local distinction

Yecta represents each guest instruction as a distinct wasm function.  These functions chain together via `return_call`, which passes `local.get 0 .. local.get (params-1)` to the next function.  This means:

- **Parameters** (wasm locals 0..params-1): survive across `return_call` chains.  Use for state that must carry over from one translated instruction to the next — e.g. a ROP depth counter.
- **Non-param locals** (wasm locals ≥ params): reset to zero at each new function.  Use for scratch space needed only within a single instruction's translation — e.g. a bitmap index scratch register.

The trap system supports both kinds.

### 3.2 `ExtraParams` (`layout.rs`)

Declares a contiguous block of wasm **parameters** owned by a single trap implementation.

```rust
pub struct ExtraParams {
    groups: Vec<(u32, ValType)>,   // (count, type) pairs
    base:   u32,                   // set by TrapConfig::setup
}
```

Key methods:

| Method | When called | By whom |
|--------|-------------|---------|
| `new(groups)` | Once at trap construction | Trap implementation |
| `none()` | Default | Default impl of `extra_params()` |
| `iter()` | In Phase 1 setup | `TrapConfig::setup` |
| `total_count()` | During layout | `TrapConfig::setup` |
| `param(n)` | During code emission | Trap implementation |
| `set_base(base)` | In Phase 1 setup | `TrapConfig::setup` |

`set_base` is `pub(crate)` — only `TrapConfig` may call it.

### 3.3 `FunctionLayout` (`layout.rs`)

The data carrier for Phase 1 of the protocol.

```rust
pub struct FunctionLayout {
    pub base_params:        u32,
    pub total_params:       u32,
    pub(crate) extra_param_groups: Vec<(u32, ValType)>,
}
```

The recompiler creates a `FunctionLayout::new(base_params)`, passes it to `TrapConfig::setup`, and reads back `layout.total_params`.

### 3.4 `ExtraLocals` (`locals.rs`)

Declares a contiguous block of wasm **non-param locals** owned by a single trap implementation.

```rust
pub struct ExtraLocals {
    groups: Vec<(u32, ValType)>,   // (count, type) pairs
    base:   u32,                   // set by TrapConfig::set_local_base
}
```

Key methods:

| Method | When called | By whom |
|--------|-------------|---------|
| `new(groups)` | Once at trap construction | Trap implementation |
| `none()` | Default | Default impl of `extra_locals()` |
| `iter()` | In Phase 2 per-function | `TrapConfig::extend_locals` |
| `total_count()` | During layout | `TrapConfig` |
| `local(n)` | During code emission | Trap implementation |
| `set_base(base)` | Per function | `TrapConfig::set_local_base` |

### 3.5 `TrapContext` (`context.rs`)

The sole interface available to a trap when it fires.

```rust
pub struct TrapContext<'a, Context, E, F: InstructionSink<Context, E>> {
    pub sink:   &'a mut F,
    locals:     &'a ExtraLocals,
    params:     &'a ExtraParams,
    _pd:        PhantomData<…>,
}
```

Methods:

- `emit(ctx, instr)` — forward an instruction to the sink.
- `locals()` — read-only access to the trap's `ExtraLocals` (non-param, per-function).
- `extra_params()` — read-only access to the trap's `ExtraParams` (cross-function).
- `jump(ctx, target, params)` — emit `unreachable` (fallback for non-Reactor sinks).
- `jump_if(ctx, target, params)` — emit `if { unreachable } end`.

Free functions for Reactor sinks (require `F = Reactor<…>`):

- `reactor_jump(trap_ctx, ctx, target, params)` — calls `Reactor::jmp`.
- `reactor_jump_if(trap_ctx, ctx, target, params)` — wraps in `if { Reactor::jmp } end`.

**Why free functions instead of inherent methods?**  Rust does not support specialisation of inherent `impl` blocks.  The free-function approach compiles on stable Rust.

### 3.6 `InstructionTrap` (`insn.rs`)

```rust
pub trait InstructionTrap<Context, E, F: InstructionSink<Context, E>> {
    fn on_instruction(
        &mut self,
        info:     &InstructionInfo,
        ctx:      &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E>;

    fn extra_locals(&self)  -> ExtraLocals  { ExtraLocals::none() }
    fn extra_params(&self)  -> ExtraParams  { ExtraParams::none() }

    fn skip_snippet(…) -> Result<(), E> { emit unreachable }
}
```

`InstructionInfo` carries `pc: u64`, `len: u32`, `arch: ArchTag`, `class: InsnClass`.

`InsnClass` is a bitfield: `MEMORY | BRANCH | CALL | RETURN | PRIVILEGED | FLOAT | ATOMIC | INDIRECT`.

### 3.7 `JumpTrap` (`jump.rs`)

```rust
pub trait JumpTrap<Context, E, F: InstructionSink<Context, E>> {
    fn on_jump(
        &mut self,
        info:     &JumpInfo,
        ctx:      &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E>;

    fn extra_locals(&self)  -> ExtraLocals  { ExtraLocals::none() }
    fn extra_params(&self)  -> ExtraParams  { ExtraParams::none() }

    fn skip_snippet(…) -> Result<(), E> { emit unreachable }
}
```

`JumpInfo` carries:

```rust
pub struct JumpInfo {
    pub source_pc:    u64,
    pub target_pc:    Option<u64>,   // None for indirect jumps
    pub target_local: Option<u32>,   // wasm local holding runtime target (indirect only)
    pub kind:         JumpKind,
}
```

`JumpKind` variants: `DirectJump`, `ConditionalBranch`, `Call`, `Return`, `IndirectJump`, `IndirectCall`, `Syscall`.

### 3.8 `TrapAction`

```rust
pub enum TrapAction {
    Continue,  // proceed with normal emission
    Skip,      // suppress body / jump; emit skip_snippet instead
}
```

### 3.9 `TrapConfig` (`config.rs`)

The struct embedded in a recompiler.

```rust
pub struct TrapConfig<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> {
    insn_trap:   Option<&'cb mut (dyn InstructionTrap<…> + 'ctx)>,
    insn_params: ExtraParams,
    insn_locals: ExtraLocals,
    jump_trap:   Option<&'cb mut (dyn JumpTrap<…> + 'ctx)>,
    jump_params: ExtraParams,
    jump_locals: ExtraLocals,
}
```

Key methods:

| Method | Phase | Purpose |
|--------|-------|---------|
| `set_instruction_trap(trap)` | setup | Install insn trap; snapshot `extra_params` + `extra_locals` |
| `set_jump_trap(trap)` | setup | Install jump trap; snapshot `extra_params` + `extra_locals` |
| `clear_instruction_trap()` / `clear_jump_trap()` | setup | Remove installed trap |
| `setup(layout)` | **Phase 1** | Append trap params to layout; set `ExtraParams` bases; return `total_params` |
| `extend_locals(arch_iter)` | **Phase 2a** | Chain trap locals after arch locals; return merged iterator |
| `set_local_base(first_trap_local)` | **Phase 2b** | Assign bases to non-param trap locals |
| `on_instruction(info, ctx, sink)` | firing | Fire insn trap; call `skip_snippet` if `Skip` |
| `on_jump(info, ctx, sink)` | firing | Fire jump trap; call `skip_snippet` if `Skip` |
| `insn_params()` / `jump_params()` | firing | Access trap `ExtraParams` for emitting param `local.get`/`local.set` at jump sites |

---

## 4. The two-phase parameter/local protocol

### Why two phases?

Parameters must be declared in the wasm **function type** (not in the local section), and the function type must be known before any translation begins — before `init_function` is called for the first instruction.  Locals, on the other hand, are declared per-function inside `init_function`.

This requires two separate protocol phases:

- **Phase 1 (setup)**: collect param declarations from all traps → determine `total_params` → register with the module.
- **Phase 2 (per function)**: collect local declarations from all traps → pass to `next_with`.

### Phase 1 — setup

```rust
// In the recompiler's constructor / reconfiguration path:
let mut layout = FunctionLayout::new(self.base_params());
// e.g. base_params = 66 for RISC-V (32 int + 32 float + PC + expected_RA)
let total_params = self.traps.setup(&mut layout);
self.total_params = total_params;
// Register the function type with the wasm module builder using total_params.
// layout.extra_param_iter() yields the trap-contributed (count, ValType) groups
// if the builder needs them separately.
```

After `setup`, `total_params` must be used as the `params` argument to **every** `jmp`, `ji`, and `ji_with_params` call.  This ensures `return_call` forwards all trap parameters (including the ROP depth counter) to the next function.

### Phase 2 — per function (inside `init_function`)

```rust
let arch_locals: [(u32, ValType); N] = […];
let arch_local_count: u32 = arch_locals.iter().map(|(n, _)| n).sum();

// 2a: chain trap locals after arch locals
let arch_iter = arch_locals.iter().copied();
let mut all_locals = self.traps.extend_locals(arch_iter);
self.reactor.next_with(ctx, f(&mut all_locals), depth)?;

// 2b: tell traps where their locals live
// first_trap_local = total_params + arch_local_count
// (params are locals 0..total_params-1; arch non-param locals follow)
self.traps.set_local_base(self.total_params + arch_local_count);
```

### Layout diagram

```
wasm local index:
  0                        base_params-1    : recompiler params (regs, PC, ...)
  base_params              total_params-1   : trap params (e.g. ROP depth counter)
  total_params             total_params     : ─── non-param locals begin ───
  total_params             +arch_locals-1   : arch non-param locals (temps, pool slots, ...)
  total_params+arch_locals +...             : insn-trap non-param locals
  ...                      (end)            : jump-trap non-param locals
```

`set_local_base` receives `total_params + arch_local_count` as `first_trap_local`.

---

## 5. Composition

### `Vec<Box<dyn …>>`

Both `Vec<Box<dyn InstructionTrap<…>>>` and `Vec<Box<dyn JumpTrap<…>>>` implement their respective traits.  Each element's `on_*` is called in order; the first `Skip` short-circuits.

The `Vec` impl's `extra_params()` and `extra_locals()` return empty (the vec itself declares no resources).  When installing a `Vec`-based trap whose elements need locals, pre-merge with the provided helpers:

```rust
let merged_locals  = merge_insn_trap_locals(&my_trap_vec);
config.set_instruction_trap_with_locals(&mut my_trap_vec, merged_locals);
```

For params, the caller must track per-element param bases manually (similar to locals).

### `ChainedTrap<A, B>`

A static, zero-allocation alternative to `Vec`.  `extra_locals()` and `extra_params()` both return the concatenation of A and B.

```rust
let trap = ChainedTrap::new(RopDetectTrap::new(…), CfiReturnTrap::new(…));
config.set_jump_trap(&mut trap);
```

---

## 6. Standard implementations (`impls.rs`)

| Type | Trait | Extra params | Extra locals | Description |
|------|-------|:---:|:---:|-------------|
| `NullTrap` | both | 0 | 0 | No-op; zero overhead when monomorphised |
| `ChainedTrap<A,B>` | both | A+B | A+B | Run A then B; `Skip` short-circuits |
| `CounterTrap` | `InstructionTrap` | 0 | 0 | Increment a wasm global for matching `InsnClass` |
| `CfiReturnTrap` | `JumpTrap` | 0 | 1×i32 | Bitmap-based return-target allowlist from native CFI data |
| `RopDetectTrap` | `JumpTrap` | 1×i32 | 0 | Call/Return depth counter (param); `depth < 0` → violation |
| `TraceLogTrap` | `JumpTrap` | 0 | 0 | Emit a wasm `call` to a logging import before every jump |

`RopDetectTrap` uses a **parameter** for the depth counter so it persists across the `return_call` chain of per-instruction wasm functions.  `CfiReturnTrap` uses a **local** for its bitmap index scratch since it is only needed within a single function body.

---

## 7. The `TrapAction::Skip` + `skip_snippet` contract

When a trap returns `TrapAction::Skip`, `TrapConfig` immediately calls `skip_snippet` before returning `Skip` to the recompiler.  The recompiler must then:

- **Not** emit the normal instruction body (for `on_instruction`).
- **Not** emit the original jump (for `on_jump`).

The recompiler must still leave the wasm function in a valid state.  `skip_snippet` defaults to emitting `unreachable`, which is always a valid terminator.

**Recommended recompiler pattern:**

```rust
// Instruction trap:
if self.traps.on_instruction(&info, ctx, &mut self.reactor)? == TrapAction::Skip {
    return Ok(());  // skip_snippet already emitted by TrapConfig
}
// … normal instruction body …

// Jump trap:
if self.traps.on_jump(&info, ctx, &mut self.reactor)? == TrapAction::Skip {
    return Ok(());  // skip_snippet already emitted
}
self.jump_to_pc(ctx, target_pc, self.total_params)?;
```

---

## 8. Integration plan for architecture recompilers

### 8.1 Cargo.toml

```toml
speet-traps = { path = "../speet-traps" }
```

### 8.2 Recompiler struct

Add to each recompiler:

```rust
traps: TrapConfig<'cb, 'ctx, Context, E, Reactor<Context, E, F>>,
total_params: u32,   // set by setup(); used in every jmp/ji call
```

Add public setters mirroring existing callback setters:

```rust
pub fn set_instruction_trap(&mut self, trap: &'cb mut (dyn InstructionTrap<…> + 'ctx));
pub fn clear_instruction_trap(&mut self);
pub fn set_jump_trap(&mut self, trap: &'cb mut (dyn JumpTrap<…> + 'ctx));
pub fn clear_jump_trap(&mut self);
```

### 8.3 Phase 1 — setup

Add a `setup_traps` method (or call from `new`/`reconfigure`):

```rust
pub fn setup_traps(&mut self) {
    let base = self.base_params(); // 66 for RV, 35 for MIPS, 22 for x86-64
    let mut layout = FunctionLayout::new(base);
    self.total_params = self.traps.setup(&mut layout);
    // If the module builder needs to register the updated function type,
    // iterate layout.extra_param_iter() here and extend the type.
}
```

Calling `setup_traps` after installing a trap and before the first `translate_instruction` ensures `total_params` is correct.

### 8.4 Phase 2 — `init_function`

```rust
fn init_function(…) {
    let arch_locals = […];  // existing (count, ValType) array
    let arch_local_count: u32 = arch_locals.iter().map(|(n, _)| n).sum();

    let arch_iter = arch_locals.iter().copied();
    let mut all_locals = self.traps.extend_locals(arch_iter);
    self.reactor.next_with(ctx, f(&mut all_locals), depth)?;

    self.traps.set_local_base(self.total_params + arch_local_count);
    Ok(())
}
```

#### RISC-V specifics

`base_params = 66` (32 int + 32 float + PC + expected_RA).  `arch_local_count` = sum of the 8-group locals array (varies by `num_temps` and memory model; compute from the array, don't hardcode).

Replace the existing call:
```rust
// Before:
self.reactor.next_with(ctx, f(&mut locals.into_iter()), 2)

// After:
let arch_local_count: u32 = locals.iter().map(|(n, _)| n).sum();
let arch_iter = locals.iter().copied();
let mut all_locals = self.traps.extend_locals(arch_iter);
self.reactor.next_with(ctx, f(&mut all_locals), 2)?;
self.traps.set_local_base(self.total_params + arch_local_count);
Ok(())
```

Also replace the hardcoded `66` in `jump_to_pc` and all `ji_with_params` calls with `self.total_params`.

#### MIPS specifics

`base_params = 35` (32 GPRs + HI + LO + PC).  Same pattern; replace hardcoded `35` in `jump_to_pc` with `self.total_params`.

#### x86-64 specifics

`base_params = 22` (16 GPRs + flags + RIP + …; check exact count from crate).  Same pattern.

### 8.5 `translate_instruction` changes

After `init_function` and PC write:

```rust
let info = InstructionInfo {
    pc:    pc as u64,
    len:   inst_len_bytes,
    arch:  ArchTag::RiscV,   // or Mips / X86_64
    class: classify_insn(inst),
};
if self.traps.on_instruction(&info, ctx, &mut self.reactor)? == TrapAction::Skip {
    return Ok(());
}
```

### 8.6 Jump site changes

Before every `jmp` / `ji` / `ji_with_params` call:

```rust
let info = JumpInfo::direct(pc as u64, target_pc, JumpKind::DirectJump);
if self.traps.on_jump(&info, ctx, &mut self.reactor)? == TrapAction::Skip {
    return Ok(());
}
self.jump_to_pc(ctx, target_pc, self.total_params)?;
```

For indirect jumps, `local.tee` the computed target into `load_addr_scratch_local` before calling `on_jump` and record it in `JumpInfo::target_local`.  For `Return`, do the same with the link register value.

Full per-architecture jump kind tables (RISC-V / MIPS / x86-64) are unchanged from the original integration plan.

### 8.7 `classify_insn` per architecture

Unchanged from original §7.6; add to each architecture's module.

### 8.8 Test plan

1. **`NullTrap` round-trip** — `total_params` equals `base_params`; output identical to no-trap baseline.
2. **`CounterTrap`** — instruction count matches expected value.
3. **`TrapAction::Skip`** — branch instructions produce only `unreachable` in output.
4. **`CfiReturnTrap`** — extra `i32` scratch local appears in function local declaration; bitmap load + range check appear before jump.
5. **`RopDetectTrap`** — extra `i32` **parameter** appears at index `base_params` in function signature (not in local section); depth increment/decrement appear at the correct call/return sites; `total_params = base_params + 1` after `setup_traps`.
6. **Cross-function depth** — translate a two-instruction sequence (Call in function N, Return in function N+1); verify the depth param is forwarded from N to N+1 via `return_call`.

---

## 9. Design decisions and rationale

### Parameters for cross-function state, locals for per-function scratch

The defining constraint is yecta's model: one wasm function per guest instruction, chained by `return_call`.  `return_call N` passes `local.get 0 .. local.get (params-1)`.  Any state stored in a non-param local is silently reset to zero when the chain advances to the next function.

`RopDetectTrap`'s depth counter must persist across that chain — if a call happens in function N and the return happens in function N+1, the counter must carry the `depth=1` value forward.  Making it a parameter is the only correct solution within yecta's model; a global would also work but requires coordination with the mapper and is not local to the translated code.

`CfiReturnTrap`'s bitmap index scratch is only needed within the body of the function that performs the return check — it is computed, used for two memory accesses, and discarded.  A local is the right fit.

### Two-phase protocol instead of per-function setup

Parameters must be declared in the wasm function type, which must be registered with the module builder *before* any function body is emitted.  Locals are declared per-function, inside `init_function`.  The two-phase split mirrors this wasm-level constraint exactly.

### `FunctionLayout` as a data carrier

`FunctionLayout` is a plain struct, not a controller.  All logic (base assignment, group accumulation) lives in `TrapConfig::setup`.  This keeps the interface surface of `FunctionLayout` minimal and makes the recompiler's usage pattern clear: construct, pass to setup, read back `total_params`.

### `set_local_base` receives `total_params + arch_local_count`

Non-param wasm locals are indexed from 0 in the binary encoding, but wasm local indices in a function body include parameters at the start.  So local index 0 in a function with `total_params = 67` is still the first parameter.  The first non-param local is at index `total_params`.  Trap non-param locals follow arch non-param locals, so their base is `total_params + arch_local_count`.  Passing this compound value to `set_local_base` avoids requiring the trap system to know `total_params` separately from `arch_local_count`.

### Why one trap slot per kind

Keeping a single slot per kind keeps `TrapConfig` a thin struct with no `Vec` allocation by default.  Composition is delegated via `ChainedTrap` (zero allocation) or `Vec<Box<dyn …>>` (heap-allocated, dynamic).

### Free functions for Reactor jumps

Rust stable has no inherent method specialisation.  Free functions with `F = Reactor<…>` bounds compile correctly and give the caller an explicit, readable path to real jumps.

---

## 10. Known limitations

1. **One trap slot per kind** — use `ChainedTrap` or `Vec<Box<dyn …>>` to compose multiple.

2. **Vec param bases** — when using a `Vec<Box<dyn …>>` where elements have `extra_params`, the caller must manually track per-element param bases (analogous to the existing manual locals tracking).

3. **`skip_snippet` on Vec** — the `Vec` impl's `skip_snippet` emits `unreachable`; it does not delegate to the element that returned `Skip`.  Use `ChainedTrap` for pairs needing custom skip snippets.

4. **No cross-group param initialisation** — trap params are initialised to 0 by the wasm runtime at function entry, but only when the function is entered fresh (i.e. not via `return_call`).  The first function in a translated basic block that has been entered from outside the chain will have `depth = 0`, which is correct.  There is no mechanism for the host to pre-seed trap params to a non-zero value — use a wasm global if that is needed.
