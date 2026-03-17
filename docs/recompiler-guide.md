# Speet Recompiler Architecture Guide

This document covers three subjects in depth:

1. [The yecta one-function-per-instruction model](#1-the-yecta-one-function-per-instruction-model)
2. [The three target kinds](#2-the-three-target-kinds)
3. [The four principal APIs](#3-the-four-principal-apis): mapper, local allocation, traps, and linker

---

## 1. The yecta one-function-per-instruction model

### Why one function per instruction?

WebAssembly has structured control flow: every branch and loop must sit inside a
`block`/`loop`/`if` nesting that resolves at compile time. Guest ISAs (x86-64,
RISC-V, DEX, …) have arbitrary control-flow graphs — `jmp reg`, computed gotos,
fall-through between switch arms, loops entered from the middle. There is no
general way to turn an arbitrary CFG directly into valid WASM structured control
flow without either inserting a dispatcher or restructuring the CFG itself.

Yecta avoids both: it represents each control-flow edge as a `return_call` (a
tail call to the next WASM function). Because WASM tail calls forward parameters
without growing the call stack, the entire translated binary runs in O(1) stack
depth regardless of how many instruction-function hops it takes. The caller's
local variables serve as the guest register file; they are forwarded on every
`return_call` unchanged, so the register state is persistent across hops.

### Anatomy of a translated instruction

Each guest instruction occupies one or more WASM functions. The outermost
function's parameters are the guest architectural registers (and any trap
parameters added by installed hooks). Conceptually:

```
WASM function for guest instruction at PC=0x1004
  params:  [rax, rbx, …, rsp, rip, zf, sf, cf, of, pf, <trap params>]
  locals:  [scratch0, scratch1, pool_i32_0, …, <trap locals>]
  body:
    ; instruction preamble
    i64.const 0x1004
    local.set rip             ; write PC for traps / speculative-call check

    ; trap fires here (InstructionTrap::on_instruction)

    ; instruction body
    local.get rax
    local.get rbx
    i64.add
    local.set rax

    ; end: tail-call next instruction
    local.get rax
    local.get rbx
    …
    return_call $fn_0x1008    ; jump to the next instruction's function
```

The `return_call` passes the full parameter list to the next function; updated
registers appear in the right parameter slots and the hop is zero-cost in stack
terms.

### Reactor core operations

`Reactor<Context, E, F, P>` manages a growable list of in-progress WASM functions
(`Entry<F>`) and the predecessor graph between them.

| Method | What it does |
|--------|-------------|
| `next_with(ctx, f, len)` | Append a new function `f`; `len` is the control-flow depth: functions in `lens[0..len]` become predecessors |
| `feed(ctx, &Instruction)` | Emit one WASM instruction into the tail function |
| `jmp(ctx, target, params)` | Emit a `return_call target` forwarding `params` parameters; records `target` as a successor so its function can be sealed |
| `ji(ctx, params)` | Emit an indirect `return_call_indirect` via the configured table |
| `seal(ctx, &Instruction)` | Close the current function group with a terminal instruction (`return_call`, `unreachable`, …) and drain it from the in-progress list |
| `drain_fns()` | Collect all completed `F` values and advance `base_func_offset` |
| `set_base_func_offset(n)` | Set the absolute WASM function index of the first function the reactor will emit |

### Function splitting

A single guest instruction may generate more than one WASM function. Two limits
trigger an automatic split:

- **`max_insts_per_fn`** (default 256): when any function reachable from the
  current tail has accumulated this many instructions, the current group is
  sealed before the next `next_with` call.
- **`max_ifs_per_fn`** (default 16): when a conditional branch would push the
  open-`if` depth past this limit in any reachable predecessor, the group is
  sealed first.

On a split, every open `if` block in every reachable function is closed with
`End`, the function is terminated with `Unreachable` (the actual control-flow
exit was already emitted by `jmp` earlier), and predecessor edges are severed so
the next function starts with a clean slate.

The `len` parameter to `next_with` controls how many functions from the *lens
queue* become direct predecessors of the new entry. For sequential instructions
`len=0` makes the previous function the sole predecessor; a conditional branch
with two targets uses `len=1` so both branches lead to the merge function.

### Predecessor graph and transitive saturation

The reactor tracks a `preds: BTreeSet<FuncIdx>` for each entry. When deciding
whether to force a split, it computes the *transitive predecessor set* of the
current tail via BFS — every function that can reach the tail through `preds`
edges. If any member of that set has hit a limit, the whole group is sealed.
The transitive set is cached per entry and lazily invalidated whenever any
`preds` set changes.

### Speculative calls and `EscapeTag`

When a native recompiler knows a call target statically (e.g. x86-64 `CALL
rel32`), it can lower the call to a direct WASM `call` (not `return_call`).
Because WASM's `call` returns, the callee runs in a nested call frame; when it
returns, the generated code checks the guest return address against the
statically-expected value. If they match, execution continues normally. If they
do not (e.g. the callee modified the shadow stack), the code throws an
`EscapeTag` exception carrying the actual return address. The surrounding
`TryTable` catch block receives the exception and dispatches to the correct
continuation.

`EscapeTag` carries a `TagIdx` (the WASM exception tag) and a `TypeIdx` (the
exception payload type, which matches the function parameter signature). The
reactor's `call` and `ret` operations emit the `Block`/`TryTable`/`Throw`
scaffolding automatically when given an `EscapeTag`.

---

## 2. The three target kinds

Targets differ by how much structure their control flow already has. This
determines whether yecta is needed and whether a shared recompiler trait or a
custom frontend is more appropriate.

### 2a. Native targets (x86-64, RISC-V, MIPS, …)

Native ISA binaries have completely unstructured CFG: arbitrary indirect jumps,
computed call targets, inline data, overlapping functions. There is no type or
boundary information embedded in the binary.

**Translation path:**

```
guest bytes
  └─► arch-specific decoder (e.g. iced-x86, riscv-decode)
        └─► arch recompiler (X86Recompiler, RiscVRecompiler, …)
              └─► ReactorContext (Linker)
                    └─► Reactor  →  Vec<F>
```

Each decoded instruction calls:
1. `rc.init_function(ctx, ctx_rc)` — opens a new yecta function via
   `ctx_rc.next_with`, declares locals via the `LocalLayout`.
2. `ctx_rc.on_instruction(info, ctx)` — fires any installed `InstructionTrap`.
3. The instruction body — register reads/writes via `ctx_rc.feed`.
4. At control-flow transfers: `ctx_rc.on_jump(info, ctx)` followed by
   `ctx_rc.jmp(target, params)` or `ctx_rc.ji(params)`.
5. `rc.drain_unit(ctx_rc, entry_points)` — collects functions into a
   `BinaryUnit<F>` and advances `base_func_offset`.

The arch recompiler implements `Recompile<Context, E, F>` and is fully decoupled
from the `Reactor` and `TrapConfig` — it interacts with both only through the
`ReactorContext` trait surface exposed by `Linker`.

**Local layout** for a native arch follows the pattern:

```
0 .. N_REGS-1       integer register file (i64 each)
N_REGS              program counter / RIP (i64)
N_REGS+1 ..         condition flags, link register, etc. (arch-specific)
<trap params>       appended by TrapConfig::declare_params
--- params_mark ---
<arch locals>       scratch temporaries, pool slots (per function)
<trap locals>       appended by TrapConfig::declare_locals
```

### 2b. Managed unstructured targets (DEX / Dalvik)

DEX bytecode is higher-level than native ISA code but still contains arbitrary
gotos (`goto`, `switch`, back-edges) and has no structured-control-flow
guarantee at the bytecode level. Yecta is still needed to flatten the CFG.

**Translation path:**

```
.dex file
  └─► DexReader (parse methods, instructions)
        └─► DexRecompiler
              └─► ReactorContext (Linker)
                    └─► Reactor  →  Vec<F>
```

DEX differs from native in that the unit of compilation is a *method* rather
than a raw byte range. Every DEX instruction at code-unit offset `o` inside a
method whose flat base is `b` becomes WASM function `base_func_offset + b + o`.
All `max_registers` DEX register slots are WASM *parameters* so they persist
across `return_call` chains within the method.

The `DexRecompiler` still uses `TrapConfig` and `LocalLayout` in the same way as
a native arch recompiler.

**Object model integration.** Unlike purely numeric native ISAs, DEX has typed
references (`invoke-virtual`, field access, array operations). An optional
`ObjectModel` trait lets the recompiler ask for the numeric type identifier and
field offsets of named types at recompile time, enabling static dispatch where
the runtime layout is known ahead of time.

### 2c. Managed structured targets (WASM)

A WASM binary already has structured control flow and self-contained functions.
There is nothing to restructure: each input function maps 1-to-1 to one output
WASM function `F`. Yecta adds no value here — its CFG management would only
introduce overhead.

**Translation path:**

```
.wasm file
  └─► wasmparser (parse payloads, sections)
        └─► WasmFrontend
              └─► BaseContext (Linker — layout and offset only)
                    └─► Vec<F>  (accumulated directly, no Reactor)
```

`WasmFrontend` implements `Recompile<Context, E, F>` so it can participate in
the same `FuncSchedule`-based linking workflow, but it accumulates translated
functions in its own `compiled` field and calls only `ctx.advance_base_func_offset`
(not reactor emission methods).

**Address translation.** Each guest linear memory can have an associated
`MapperCallback` that transforms guest virtual addresses to host physical
addresses at the WASM instruction level. Load/store instructions are rewritten
using `MemoryEmitter` to call the mapper before the actual memory operation.

**Data segments.** Guest data sections become WASM *passive* segments (no
embedded load address). A generated `data_init_fn` (type `() → ()`) emits
`memory.init` + `data.drop` for each segment at WASM runtime, using the mapper
to compute physical addresses.

**`WasmFrontend::parse_fn_count`** scans the Function section header in O(1) to
report how many functions the module contains, without performing full translation.
This is the value passed to `FuncSchedule::push` during the registration phase.

---

## 3. The four principal APIs

### 3a. Mapper API (`speet-memory`)

The mapper API translates a guest virtual address (already on the WASM value
stack) into a host physical address by emitting any required WASM instructions.

#### `MapperCallback<Context, E, F>`

```rust
pub trait MapperCallback<Context, E, F: InstructionSink<Context, E>> {
    fn call(&mut self, ctx: &mut Context,
            callback_ctx: &mut CallbackContext<Context, E, F>) -> Result<(), E>;
    fn chunk_size(&self) -> Option<u64> { None }
    fn declare_params(&mut self, _layout: &mut LocalLayout) {}
    fn declare_locals(&mut self, _layout: &mut LocalLayout) {}
}
```

**Stack contract:** the mapper is called with a `i32` or `i64` guest virtual
address on the value stack; it must leave a `i32` or `i64` physical address on
the stack when it returns. Everything in between is up to the mapper.

**`declare_params` / `declare_locals`** follow the same three-phase protocol as
trap hooks (see §3c): the mapper calls `layout.append(...)` and stores the
returned `LocalSlot` handles, which it then resolves via
`layout.local(slot, n)` inside `call`.

**`chunk_size`** signals the preferred data-segment page granularity. When
`Some(n)`, the WASM frontend splits large passive segments at `n`-byte
boundaries so that each chunk maps to exactly one physical page. The built-in
page-table mapper returns `Some(0x10000)` (64 KiB pages).

#### `CallbackContext<'a, Context, E, F>`

A thin wrapper that gives the mapper a mutable reference to the instruction sink
without exposing the concrete sink type. Recompilers construct this on the stack
before invoking the mapper:

```rust
let mut cb = CallbackContext::new(&mut self.reactor);
mapper.call(ctx, &mut cb)?;
```

#### `ChunkedMapper<M>`

Wraps any `MapperCallback` and advertises a fixed `chunk_size`. The four
built-in page-table mapper constructors (`standard_page_table_mapper`, etc.)
return a `ChunkedMapper<impl MapperCallback<…>>` so callers can read the chunk
size without knowing the concrete mapper type.

#### Closure shorthand

Any `FnMut(&mut Context, &mut CallbackContext<…, F>) -> Result<(), E>` closure
implements `MapperCallback` automatically (no `declare_params`/`declare_locals`
needed for simple inline mappers).

---

### 3b. Local allocation API (`wasm-layout`)

Every WASM function needs a list of local variable declarations. `LocalLayout`
provides a named-slot system that lets the arch recompiler, installed traps, and
the mapper all append their own local groups to a single unified layout without
knowing each other's indices.

#### `LocalLayout`

A growable list of `(count, ValType)` groups. Each group gets a `LocalSlot`
handle; calling `layout.local(slot, n)` resolves to the absolute WASM local
index of the `n`-th local in that group.

```rust
let mut layout = LocalLayout::empty();
let regs  = layout.append(16, ValType::I64);  // locals 0-15
let pc    = layout.append(1,  ValType::I64);  // local  16
let flags = layout.append(5,  ValType::I32);  // locals 17-21
// later:
let rax_idx  = layout.local(regs,  0);  // → 0
let rip_idx  = layout.local(pc,    0);  // → 16
let zf_idx   = layout.local(flags, 0);  // → 17
```

#### `Mark` and the params/locals split

The layout serves two roles separated by a `Mark`:

1. **Params region** (`0 .. params_mark.total_locals - 1`): arch register slots
   and trap parameter slots. These are WASM *parameters* — they survive
   `return_call` chains and carry the guest state between translated-instruction
   functions. Declared once per recompiler instance.

2. **Locals region** (`params_mark.total_locals .. `): non-parameter scratch
   temps, pool slots, and trap locals. These are reset to zero at the start of
   each new WASM function. Appended freshly for every `next_with` call by
   rewinding the layout to the params mark first.

```rust
// Phase 1 — setup (once):
layout.append(N_REGS, ValType::I64);
traps.declare_params(&mut layout);
let params_mark = layout.mark();
let total_params = params_mark.total_locals;

// Phase 2 — per function:
layout.rewind(&params_mark);
let scratch = layout.append(2, ValType::I64);
traps.declare_locals(&mut layout);
reactor.next_with(ctx, Function::new(layout.iter_since(&params_mark)), depth)?;
// layout.local(scratch, 0) → total_params + 0
// layout.local(scratch, 1) → total_params + 1
```

#### `LocalSlot`

An opaque index into the layout's slot list. Store it in any struct that needs
to resolve a local index later:

```rust
struct MyTrap { depth_slot: LocalSlot }

fn declare_params(&mut self, layout: &mut LocalLayout) {
    self.depth_slot = layout.append(1, ValType::I32);
}
fn on_instruction(&mut self, info: &InstructionInfo, ctx: &mut Context,
                  trap_ctx: &mut TrapContext<…>) -> Result<TrapAction, E> {
    let depth_idx = trap_ctx.layout().local(self.depth_slot, 0);
    trap_ctx.emit(ctx, &Instruction::LocalGet(depth_idx))?;
    // …
}
```

---

### 3c. Trap API (`speet-traps`)

Traps are pluggable hooks that fire during translation to inject monitoring code,
enforce security policies, or redirect control flow. There are two independent
hook points.

#### `InstructionTrap<Context, E, F>`

Fires **once per translated guest instruction**, after the PC local is written
but before the instruction body is emitted.

```rust
pub trait InstructionTrap<Context, E, F: InstructionSink<Context, E>> {
    fn on_instruction(&mut self, info: &InstructionInfo, ctx: &mut Context,
                      trap_ctx: &mut TrapContext<Context, E, F>) -> Result<TrapAction, E>;
    fn declare_params(&mut self, params: &mut LocalLayout) {}
    fn declare_locals(&mut self, locals: &mut LocalLayout) {}
    fn skip_snippet(&self, info: &InstructionInfo, ctx: &mut Context,
                    skip_ctx: &mut TrapContext<Context, E, F>) -> Result<(), E> {
        skip_ctx.emit(ctx, &Instruction::Unreachable)
    }
}
```

`InstructionInfo` provides:
- `pc: u64` — guest address of the instruction
- `len: u32` — byte length in the guest ISA
- `arch: ArchTag` — which architecture emitted it
- `class: InsnClass` — bitfield: `MEMORY`, `BRANCH`, `CALL`, `RETURN`,
  `PRIVILEGED`, `FLOAT`, `ATOMIC`, `INDIRECT`

Returning `TrapAction::Skip` suppresses the normal instruction body; the reactor
emits `skip_snippet` instead (default: `unreachable`). Override `skip_snippet`
to redirect to a violation handler.

#### `JumpTrap<Context, E, F>`

Fires **immediately before each control-flow transfer** the recompiler is about
to emit.

```rust
pub trait JumpTrap<Context, E, F: InstructionSink<Context, E>> {
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context,
               trap_ctx: &mut TrapContext<Context, E, F>) -> Result<TrapAction, E>;
    fn declare_params(&mut self, params: &mut LocalLayout) {}
    fn declare_locals(&mut self, locals: &mut LocalLayout) {}
    fn skip_snippet(&self, info: &JumpInfo, ctx: &mut Context,
                    skip_ctx: &mut TrapContext<Context, E, F>) -> Result<(), E> { … }
}
```

`JumpInfo` provides:
- `source_pc: u64` — guest PC of the transferring instruction
- `target_pc: Option<u64>` — static target (absent for indirect jumps)
- `target_local: Option<u32>` — WASM local holding the runtime target address
  (indirect jumps only; inspect this for ROP detection)
- `kind: JumpKind` — `DirectJump`, `ConditionalBranch`, `Call`, `Return`,
  `IndirectJump`, `IndirectCall`, `Syscall`

#### `TrapConfig<'cb, 'ctx, Context, E, F>`

The object an arch recompiler embeds. It holds at most one `InstructionTrap`
and one `JumpTrap` (install them with `set_instruction_trap` /
`set_jump_trap`). All methods are no-ops when no trap is installed.

Three-phase lifecycle:

```
Phase 1 — setup (once per recompiler instance)
  traps.declare_params(&mut layout)
  locals_mark = layout.mark()

Phase 2 — per function (in init_function)
  traps.declare_locals(&mut layout)

Phase 3 — firing (during translation)
  traps.on_instruction(info, ctx, &mut reactor, &layout) → TrapAction
  traps.on_jump(info, ctx, &mut reactor, &layout) → TrapAction
```

#### `TrapContext<Context, E, F>`

Passed to the trap's `on_instruction` / `on_jump` / `skip_snippet` methods.
Provides three capabilities:

| Method | What it does |
|--------|-------------|
| `emit(ctx, &Instruction)` | Emit a WASM instruction into the current function |
| `layout() -> &LocalLayout` | Resolve `LocalSlot` handles to absolute indices |
| `jump(ctx, target, params)` | Emit `return_call target` (Reactor sink only) |
| `jump_if(ctx, target, params)` | Emit conditional `return_call target` |

For reactor sinks use the free functions `reactor_jump` and `reactor_jump_if`
which call `Reactor::jmp` directly for full predecessor-graph bookkeeping.

#### Closure shorthand

Any compatible `FnMut` closure implements both `InstructionTrap` and `JumpTrap`
automatically:

```rust
linker.traps.set_instruction_trap(&mut |info, ctx, trap_ctx| {
    // emit a counter increment …
    Ok(TrapAction::Continue)
});
```

#### Composition

- **`ChainedTrap<A, B>`** — zero-cost static pair; `A` fires before `B`.
- **`Vec<Box<dyn InstructionTrap<…>>>`** — dynamic list; each fires in order,
  short-circuiting on the first `Skip`.
- Built-in implementations: `NullTrap`, `CounterTrap` (per-instruction counter),
  `CfiReturnTrap` (call/return depth counter), `RopDetectTrap` (allowlist of
  indirect-jump targets), `TraceLogTrap` (log every transfer to a WASM import).

---

### 3d. Linker API (`speet-link`)

The linker API collects translated binaries from one or more recompilers and
merges them into a single WASM module.

#### `BinaryUnit<F>`

The atom of linking — the output of translating one contiguous binary:

| Field | Contents |
|-------|---------|
| `fns: Vec<F>` | Translated WASM functions in order |
| `base_func_offset: u32` | Absolute WASM function index of `fns[0]` |
| `entry_points: Vec<(String, u32)>` | Named exports: `(symbol, abs_func_idx)` |
| `func_types: Vec<FuncType>` | Per-function type, parallel to `fns` |
| `data_segments: Vec<DataSegment>` | Passive data blobs (no addresses) |
| `data_init_fn: Option<(F, FuncType)>` | Generated `() → ()` data initialiser |

#### `FuncLayout` and `FuncSlot`

Declare function index ranges **before** any translation begins:

```rust
let mut layout = FuncLayout::empty();
let slot_a = layout.append(n_a);  // declares n_a functions
let slot_b = layout.append(n_b);  // immediately follows slot_a

layout.base(slot_a)  // → 0
layout.base(slot_b)  // → n_a
layout.total()       // → n_a + n_b
```

`FuncSlot` handles are stable — `layout.base(slot)` always returns the same
value once `append` has returned.

#### `FuncSchedule<'a, Context, E, F>` — two-pass coordinator

`FuncSchedule` separates declaration from emission:

**Phase 1 — Registration** (`push`): declare each binary's function count and
provide an emit closure. The layout is final after all `push` calls.

**Phase 2 — Emit** (`execute`): iterate in registration order. For each item:
set `base_func_offset` on the linker's reactor, call the emit closure, assert
the produced function count matches the declaration, forward to the plugin.

```rust
let mut schedule = FuncSchedule::new();

// Registration — layout is built incrementally:
let wasm_slot   = schedule.push(WasmFrontend::parse_fn_count(&wasm_bytes)?,
                                |ctx_rc, ctx| { frontend.translate_module(…); frontend.drain_unit(…) });
let native_slot = schedule.push(rc.count_fns(&native_bytes),
                                |ctx_rc, ctx| { rc.reset_for_next_binary(…); translate(…); rc.drain_unit(…) });

// Layout is final — build cross-binary IndexOffsets:
let offsets = IndexOffsets { func: schedule.layout().base(native_slot), .. };

// Emit — base_func_offset set automatically per slot:
schedule.execute(&mut linker, &mut ctx);
```

The `execute` method panics if any emit closure produces a function count that
differs from its declared count, catching mismatches at the boundary rather than
producing a silently corrupt module.

#### `Linker<'cb, 'ctx, Context, E, F, P, Plugin>`

Owns the `Reactor`, `TrapConfig`, `LocalLayout`, and plugin. Implements
`ReactorContext<Context, E>` so recompilers can borrow it as their single
environment parameter. The plugin receives every `BinaryUnit` via
`LinkerPlugin::on_unit`.

Important public fields:

| Field | Purpose |
|-------|---------|
| `reactor` | Direct access for `set_base_func_offset` (used by `FuncSchedule::execute`) |
| `traps` | Install / remove instruction and jump traps |
| `layout` | The unified parameter + locals layout |
| `locals_mark` | Mark after all parameter slots |
| `pool` | `call_indirect` table/type indices |
| `escape_tag` | Optional EscapeTag for speculative calls |
| `plugin` | The downstream `LinkerPlugin` (e.g. `MegabinaryBuilder`) |

#### `MegabinaryBuilder<F>` and `MegabinaryOutput<F>`

The default `LinkerPlugin`. Accumulates `BinaryUnit`s incrementally, deduplicating
`FuncType`s via a `BTreeMap` so the final WASM `TypeSection` is compact.

`finish()` returns a `MegabinaryOutput<F>`:

| Field | WASM section |
|-------|-------------|
| `types: Vec<FuncType>` | TypeSection (deduplicated) |
| `func_type_indices: Vec<u32>` | FunctionSection |
| `fns: Vec<F>` | CodeSection |
| `exports: Vec<(String, u32)>` | ExportSection |
| `passive_data: Vec<Vec<u8>>` | DataSection (passive segments) |
| `data_init_fns: Vec<(F, FuncType)>` | Data-init `() → ()` functions |

#### `Recompile<Context, E, F>` trait

Arch recompilers implement this three-method interface:

```rust
pub trait Recompile<Context, E, F> {
    type BinaryArgs;

    /// Lightweight pre-pass: count the WASM functions this binary will produce.
    /// Called during FuncSchedule registration, before any translation.
    fn count_fns(&self, bytes: &[u8]) -> u32 { /* default panics */ }

    /// Reset per-binary state for a new binary unit (addresses, hints, …).
    fn reset_for_next_binary(
        &mut self,
        ctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        args: Self::BinaryArgs,
    );

    /// Drain accumulated functions into a BinaryUnit and return it.
    fn drain_unit(
        &mut self,
        ctx: &mut dyn ReactorContext<Context, E, FnType = F>,
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F>;
}
```

`WasmFrontend` provides `count_fns` (and the more ergonomic
`parse_fn_count` → `Result<u32, E>` for error-propagating callers). Native arch
recompilers should implement `count_fns` with a lightweight CFG block count.

---

## Putting it all together

The typical end-to-end flow for a two-binary megabinary:

```
  ┌─────────────────────────────── Registration phase ───────────────────────────────┐
  │                                                                                   │
  │  WasmFrontend::parse_fn_count(&wasm_bytes)?  → n_wasm                            │
  │  RiscVRecompiler::count_fns(&native_bytes)   → n_native                          │
  │                                                                                   │
  │  schedule.push(n_wasm,   emit_wasm_closure)  → wasm_slot                         │
  │  schedule.push(n_native, emit_native_closure)→ native_slot                       │
  │                                                                                   │
  │  // layout is final                                                               │
  │  IndexOffsets { func: schedule.layout().base(native_slot), .. }                  │
  │                                                                                   │
  └─────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────── Emit phase ────────────────────────────────────┐
  │                                                                                   │
  │  schedule.execute(&mut linker, &mut ctx)                                         │
  │    ├── [wasm_slot]   linker.reactor.set_base_func_offset(0)                      │
  │    │                 emit_wasm_closure → BinaryUnit { fns: n_wasm items, … }     │
  │    │                 linker.plugin.on_unit(unit)                                  │
  │    └── [native_slot] linker.reactor.set_base_func_offset(n_wasm)                 │
  │                      emit_native_closure → BinaryUnit { fns: n_native items, … } │
  │                      linker.plugin.on_unit(unit)                                  │
  │                                                                                   │
  └─────────────────────────────────────────────────────────────────────────────────┘

  linker.plugin.finish()
    → MegabinaryOutput { types, func_type_indices, fns, exports, … }
    → assemble into final .wasm module
```
