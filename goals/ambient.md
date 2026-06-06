# Ambient Library Integration

← [goals.md](../goals.md)

---

## Overview

Ambient info allows a speet recompiler to call into **pre-existing, unrecompiled native libraries** without recompiling them. It is a distinct mechanism from WASM imports: instead of going through the `{module}__{name}` linker-stub convention, the generated code emits a **label reference** (`__ambient_{name}`) that the final link/load step resolves to the library's actual address.

### Three API pieces (defined in `wax-core`)

| API | Where | Purpose |
|-----|-------|---------|
| `wax_core::build::AmbientInfo` | trait on `Ctx` | Stores `name → u64` address table for link-time resolution |
| `wax_core::build::AmbientSink` | trait on `Sink` | Emits label-based `push_ambient_addr` / `call_ambient` / `jump_ambient` |
| `InstructionSink::as_ambient_sink()` | default method | Capability query — returns `Some` only on native backends |

---

## Using Ambient Info in a Recompiler

### 1. Declare the ambient symbols needed

Before calling the recompiler, populate `ctx.ambient_addrs` (on `WaxHandle`) with the library symbol → address pairs:

```rust
use portal_solutions_blitz_common::sink::WaxHandle;

ctx.ambient_addrs.insert("libc_printf".into(), libc_symbol_address("printf"));
ctx.ambient_addrs.insert("libc_exit".into(),   libc_symbol_address("exit"));
```

### 2. Write the recompiler to use `as_ambient_sink`

Accept any `S: InstructionSink<Ctx, E>` and query the capability at runtime.
Both `call_ambient` and `jump_ambient` require a `&wasm_encoder::FuncType` that
describes the callee's parameter and return types. The backend marshals arguments
from the WASM value stack into the platform's native argument registers (x86-64
SysV, AAPCS64, or RISC-V psABI) and pushes return values back onto the stack.

```rust
use wax_core::build::{AmbientSink, InstructionSink};
use wasm_encoder::FuncType;

fn emit_printf_call<Ctx, E, S>(
    sink: &mut S, ctx: &mut Ctx,
    printf_sig: &FuncType,   // e.g. (i64, i32) -> i32
) -> Result<(), E>
where
    S: InstructionSink<Ctx, E>,
{
    if let Some(ambient) = sink.as_ambient_sink() {
        // Native backend: pop args into native registers, call, push results.
        ambient.call_ambient(ctx, "libc_printf", printf_sig)?;
    } else {
        // Pure-WASM backend: fall back to a WASM import call or return an error.
        todo!("WASM backend ambient fallback");
    }
    Ok(())
}
```

### 3. Push an address as a value (function pointer)

`push_ambient_addr` does not need a type signature — it simply pushes the
symbol's address as a 64-bit integer onto the WASM value stack:

```rust
if let Some(ambient) = sink.as_ambient_sink() {
    ambient.push_ambient_addr(ctx, "libc_printf")?;
    // Stack now has the address; use with call_indirect or store to memory.
}
```

### 4. Tail-jump to an ambient symbol

```rust
if let Some(ambient) = sink.as_ambient_sink() {
    // Same arg marshalling as call_ambient but emits JMP/BR, no return-value push.
    ambient.jump_ambient(ctx, "libc_exit", &exit_sig)?;
}
```

---

## How the Labels Are Resolved

Each native backend emits `__ambient_{name}` as a **linker symbol**:

- **x86-64**: `lea rax, [rip + __ambient_libc_printf]; call rax` (or push/jmp)
- **AArch64**: `adr x9, __ambient_libc_printf; blr x9`
- **RISC-V 64**: `la t0, __ambient_libc_printf; jalr ra, t0`

The `WaxHandle::ambient_addrs` map holds `name → u64` entries. The concrete assembler/linker must resolve `__ambient_*` labels using those addresses (e.g., as absolute symbols in the ELF symbol table, or via a separate relocation pass).

---

## Constraint

`as_ambient_sink()` returns `None` for pure-WASM backends (`blitz-reencode`, `blitz-js`, `blitz-c`). Recompilers that use ambient calls must handle both cases.

---

## Supported Backends

| Backend | Supported |
|---------|-----------|
| x86-64 (Naive, SysV, LFI) | ✓ |
| AArch64 (Naive, SysV, LFI) | ✓ |
| RISC-V 64 (Naive, SysV) | ✓ |
| Pure WASM (blitz-reencode, blitz-js, blitz-c) | — |
| PowerPC 64 | — (stub, not yet implemented) |

---

## Tasks

- [ ] Add linker/binder support to resolve `__ambient_*` symbols at link time using `WaxHandle::ambient_addrs`
- [ ] Add e2e test: RISC-V recompiler that tail-jumps into a pre-built libc stub
- [ ] Document the `FedContext` and `ReactorMemorySink` tunnelling gap (these wrappers return `None` for `as_ambient_sink` because they hold immutable reactor references; see `speet-link-core/src/context.rs`)
