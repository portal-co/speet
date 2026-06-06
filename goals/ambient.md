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

Accept any `S: InstructionSink<Ctx, E>` and query the capability at runtime:

```rust
use wax_core::build::{AmbientSink, InstructionSink};

fn emit_printf_call<Ctx, E, S>(sink: &mut S, ctx: &mut Ctx) -> Result<(), E>
where
    S: InstructionSink<Ctx, E>,
{
    if let Some(ambient) = sink.as_ambient_sink() {
        // Native backend: emit a direct label-based call.
        ambient.call_ambient(ctx, "libc_printf")?;
    } else {
        // Pure-WASM backend: fall back to a WASM import call or return an error.
        todo!("WASM backend ambient fallback");
    }
    Ok(())
}
```

### 3. Push an address as a value (function pointer)

```rust
if let Some(ambient) = sink.as_ambient_sink() {
    ambient.push_ambient_addr(ctx, "libc_printf")?;
    // Stack now has the address of printf; combine with call_indirect or store.
}
```

### 4. Tail-jump to an ambient symbol

```rust
if let Some(ambient) = sink.as_ambient_sink() {
    ambient.jump_ambient(ctx, "libc_exit")?;
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
