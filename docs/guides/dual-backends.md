# Dual backends and generic support paths

Speet's shared emission (yecta native recompilers → WASM megabinary) feeds **two downstream backends**:

| Backend | Output | Primary host |
|---------|--------|--------------|
| **A — full sandbox** | Valid WASM module | WAFFLE / portal-pc-waffle |
| **B — WASM-as-IR** | Native `.o` via wasm-blitz | Thin runtime link (`speet-recompile` drive) |

[`speet-wasm`](../managed/speet-wasm/) is **not** Backend B. It is a **WASM-runtime OS implementation** over Backend A for arch-recompiler tests and vkernel work.

## Shared megabinary contract

Both backends must receive the same artifact from `finish_module`:

- [`GuestImageLayout`](../../crates/os/speet-link-core/src/image_layout.rs) + patched data segments
- [`FuncSignature`](../func-signature.md) with layout + trap injected params as returns
- Redirect shim functions: `[translated…][shims…][halt][data_init]`
- [`EntityIndexSpace`](../../crates/os/speet-link-core/src/layout.rs) import indices — never hand-counted

## Generic support paths

This foundation unblocks uniform BridgeSupport-driven handling:

1. **HostOffset memory** — `guest_va + host_mem_base` via [`HostOffsetMapper`](../../crates/helper/speet-memory/src/mapper.rs) and [`ParamSlotMap`](../../crates/os/speet-link-core/src/layout_params.rs)
2. **WASM-runtime OS** — in-binary externals via [`FuncSchedule`](../../crates/os/speet-schedule/src/lib.rs) + [`IndexOffsets`](../../crates/managed/speet-wasm/src/lib.rs)

## WASM-runtime OS standard flow

```rust
let external_slot = schedule.push(n_externals, |rctx, ctx| { /* host stubs */ });
let guest_slot    = schedule.push(n_guest,     |rctx, ctx| { /* WasmFrontend */ });

let offsets = IndexOffsets::wasm_os_external_hosting(&schedule, external_slot);
WasmFrontend::new(..., offsets, ...);
```

## Parity checklist (three lanes)

Shared input: **merged-with-mock** canonical multi-memory module (post-shimming pass).

| Lane | Runner | Validates |
|------|--------|-----------|
| **A** | wasmi/wasmtime | Reference sandbox execution |
| **B-wasm** | wasm-blitz as WASM compiler (not wasmi) | Same module as A; compiler path |
| **B-native** | wasm-blitz native + blitz extensions | Import bridge + `__wasm_mem_N` |

When changing emission or link:

- [ ] Does the megabinary validate under wasmparser?
- [ ] Does the shimming pass produce canonical multi-memory when host-mem imports are present?
- [ ] Does wasm-blitz compile the same module (`speet-recompile` drive)?
- [ ] Are redirect shims in the elem segment before halt?
- [ ] Does `PltCallPlan` use manifest indices only?
- [ ] Are layout params declared via `RuntimeLayoutParams` (`LocalSlot` handles)?
- [ ] Does lane B-wasm match lane A semantics on merged-with-mock?
- [ ] Does lane B-native match lane A with blitz extension overlay?

Note: A-OS (`WasmFrontend` + `FuncSchedule`) is **not** a parity lane — covered by A + B-wasm + unit tests.
