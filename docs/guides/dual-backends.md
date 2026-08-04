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

1. **ZeroOffset / OwnedLinear memory** — `guest_va` == WASM linear offset (`host_mem_base = 0`); ZeroOffset additionally leaves unrecompiled `.text` unmapped at `text_base` so BridgeSupport pointer stubs can autogen safely — see [`zero-offset.md`](../future/zero-offset.md)
2. **HostOffset memory** — `guest_va + host_mem_base` via [`HostOffsetMapper`](../../crates/helper/speet-memory/src/mapper.rs) and [`ParamSlotMap`](../../crates/os/speet-link-core/src/layout_params.rs) (high-VA packing)
3. **WASM-runtime OS** — in-binary externals via [`FuncSchedule`](../../crates/os/speet-schedule/src/lib.rs) + [`IndexOffsets`](../../crates/managed/speet-wasm/src/lib.rs)

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
| **B-wasm** | wasm-blitz **source** backends (`blitz-c` / `blitz-js` via `speet_recompile::drive_source`) plus native object compile-check | Same megabinary as A; C/JS/native compiler paths |
| **B-native** | wasm-blitz native `.o` + speet-rt | Import bridge + `__wasm_mem_N`. May emit **ILP32 ELF** (`BinArch::{RiscV32,Arm,X86}` / `blitz-riscv32`/`blitz-arm`/`blitz-i686`) as well as LP64; WASM slots stay 8 bytes, host pointers 4. |

When changing emission or link:

- [ ] Does the megabinary validate under wasmparser?
- [ ] Does the shimming pass produce canonical multi-memory when host-mem imports are present?
- [ ] Does wasm-blitz compile the same module (`speet-recompile` drive)?
- [ ] Are redirect shims in the elem segment before halt?
- [ ] Does `PltCallPlan` use manifest indices only?
- [ ] Are layout params declared via `RuntimeLayoutParams` (`LocalSlot` handles)?
- [ ] Under ZeroOffset, is unrecompiled `.text` omitted from data-init and protected in the host mirror?
- [ ] Does lane B-wasm match lane A semantics on merged-with-mock?
- [ ] Does lane B-native match lane A with blitz extension overlay?
- [ ] Does `flag_spec` (`CallEscape::Flag`) execute under wasmi without soft-skip (no EH proposal)?
- [ ] Does `eh_spec` still use the known wasmi→wasmtime exception gap fallback only when needed?

Note: A-OS (`WasmFrontend` + `FuncSchedule`) is **not** a parity lane — covered by A + B-wasm + unit tests.

## Canonical dual-lane e2e matrix

Generator: [`crates/test/speet-e2e/generate_tests.py`](../../crates/test/speet-e2e/generate_tests.py)  
Filter: [`harness/capabilities.rs`](../../crates/test/speet-e2e/tests/harness/capabilities.rs)  
Configs: `EscapeConfig::{None, Exception, ExceptionSpec, FlagSpec}`  
Paths: wasmi (A), blitz / BlitzC / BlitzJs (B-wasm — native object + C/JS source), thin_native (B-native), linux_wasi (A), darwin_wasi (A).
`native_compile_check` also runs `source_backend_compile_check` (blitz-c + blitz-js).

Capability filter (no invalid cartesian): speculative configs for RV/x86/aarch64; darwin-wasi / linux-wasi / thin_native support Jump + FlagSpec until TagSection is wired for ExceptionSpec. Hand-written WASI/dual_lane files are thin wrappers over `harness/env_*`; regenerate `e2e.rs` after editing the generator.

## Darwin-WASI / Linux-WASI guest rows

| Guest | Lane A | Lane B-native | Notes |
|-------|--------|---------------|-------|
| RV64 Linux (`ecall`) | [`speet-linux-wasi`](../speet-linux-wasi.md) → wasmi | Thin runtime + `MacLibSystemTunnel` / `LinuxLibcTunnel` | Matrix cells + `dual_lane.rs` wrappers; export `"memory"` is host mem (index 1) |
| aarch64 Darwin/BSD (libSystem dylib/GOT; `svc #0x80` optional) | [`speet-darwin-wasi`](../speet-darwin-wasi.md) → wasmi | Thin runtime Mach-O path (when wired) | Lane A GOT/lazy → redirect shims; GOT edge case in `darwin_wasi_tests.rs` |

Parity for a shared corpus slice:

- [ ] Megabinary validates under wasmparser
- [ ] Lane A wasmi exit code / stdout match Lane B-native for the same guest bytes
- [ ] Syscall numbers stay in the mapping crate (`os-linux-wasi` / `os-darwin-wasi`), not hand-matched in the recompiler
- [ ] Preview1 `fd_write` / seed use exported host `"memory"` (mem1), not private guest mem0
