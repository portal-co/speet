# Zero-offset recompilation

**Status: landed (v1)** — see also [dual-backends.md](../guides/dual-backends.md), [abi-spec-redirects.md](abi-spec-redirects.md).

## Purpose

Run thin-runtime guests with **guest VA == WASM linear offset** (`host_mem_base = 0`) into a host-mirrored `__wasm_mem`, while keeping unrecompiled `.text` **out of that mirror** at `text_base`. That makes BridgeSupport data-pointer stubs safe to autogenerate (`__wasm_mem + guest_ptr`) without handing host libc a pointer into original machine code.

## Model

| Region | Placement |
|--------|-----------|
| `.data` / `.rodata` / heap / stack | Mirrored at guest VA into `__wasm_mem` |
| `[text_base, text_base+text_len)` | Unmapped / `mprotect(PROT_NONE)` — PC math still uses `text_base` |
| Fn-ptrs into text | Guest trampolines / `__speet_stub_for_pc` only |

[`MemoryModel::ZeroOffset`](../../crates/os/speet-link-core/src/image_layout.rs) shares identity address math with `OwnedLinear`; the difference is the text-unmapped policy at data-init and runtime.

`GuestImageLayout::from_loaded_binary` selects ZeroOffset when `data_end_max() <= ZERO_OFFSET_MAX_DATA_END` (16 MiB, matching the default `__wasm_mem` reservation); otherwise it keeps `OwnedLinear`. Suitability rejects an explicit ZeroOffset layout whose data span exceeds that cap. Use [`MemoryModel::HostOffset`](../../crates/os/speet-link-core/src/image_layout.rs) for packed high-VA layouts later.

## BridgeSupport

Under ZeroOffset, symbols with checked-in stub metadata and **no unbound fn-ptr args** (`speet_abi_stubs::zero_offset_data_pointer_safe`) are suitability-safe. Import stubs rewrite data pointers via `__wasm_mem +` (hand-written bridges take precedence; autogen covers the rest).

Same redirect policy as HostOffset / WASM-runtime OS — do not fork BridgeSupport rules per memory model (AGENTS §10).

## wasm-blitz

Per-`memory_index` [`MemBase`](../../../../wasm-blitz/crates/blitz-x86-64/src/naive.rs): `Raw` means base zero (skip `__wasm_mem` add). Thin runtime still uses `WasmMemSymbol` for mem0 (guest VA is an offset into the mirror). x86 elides `memarg.offset == 0` materialization.

## Deferred

- Self-read intercept that synthesizes bytes for the text hole (v1 hard-traps via `PROT_NONE` / OOB)
- Host-returned pointers into guest space, variadic marshal, fn-ptr returns ([abi-spec-redirects.md](abi-spec-redirects.md))
- HostOffset packing for multi-GB PIE images
