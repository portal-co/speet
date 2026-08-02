# Redirect Shims via Virtual GOT / Lazy Pointers

**Status: Phase 2 in progress** — virtual GOT active; PC-check hooks retired on x86_64/aarch64.

## Purpose

Replace the interim **PC-check inline hook** path (see [thin-runtime-genericity.md](../guides/thin-runtime-genericity.md) principle 2) with a model where redirect stubs are normal out-of-bounds WASM functions and guest code pointers in `.got.plt` / lazy-bind sections are **pre-populated** to refer to them. This matches how real dynamic linkers work, generalizes to plugins and non-native targets, and avoids re-emitting import-call + synthetic-return sequences at every hooked decode slot.

## Future model

1. **Shim placement.** After translated guest functions, allocate redirect shim bodies through the same [`EntityIndexSpace`](../entity-index-space.md) discipline as the halt stub: one slot per hooked external, appended past the translated set, registered in the function table's `elem` segment. Multiple table indices may reference the same shim WASM function.

2. **Virtual code pointers.** Guest `.got.plt` / lazy-pointer cells hold **virtual guest PCs** — addresses that the existing addr→slot formula maps to shim table indices, not host addresses. Indirect calls through GOT therefore land on shims without a per-slot PC-check.

3. **Runtime PC/IP base.** Guest virtual addresses and table indices stay in sync only when the loader/recompiler agrees on a **PC base** for the image (where slot 0 starts, granularity per arch). Linking data sections requires this base when writing initial GOT contents and when resolving PC-relative relocations into the guest address space. Today the thin runtime assumes text starts at a known `start_addr`; the full model makes that base a first-class link input shared by text translation, data init, and GOT pre-population.

4. **Library identity.** [`ExternalTargetTable`](../plugin/speet-plugin-api) keys hooks by `(LibraryId, address)` so plugins can describe multiple images without address collisions. GOT entries are scoped per library the same way.

5. **ABI-spec stubs.** Generated redirect bodies from [abi-spec-redirects.md](abi-spec-redirects.md) emit through the same shim slots; fn-ptr arguments get trampolines inside the shim rather than blanket suitability denial.

## Why defer

- Requires **linking data sections** into the guest linear memory model with correct relocations, not just `.text` translation.
- Requires a stable **runtime PC/IP base** contract between load, GOT init, and indirect-call target computation (`base_func_offset` + slot formula).
- The interim PC-check path (WASM import call + synthetic return at hooked PCs) is sufficient for integrated thin-runtime v1 (`write`/`exit`/`execve` hooks) and remains valuable for debugging coverage until this lands.

## Interim (retired in Phase 2)

- ~~[`PltHookTable`](../plugin/speet-plugin-api) + per-slot PC check in `speet-x86_64` / `speet-aarch64`.~~
- Redirect dispatch is via patched GOT cells + [`PcSlotMap::with_redirect_shims`](../crates/helper/speet-reach/src/pc_slot_map.rs) + redirect shim WASM functions.
- [`PltCallPlan`](../crates/os/speet-recompile/src/plt.rs) resolves `PltRedirect::WasmImport` and `PltRedirect::NativeShim` hooks.
- Keys generalized to `(LibraryId, u64)`; default `LibraryId::MAIN_IMAGE` for same-platform ELF/Mach-O.

## Phase 1 foundation (landed)

The first-component plan adds shared infrastructure this doc's Phase 2 will consume:

- [`GuestImageLayout`](../crates/os/speet-link-core/src/image_layout.rs) — text/data/GOT contract, `shim_guest_pc`, `MemoryModel::HostOffset`.
- [`RuntimeLayoutParams`](../crates/os/speet-link-core/src/layout_params.rs) + user-specified [`TextBaseSnippet`](../crates/os/speet-link-core/src/layout_params.rs) — relocatable PC base via layout params (not hardcoded in arch frontends).
- Redirect shim **WASM function slots** before halt in `finish_module`; [`data_link`](../crates/os/speet-recompile/src/data_link.rs) patches GOT cells.
- `PltRedirect::NativeShim` resolves through manifest/`os_shim_*`; redirect shim bodies emitted for both WASM-import and native-shim addresses.
- Dual-backend contract documented in [`dual-backends.md`](../guides/dual-backends.md) and AGENTS.md §10.

PC-check hooks were removed in Phase 2; indirect PLT/GOT calls land on redirect shims via patched data + extended slot map.

## Phase 2 (in progress)

- Full [`GuestImageLayout`](../crates/os/speet-link-core/src/image_layout.rs) threaded through integrated recompile (relocs + `memory_model`).
- [`memory_access_for_model`](../crates/helper/speet-memory/src/factory.rs) shared factory for native emitters.
- Pre-pass [`__speet_host_mem_*`](../crates/runtime/os-host-api/src/manifest.rs) imports + [`host_mem_shim`](../crates/os/speet-recompile/src/host_mem_shim.rs) lowering pass (canonical multi-memory).
- Three-lane parity harness foundation: [`megabinary_parity.rs`](../crates/os/speet-recompile/tests/megabinary_parity.rs).

## Cross-links

- [thin-runtime-genericity.md](../guides/thin-runtime-genericity.md) — principle 2 (PC check, not instruction shape)
- [abi-spec-redirects.md](abi-spec-redirects.md) — stub generation consumed at shim emission
- [plugin-api.md](../plugin-api.md) §2 — `ExternalTargetPlugin` / resource-kind pattern
- [entity-index-space.md](../entity-index-space.md) — shim slot allocation
- [thin-runtime-plan.md](../thin-runtime-plan.md) — `PltRedirect::WasmImport` / `PltRedirect::Ambient` symmetry
