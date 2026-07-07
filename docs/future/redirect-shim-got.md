# Redirect Shims via Virtual GOT / Lazy Pointers

**Status: Planned (deferred)** — see [future-features.md](../future-features.md).

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

## Interim (current)

- [`PltHookTable`](../plugin/speet-plugin-api) + per-slot PC check in `speet-x86_64` / `speet-aarch64`.
- [`PltCallPlan`](../crates/os/speet-recompile/src/plt.rs) resolves `PltRedirect::WasmImport` hooks; `PltRedirect::Ambient` stays link-time only (no WASM PC-check emission).
- Keys generalized to `(LibraryId, u64)`; default `LibraryId::MAIN_IMAGE` for same-platform ELF/Mach-O.

## Cross-links

- [thin-runtime-genericity.md](../guides/thin-runtime-genericity.md) — principle 2 (PC check, not instruction shape)
- [abi-spec-redirects.md](abi-spec-redirects.md) — stub generation consumed at shim emission
- [plugin-api.md](../plugin-api.md) §2 — `ExternalTargetPlugin` / resource-kind pattern
- [entity-index-space.md](../entity-index-space.md) — shim slot allocation
- [thin-runtime-plan.md](../thin-runtime-plan.md) — `PltRedirect::WasmImport` / `PltRedirect::Ambient` symmetry
