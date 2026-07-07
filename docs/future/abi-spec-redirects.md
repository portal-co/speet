# ABI-Spec Redirect Stubs

**Status: Planned** — see [future-features.md](../future-features.md).

## Purpose

Let the [thin runtime](../thin-runtime-plan.md) safely support host imports that take pointer or function-pointer arguments, without requiring the guest and host to share an address space. Today, [`speet-runtime::suitability`](../../crates/os/speet-runtime/src/suitability.rs)'s `fn_ptr_free_allowlist` rejects any binary importing a symbol not on a small hardcoded allowlist, specifically because a function pointer into guest (WASM-linear) memory is meaningless to a host function expecting a real address. This plan replaces "reject" with "translate" for a deliberately curated set of symbols.

## Two phases

1. **Ingest ABI-spec files** (`speet-abi-spec` crate). Parse ABI description files — starting with Apple's BridgeSupport XML format, since it already models functions, argument types, pointer-vs-value distinctions, and function-pointer parameters — into a structured `AbiSpec { functions: Vec<AbiFunction> }`. Pure data ingestion; feeds `speet_host_api::ImportManifest` construction. No codegen, no behavior change.
2. **Generate checked-in redirect stubs** (`speet-abi-codegen` tool). Consume an `AbiSpec` and emit **Rust source, checked into the repository**, implementing one stub-emission function per `AbiFunction`, parameterized over the guest's configuration (arch, calling convention). Each generated function emits the actual address-translation / function-translation stub through `asm-arch`'s `WriterCore` emitter — the same emitter wasm-blitz already lowers WASM→native through (see [asm-arch-instruction-sync.md](../guides/asm-arch-instruction-sync.md)) — rather than a separately-compiled C shim string. A function-pointer argument gets a generated trampoline so the *host* can call back into guest code with correctly re-translated argument pointers.

This is consumed at the address-to-label PC-check hook described in [thin-runtime-genericity.md](../guides/thin-runtime-genericity.md): when recompilation reaches a hooked address that needs a real stub rather than a plain host call, the generated emitter for that symbol runs inline.

## Why generate Rust code instead of interpreting the spec at recompile time

Checking in generated code (rather than shipping the raw ABI-spec files and interpreting them at guest-recompile time) means:

- Stub correctness can be checked with the same decode-coverage-style tests already used for asm-arch sync.
- Cross-platform portability is explicit and reviewable — a generated stub either exists and was checked in, or it doesn't exist and the symbol falls back to the suitability-gate denial.
- No ABI-spec parser needs to run (or even be linked in) at guest-recompile time for the common case.

## Scope discipline (do not skip — see AGENTS.md §9)

Only generate-and-check-in stubs for genuinely cross-platform behavior, plus a small, deliberately curated set of easily-ported per-OS surfaces (e.g. libSystem and other easily-ported macOS stubs, low-risk Linux stubs). **Do not** check in generated stubs for the entirety of any OS's API surface (e.g. all of macOS/libSystem) — each addition is a deliberate, individually reviewed inclusion, never a bulk import. This bounds the checked-in surface area and keeps stub review tractable as the set grows.

## Relationship to other plans

- **Suitability gate:** once a symbol has an ABI spec with a generated stub, `speet-runtime::suitability` should move it off the `fn_ptr_free_allowlist` denial path and onto the "safe via translation" path — the allowlist and the ABI-spec set are the same tradeoff (deny vs. translate) and must be kept in sync deliberately, not left to drift apart.
- **Direct linking ([direct-linking.md](direct-linking.md)):** the redirect-stub layer is intentional, permanent indirection, not a stopgap for a future "link everything in-host" mode. The two are meant to coexist — stub generation doesn't need to be torn out if/when in-process linking lands.
- **Host JIT ([host-jit.md](host-jit.md)):** the same generated stub-emission functions are what a future asm→asm host-JIT path calls inline during translation instead of routing through a WASM import at all.
