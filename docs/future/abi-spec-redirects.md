# ABI-Spec Redirect Stubs

**Status: Planned** — see [future-features.md](../future-features.md).

## Purpose

Let the [thin runtime](../thin-runtime-plan.md) safely support host imports that take pointer or function-pointer arguments, without requiring the guest and host to share an address space. Today, [`speet-runtime::suitability`](../../crates/os/speet-runtime/src/suitability.rs)'s `fn_ptr_free_allowlist` rejects any binary importing a symbol not on a small hardcoded allowlist, specifically because a function pointer into guest (WASM-linear) memory is meaningless to a host function expecting a real address. This plan replaces "reject" with "translate" for symbols that have checked-in stub metadata and runtime wiring.

## Two phases

1. **Ingest ABI-spec files** (`speet-abi-spec` crate). Parse ABI description files — starting with Apple's BridgeSupport XML format, since it already models functions, argument types, pointer-vs-value distinctions, and function-pointer parameters — into a structured `AbiSpec { functions: Vec<AbiFunction> }`. Pure data ingestion; feeds `speet_host_api::ImportManifest` construction. No codegen, no behavior change. **Status: shipped** (`crates/os/speet-abi-spec`).
2. **Generate redirect stubs** (`speet-abi-codegen` tool). Consume an `AbiSpec` and emit **Rust source** implementing one stub-emission function per `AbiFunction`, parameterized over the guest's configuration (arch, calling convention). Each generated function emits the actual address-translation / function-translation stub through `asm-arch`'s `WriterCore` emitter — the same emitter wasm-blitz already lowers WASM→native through (see [asm-arch-instruction-sync.md](../guides/asm-arch-instruction-sync.md)) — rather than a separately-compiled C shim string. A function-pointer argument gets a generated trampoline so the *host* can call back into guest code with correctly re-translated argument pointers. **Status: partial fn-ptr v1 shipped** — checked-in `write`/`exit`/`printf` under `src/generated/`; full libc/libSystem baseline codegen supported via `--all-symbols` into gitignored `.generated/`. At link time, [`GuestFuncCatalog`](../../crates/os/speet-recompile/src/guest_func_catalog.rs) drives one host stub per translated guest function (`__speet_guest_fn_N` → `__speet_invoke` → existing `__wasm_table` body); PLT hooks rewrite fn-ptr args through `env.__speet_stub_for_pc` before the redirect import call. **Deferred:** fn-ptr returns, load/store through guest memory, variadic host marshal, targets outside the translated set. Regenerate checked-in stubs via:

```bash
cargo run -p speet-abi-codegen -- \
  -i test-data/abi-spec/libc-minimal.bridgesupport.xml \
  -o crates/os/speet-abi-stubs/src/generated \
  -s write,exit,printf -a x86_64,aarch64
```

This is consumed at the address-to-label PC-check hook described in [thin-runtime-genericity.md](../guides/thin-runtime-genericity.md): when recompilation reaches a hooked address that needs a real stub rather than a plain host call, the generated emitter for that symbol runs inline.

## Why generate Rust code instead of interpreting the spec at recompile time

Checking in generated code (rather than shipping the raw ABI-spec files and interpreting them at guest-recompile time) means:

- Stub correctness can be checked with the same decode-coverage-style tests already used for asm-arch sync.
- Cross-platform portability is explicit and reviewable — a generated stub either exists and was checked in, or it doesn't exist and the symbol falls back to the suitability-gate denial.
- No ABI-spec parser needs to run (or even be linked in) at guest-recompile time for the common case.

## Scope discipline (do not skip — see AGENTS.md §9)

Licensing / check-in policy (same three bullets as AGENTS.md §9):

1. **libc / libSystem BridgeSupport-derived stub metadata MAY be checked in** (baseline is licensed for this use; checked-in subset under `speet-abi-stubs/src/generated/` via `STUB_SYMBOLS` / `-s` can grow with wiring; full baseline codegen into `speet-abi-stubs/.generated/` remains fine for local use).
2. **Custom / AI-generated implementations that do not copy original system source are fine** (runtime tunnels, WASI handlers, hand-written shims).
3. **Do NOT check in** BridgeSupport-derived definitions for **higher-level Apple frameworks** (beyond libSystem), GPLed SDK drops, other third-party licensed API surfaces, or **original system code** copied from Apple/GNU trees.

Growing POSIX coverage is an **implementation** task (suitability, manifest intercepts, redirect emit), not a codegen allowlist or “libc is too large” decision. `.generated/` is gitignored as local bulk convenience, not because libc/libSystem stubs are unlicensed for check-in.

```bash
# Checked-in / wired symbols (committed under src/generated/)
cargo run -p speet-abi-codegen -- \
  -i test-data/abi-spec/libc-minimal.bridgesupport.xml \
  -o crates/os/speet-abi-stubs/src/generated \
  -s write,exit,printf -a x86_64,aarch64

# Full libc/libSystem baseline locally (gitignored .generated/; optional bulk)
cargo run -p speet-abi-codegen -- \
  -i PATH/TO/libc.bridgesupport.xml \
  -o crates/os/speet-abi-stubs/.generated/libc \
  --all-symbols -a x86_64,aarch64
```

## Relationship to other plans

- **Zero-offset memory ([zero-offset.md](zero-offset.md)):** when the guest uses identity VA offsets into a host-mirrored `__wasm_mem` with unrecompiled `.text` unmapped, BridgeSupport **data-pointer** stubs are safe to autogenerate (`__wasm_mem + ptr`) without a shared process address space for code. Fn-ptr args still need trampolines.
- **Suitability gate:** once a symbol has an ABI spec with a generated stub, `speet-runtime::suitability` should move it off the `fn_ptr_free_allowlist` denial path and onto the "safe via translation" path — the allowlist and the ABI-spec set are the same tradeoff (deny vs. translate) and must be kept in sync deliberately, not left to drift apart. Under ZeroOffset, `zero_offset_data_pointer_safe` additionally admits data-pointer-only stub metadata.
- **Direct linking ([direct-linking.md](direct-linking.md)):** the redirect-stub layer is intentional, permanent indirection, not a stopgap for a future "link everything in-host" mode. The two are meant to coexist — stub generation doesn't need to be torn out if/when in-process linking lands.
- **Host JIT ([host-jit.md](host-jit.md)):** the same generated stub-emission functions are what a future asm→asm host-JIT path calls inline during translation instead of routing through a WASM import at all.
