# speet-wasm

**Crate:** `crates/managed/speet-wasm`  
**Status: Active implementation.**

WASM-to-WASM transformer. Re-emits WASM functions from an input module with address-space translation and function index remapping, without going through the `Reactor`. This is the WASM frontend for the megabinary pipeline.

---

## Purpose

When the input binary is itself a WASM module (rather than a native ISA binary), the recompiler does not need the one-function-per-slot model: WASM functions are already structured and self-contained. `speet-wasm` handles this case by:

1. Parsing the input WASM module's function bodies.
2. Remapping all internal `call` and `return_call` indices to the megabinary's function index space (using the `EntityIndexSpace` offsets from Pass 1).
3. Rewriting memory access instructions if the megabinary uses a different address space base.
4. Emitting the remapped function bodies directly into the output `MegabinaryBuilder`.

---

## Why bypass the Reactor

WASM functions already have structured control flow (no arbitrary computed gotos). The `Reactor`'s job — turning unstructured guest CFGs into `return_call` chains — does not apply. Using `speet-wasm` directly avoids the overhead of a full CFG reconstruction pass.

See [docs/guides/linker.md §4](guides/linker.md) (per-emit-closure Reactor creation) — WASM-frontend emit closures use `LinkerInner` as `BaseContext` without constructing a `Reactor`.

---

## Integration

`speet-wasm` is registered as a frontend in the `FuncSchedule` alongside native frontends. Its emit closure receives the frozen `EntityIndexSpace` and emits remapped function bodies. The two-pass discipline (index registration before emission) applies identically.
