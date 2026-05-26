# Module Builder

**Crates:** `crates/os/speet-module-target`, `crates/os/speet-module-builder`  
**Status: Active implementation.**

Abstract interface (`ModuleTarget`) and concrete implementation (`MegabinaryBuilder`) for assembling the final WASM module output.

---

## `ModuleTarget` — abstract interface

`crates/os/speet-module-target/src/lib.rs`

Defines the operations needed to build a WASM module incrementally:

- `add_import` — declare a WASM import (function, memory, table, global, tag)
- `add_global` — declare a WASM global with initial value
- `add_memory` — declare a WASM memory with limits
- `add_table` — declare a WASM table
- `add_data_segment` — add a data segment (active or passive)
- `add_function` — register a function body

Any WASM output sink implements `ModuleTarget`. This allows tests to use a simple in-memory recorder while production code uses `MegabinaryBuilder`.

---

## `MegabinaryBuilder` — concrete implementation

`crates/os/speet-module-builder/src/lib.rs`

The production `ModuleTarget` that assembles all translated binaries into a single WASM megabinary. Key features:

### Type deduplication

All WASM function types (signatures) are interned. When multiple translated binaries declare the same function type, `MegabinaryBuilder` returns a single canonical type index. This keeps the WASM type section linear in the number of distinct signatures rather than proportional to the number of translated binaries.

### `MegabinaryOutput`

The assembled module is produced as a `MegabinaryOutput` struct containing:
- The serialized WASM binary (suitable for passing to a WASM runtime)
- The function index map (guest PC → WASM function index, for debugging and `manifest.json` generation)

---

## Integration with `EntityIndexSpace`

`MegabinaryBuilder` consumes type, function, memory, table, and tag indices from the frozen `EntityIndexSpace` (see [docs/guides/linker.md §2](guides/linker.md)). It does not self-assign indices; all assignments happen in the `FuncSchedule` Pass 1 registration phase.
