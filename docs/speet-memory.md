# speet-memory

**Crate:** `crates/helper/speet-memory`  
**Status: Active implementation.**

Architecture-agnostic memory abstractions for recompilers. Provides local-variable slot management, virtual memory mapping, and page table structures used by architecture frontends when translating guest memory accesses.

---

## Modules

### `layout` — local variable slot allocation

Tracks the assignment of guest architectural registers (and scratch temporaries) to WASM local variable slots. Architecture frontends call into this to allocate and look up slot indices rather than hard-coding offsets.

### `mapper` — address translation callbacks

The `set_mapper_callback` mechanism allows a recompiler consumer to inject a custom address translation function into the emitted WASM. The callback receives the virtual address and emits WASM instructions that produce the corresponding physical address on the stack. Used for page-table-aware address translation.

### `paging` — page table builders

Provides `standard_page_table_mapper()` (single-level 64KB paging, 64-bit physical), `standard_page_table_mapper_32()` (32-bit physical), and `multilevel_page_table_mapper()` / `multilevel_page_table_mapper_32()` (3-level hierarchical). These emit the WASM instruction sequences needed to walk a page table at runtime inside the translated binary.

Stack convention: the virtual address must be saved to the designated local before calling the mapper; the physical address is left on the stack afterward.

### `virtual` — virtual memory model

Tracks the guest virtual address space layout, including segment bases and sizes, for use by the mapper and page table walker.

### `mem` — memory abstraction

Low-level memory access helpers shared across the paging and mapper modules.

---

## Usage

Architecture frontends (`speet-riscv`, `speet-x86_64`, etc.) use `speet-memory` to:
1. Allocate WASM locals for guest registers via `layout`
2. Optionally attach a mapper callback via `mapper` for systems with paging
3. Emit page-table-walking WASM sequences via `paging` for paginated targets
