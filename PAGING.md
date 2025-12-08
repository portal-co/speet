# Paging System Implementation - speet

See `r5-abi-specs/PAGING.md` for the complete paging specification.

## speet-Specific Implementation

**Target:** RISC-V to WebAssembly recompilation

**API Functions:**
- `standard_page_table_mapper()` - Single-level 64KB paging (64-bit physical)
- `standard_page_table_mapper_32()` - Single-level 64KB paging (32-bit physical)
- `multilevel_page_table_mapper()` - 3-level hierarchical paging (64-bit physical)
- `multilevel_page_table_mapper_32()` - 3-level hierarchical paging (32-bit physical)

**Integration:**
Uses `set_mapper_callback()` on `RiscVRecompiler` to inject custom address translation into recompiled WebAssembly code.

**Example:**
```rust
// 64-bit physical addresses
recompiler.set_mapper_callback(&mut |ctx| {
    standard_page_table_mapper(ctx, pt_base, 0, true)
});

// 32-bit physical addresses (4 GiB limit)
recompiler.set_mapper_callback(&mut |ctx| {
    standard_page_table_mapper_32(ctx, pt_base, 0, true)
});
```

**Stack Convention:**
- Virtual address must be saved to local 66 before calling the mapper
- Physical address is left on stack after calling

See `crates/speet-riscv/src/lib.rs` for implementation details.
