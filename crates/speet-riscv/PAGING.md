# Paging System Implementation - speet

See `r5-abi-specs/PAGING.md` for the complete paging specification.

## speet-Specific Implementation

**Target:** RISC-V to WebAssembly recompilation

**API Functions:**
- `standard_page_table_mapper()` - Single-level 64KB paging
- `multilevel_page_table_mapper()` - 3-level hierarchical paging

**Unified Context:**
All callbacks use `CallbackContext` for consistency.

**Stack Convention:**
- Virtual address must be saved to local 66 before calling mapper
- Physical address left on stack after calling

**Integration:**
```rust
// With static constant
recompiler.set_mapper_callback(&mut |ctx| {
    standard_page_table_mapper(ctx, 0x1000000u64, 0, use_i64)
});

// With runtime value from local variable (e.g., local 67)
recompiler.set_mapper_callback(&mut |ctx| {
    standard_page_table_mapper(ctx, PageTableBase::Local(67), 0, use_i64)
});

// With runtime value from global variable (e.g., global 0)
recompiler.set_mapper_callback(&mut |ctx| {
    standard_page_table_mapper(ctx, PageTableBase::Global(0), 0, use_i64)
});
```

See `src/lib.rs` for implementation details.
