# Custom Paging System for RISC-V to WebAssembly

See rift/PAGING.md for complete paging architecture. This covers WebAssembly-specific details.

## 64KB Pages with Multi-Level Support

- Single-level: flat page table indexed by bits [63:16]
- Multi-level: 3-level structure with 16-bit indices (bits 63:48, 47:32, 31:16)
- Page offset: bits [15:0]

## Usage

Set mapper callback to translate addresses:

```rust
recompiler.set_mapper_callback(&mut |ctx| {
    // Emit WASM instructions to translate address on stack
    // Input: virtual address
    // Output: physical address
    Ok(())
});
```

## Context Unification

All callbacks now receive `CallbackContext` for consistency.
