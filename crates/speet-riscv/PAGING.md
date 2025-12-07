# Custom Paging System for RISC-V Backends

## Overview

This paging system provides a software-based virtual memory mechanism for RISC-V backends. It translates RISC-V virtual addresses to physical addresses using a configurable page table, enabling memory isolation, address space management, and efficient memory mapping.

## Architecture

### Page Structure

- **Page Size**: 4096 bytes (4 KiB) - standard for most architectures
- **Page Alignment**: All pages are aligned to 4096-byte boundaries
- **Page Table**: Maps virtual page numbers to physical page base addresses

### Address Translation

A virtual address is split into two components:

```
Virtual Address (64-bit)
├─ Virtual Page Number (VPN): bits [63:12] - upper 52 bits
└─ Page Offset: bits [11:0] - lower 12 bits
```

Translation process:
1. Extract VPN from virtual address: `vpn = vaddr >> 12`
2. Look up physical page base in page table: `phys_page = page_table[vpn]`
3. Combine with offset: `phys_addr = phys_page + (vaddr & 0xFFF)`

### Configuration

The paging system is configured through the `map` callback function passed to compilation:

```rust
pub type MapCallback<'a> = &'a mut dyn FnMut(
    &mut Module,      // WebAssembly module
    &mut FunctionBody, // Current function
    Block,            // Current block
    Value,            // Virtual address (as WebAssembly Value)
    &mut UserState    // User-defined state
) -> Value;          // Returns physical address (as WebAssembly Value)
```

## Implementation Details

### Memory Access Flow

For every load and store instruction:

1. **Address Computation**: Base register + immediate offset
   ```wasm
   local.get $base    ;; Get base address
   i64.const $offset  ;; Load offset
   i64.add           ;; Compute virtual address
   ```

2. **Address Translation**: Call `map` callback
   ```rust
   let phys_addr = ctx.opts.map(module, f, ctx.block, virt_addr, &mut r.user);
   ```

3. **Memory Operation**: Use translated address
   ```wasm
   ;; phys_addr is on stack
   i64.load          ;; Perform actual load with physical address
   ```

### Example Implementation

A simple identity mapping (no translation):

```rust
fn identity_map(
    _module: &mut Module,
    _function: &mut FunctionBody,
    _block: Block,
    vaddr: Value,
    _user: &mut ()
) -> Value {
    vaddr // Return address unchanged
}
```

A page table-based mapping:

```rust
fn page_table_map(
    module: &mut Module,
    function: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    page_table_base: &mut Value
) -> Value {
    // Extract VPN: vaddr >> 12
    let vpn = function.add_op(
        block,
        Operator::I64Const { value: 12 },
        &[],
        &[Type::I64]
    );
    let vpn = function.add_op(
        block,
        Operator::I64ShrU,
        &[vaddr, vpn],
        &[Type::I64]
    );
    
    // Multiply VPN by 8 (size of u64 entry) for array indexing
    let eight = function.add_op(
        block,
        Operator::I64Const { value: 8 },
        &[],
        &[Type::I64]
    );
    let vpn_offset = function.add_op(
        block,
        Operator::I64Mul,
        &[vpn, eight],
        &[Type::I64]
    );
    
    // Add page table base address
    let entry_addr = function.add_op(
        block,
        Operator::I64Add,
        &[*page_table_base, vpn_offset],
        &[Type::I64]
    );
    
    // Load physical page base from page table
    let phys_page = function.add_op(
        block,
        Operator::I64Load {
            memory: MemoryArg {
                offset: 0,
                align: 3,
                memory: Memory::from(0)
            }
        },
        &[entry_addr],
        &[Type::I64]
    );
    
    // Extract page offset: vaddr & 0xFFF
    let mask = function.add_op(
        block,
        Operator::I64Const { value: 0xFFF },
        &[],
        &[Type::I64]
    );
    let page_offset = function.add_op(
        block,
        Operator::I64And,
        &[vaddr, mask],
        &[Type::I64]
    );
    
    // Combine: phys_page + page_offset
    function.add_op(
        block,
        Operator::I64Add,
        &[phys_page, page_offset],
        &[Type::I64]
    )
}
```

## Usage

### Basic Setup

```rust
use rift::{compile_with_hints, CompileOptions};

// Define user state for page table
struct PageTableState {
    page_table_base: Value,
}

// Create mapping function
fn my_mapper(
    module: &mut Module,
    function: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    state: &mut PageTableState
) -> Value {
    // Implement page table lookup as shown above
    // ...
}

// Configure compiler options
let mut user_state = PageTableState { /* ... */ };
let opts = CompileOptions {
    map: &mut |module, function, block, vaddr, user| {
        my_mapper(module, function, block, vaddr, user)
    },
    mem: Memory::from(0),
};

// Compile with paging
let wasm_module = compile_with_hints(
    &bytecode,
    entry_point,
    opts,
    &mut user_state,
    None, // No hint handler
)?;
```

### Advanced: Dynamic Page Tables

You can maintain page table state in WebAssembly globals or memory:

```rust
// Store page table base pointer in a global
let page_table_global = module.globals.push(GlobalData {
    ty: Type::I64,
    mutable: true,
    value: ConstExpr::i64_const(0x10000), // Page table at 64KB
});

// Reference global in mapper
fn dynamic_mapper(
    module: &mut Module,
    function: &mut FunctionBody,
    block: Block,
    vaddr: Value,
    _user: &mut ()
) -> Value {
    // Load page table base from global
    let pt_base = function.add_op(
        block,
        Operator::GlobalGet { global_index: 0 },
        &[],
        &[Type::I64]
    );
    
    // Continue with page table lookup using pt_base
    // ...
}
```

## Performance Considerations

1. **Overhead**: Each memory access adds ~10-15 WebAssembly instructions for address translation
2. **Optimization**: For identity mapping, the overhead is minimal (no-op function)
3. **Caching**: Consider caching translations in registers when the same page is accessed multiple times
4. **TLB Emulation**: For high-performance needs, implement a software TLB cache

## Security Considerations

1. **Bounds Checking**: The mapper can validate addresses and trap on invalid access
2. **Memory Isolation**: Different regions can be mapped to separate physical pages
3. **Permission Bits**: Extend page table entries to include read/write/execute flags

## Limitations

1. **No Hardware Support**: This is pure software translation with no hardware acceleration
2. **Single-Level Table**: Example uses flat page table; multi-level tables need additional logic
3. **Performance**: Adds overhead to every memory operation

## Future Enhancements

- Multi-level page tables for sparse address spaces
- TLB emulation for performance
- Permission checking (read/write/execute)
- Demand paging and page fault handling
- Shared memory regions between multiple RISC-V instances
