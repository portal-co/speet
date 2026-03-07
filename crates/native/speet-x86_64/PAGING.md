# x86_64 Paging and Memory Management

This document drafts the paging system for `speet-x86_64`, modeled after the `speet-riscv` implementation but adapted for x86_64's 4-level (or 5-level) page table structure.

## Goals
- Implement a `MapperCallback` for x86_64 similar to the RISC-V one.
- Support 4-level paging (PML4, PDPT, PD, PT) with 4KB pages.
- Integrate with the vkernel for CR3 management and TLB shootdowns.

## Proposed Implementation

### Mapper Context
The x86_64 recompiler will use a `CallbackContext` to pass the virtual address and receive the physical address.

### 4-Level Page Table Walker (WASM)
Similar to `multilevel_page_table_mapper` in RISC-V, but for x86_64:
- **Level 4 (PML4)**: Bits [47:39]
- **Level 3 (PDPT)**: Bits [38:30]
- **Level 2 (PD)**: Bits [29:21]
- **Level 1 (PT)**: Bits [20:12]
- **Offset**: Bits [11:0]

### Integration with CR3
- The `page_table_base` will be tied to the guest's `CR3` register.
- The recompiler will emit a call to the mapper for every memory operand that isn't a direct relative address.

### Security Directory Integration
Align with the RISC-V "Security Directory" model to protect page table entries from guest modification while allowing fast WASM-side walking.

## Usage in Megabinary
The vkernel will manage the `CR3` value for each emulated process. When a process-switch occurs, the vkernel updates the `PageTableBase::Global` index used by the recompiled code.
