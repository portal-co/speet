# Megabinary Recompiler Integration Goals

This document outlines the goals for integrating the `speet-x86_64` and `speet-riscv` recompilers into a unified megabinary environment.

## 1. Unified Reactor Management
- **Goal**: Standardize how `yecta::Reactor` is instantiated and managed across different architectures.
- **Action**: Create a shared `MegabinaryContext` that holds the `Reactor` and manages `base_func_offset` globally across all recompiled ELFs in the container.
- **Benefit**: Ensures that function indices for `x86_64` code and `riscv64` code do not collide and can be linked into a single WASM module.

## 2. Global Function Indexing
- **Goal**: Implement a central registry for all entry points across all binaries in the megabinary.
- **Action**: Use a unified `FuncIdx` mapping that accounts for shared libraries (e.g., a single recompiled `libc` used by both `x86_64` and `riscv64` emulated processes).

## 3. Standardized Syscall Interface
- **Goal**: Unify the `CallbackContext` used in both recompilers to route all environment calls (ECALL in RISC-V, SYSCALL in x86_64) through a common vkernel bridge.
- **Action**: Implement a shared `VkernelInterface` trait that both `RiscVRecompiler` and `X86Recompiler` consume.

## 4. Multi-Architecture Dispatch
- **Goal**: Allow the megabinary to host code from different source architectures simultaneously.
- **Action**: The `_dispatch` function in the megabinary should be able to identify the source architecture of the requested hash and route to the appropriate recompiled block.

## 5. Shared Polyfill Linking
- **Goal**: Allow recompiled binaries to call into WASM-native polyfills (e.g., a single `libcrypto` WASM module).
- **Action**: Use `yecta`'s `Pool` and `Target::Static` to link recompiled calls directly to polyfill function indices.
