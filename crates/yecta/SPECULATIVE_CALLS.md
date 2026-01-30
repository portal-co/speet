# Yecta Control Flow Integration: Static Speculative Call Lowering

## Overview
This document outlines the strategy for statically detecting and lowering ABI-compliant calls (x86_64 `call`, RISC-V `jal`/`jalr`) to native WebAssembly function calls within the megabinary environment.

## 1. Static ABI Call Detection
The recompiler toolchain analyzes binary patterns to identify standard function calls:
- **x86_64**: `call <offset>` (E8) and `call <reg>` (FF /2) following the System V ABI (return address pushed to stack).
- **RISC-V**: `jal x1, <offset>` and `jalr x1, x2, <offset>` following the standard calling convention (return address in `ra`/`x1`).

## 2. Lowering to WASM Function Calls
When an ABI call is detected, it is lowered to a standard WASM `call` (instead of a `return_call` or jump). This allows the megabinary to utilize the host's native call stack.

### Lowering Pattern:
1. **Push Return Metadata**: Statically determine the expected return PC.
2. **Native Call**: Emit a WASM `call` to the target function's index.
3. **Return Validation**: After the call returns, verify the guest PC.

## 3. Speculative Return & Exception Escapes
Because guest code can manipulate the return address (e.g., buffer overflows or manual stack frame adjustment), the return must be validated.

### The "Speculative Return" Pattern:
```wasm
;; 1. Call recompiled function
call $target_func_idx

;; 2. Validate return location
local.get $pc
i64.const $expected_return_pc
i64.ne
if
  ;; 3. UNKNOWN LOCATION: Throw exception
  ;; The handler will catch this and jump to the correct location.
  local.get $pc
  throw $UNKNOWN_RETURN_TAG
end
```

## 4. Exception Handler Integration
The `yecta::Reactor` will wrap these speculative calls in `try_table` blocks.
- **Tag**: `$UNKNOWN_RETURN_TAG`.
- **Payload**: The unexpected `$pc`.
- **Behavior**: The catch block extracts the `$pc` and performs a `jump_to_pc` (using `return_call` or the dispatcher) to resume execution at the new, non-speculative location.

## 5. Benefits for Megabinary Environment
- **Performance**: Standard WASM `call` instructions are highly optimized by host engines compared to indirect jumps or dispatcher lookups.
- **Traceability**: Guest call stacks partially map to host WASM call stacks, aiding debugging.
- **Security**: The exception-based escape ensures that any deviation from the expected control flow is intercepted and re-validated by the vkernel-backed dispatcher.
