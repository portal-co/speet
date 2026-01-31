# Yecta Control Flow Integration: Static Speculative Call Lowering

## Overview
This document outlines the strategy for statically detecting and lowering ABI-compliant calls (x86_64 `call`, RISC-V `jal`/`jalr`) to native WebAssembly function calls within the megabinary environment.

## 1. Static ABI Call Detection
The recompiler toolchain analyzes binary patterns to identify standard function calls:
- **x86_64**: `call <offset>` (E8) and `call <reg>` (FF /2) following the System V ABI (return address pushed to stack).
- **RISC-V**: `jal x1, <offset>` and `jalr x1, x2, <offset>` following the standard calling convention (return address in `ra`/`x1`).

## 2. Lowering to WASM Function Calls
When an ABI call is detected, it is lowered to a standard WASM `call` (instead of a `return_call` or jump). This allows the megabinary to utilize the host's native call stack.

### Lowering Pattern (Call Site):
1. **Set Return Metadata**: Store the expected return address in both:
   - The guest `ra` register (x1) - visible to the callee
   - A hidden `expected_ra` local - for the exception handler
2. **Native Call**: Emit a WASM `call` wrapped in a `TryTable`/catch block

### Lowering Pattern (Return Site):
When an ABI-compliant return is detected (`jalr x0, ra, 0`):
1. **Throw Exception**: Use `ret` to throw the escape tag with all register state as payload
2. **Caller Catches**: The caller's try-catch block receives the exception
3. **Resume Execution**: Control flow continues after the call site with restored state

## 3. Speculative Return & Exception Escapes
The mechanism uses WebAssembly exceptions for call/return matching:

### On Call (e.g., `jal x1, target`):
```wasm
;; Set ra and expected_ra to return address
i32.const $return_addr
local.set $ra        ;; Guest register x1
i32.const $return_addr
local.set $expected_ra  ;; Hidden local for validation

;; Native call with exception handling
block (type $ty_idx)
  try_table (catch $ESCAPE_TAG 0)
    call $target_func_idx
    return
  end
end
;; Exception caught: state restored from payload, continue here
```

### On Return (e.g., `jalr x0, ra, 0`):
```wasm
;; Throw exception with all register state
local.get $x0
local.get $x1
...
local.get $pc
local.get $expected_ra
throw $ESCAPE_TAG
```

## 4. Exception Handler Integration
The `yecta::Reactor` wraps speculative calls in `TryTable`/catch regions:

- **Tag**: `$ESCAPE_TAG` (configured via `EscapeTag`)
- **Payload**: All guest registers (x0-x31, f0-f31, PC, expected_ra)
- **Behavior**: 
  1. Callee performs ABI return → throws exception
  2. Caller's catch block receives exception → state restored from payload
  3. Execution continues after the call instruction

### Why This Works
- **ABI-compliant code**: Call sets `ra`, return jumps to `ra` → exception thrown → caught by caller → normal continuation
- **Non-standard returns** (longjmp, corrupted ra, etc.): May not match the expected pattern → falls through to dispatcher
- **Nested calls**: Each call has its own try-catch → exceptions propagate correctly through the WASM call stack

## 5. Hidden State: `expected_ra` Local
Each generated function includes a hidden local (`expected_ra`) that tracks the return address set by speculative calls:

- **Local index**: 65 (after x0-x31, f0-f31, PC)
- **Purpose**: Validates that returns come from the expected callee
- **Set by**: Speculative call instructions (`jal x1`, `jalr x1`)
- **Used by**: ABI-compliant returns (`jalr x0, ra, 0`) to construct exception payload

## 6. Benefits for Megabinary Environment
- **Performance**: Standard WASM `call` instructions are highly optimized by host engines compared to indirect jumps or dispatcher lookups.
- **Traceability**: Guest call stacks map 1:1 to host WASM call stacks for ABI-compliant code, aiding debugging.
- **Security**: The exception-based escape ensures that any deviation from the expected control flow is intercepted and handled by the vkernel-backed dispatcher.
- **Compatibility**: Non-ABI code (longjmp, setjmp, computed gotos) still works through the standard dispatcher path.
