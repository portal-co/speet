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
1. **Push Return Metadata**: Statically determine the expected return PC and any parameter payloads.
2. **Native Call**: Emit a WASM `call` to the target function's index inside a validation scaffold.
3. **Return Validation**: After the call returns, validate the guest PC and throw an escape tag if it differs from expectations.

## 3. Speculative Return & Exception Escapes
Because guest code can manipulate the return address (e.g., buffer overflows or manual stack frame adjustment), the return must be validated.

### The "Speculative Return" Pattern:
```wasm
;; Conceptual pattern emitted by the Rust implementation:
block (type $ty_idx) ;; Block(FunctionType(ty_idx)) wraps call + validation
  try_table (..catch $UNKNOWN_RETURN_TAG..)
    call $target_func_idx

    ;; Validate the return location / PC
    local.get $pc
    i64.const $expected_return_pc
    i64.ne
    if
      ;; Throw the escape tag carrying the unexpected return payload
      local.get $pc ;; payload: unexpected PC (and any configured params)
      throw $UNKNOWN_RETURN_TAG
    end

    ;; Normal return path
    return
  catch $UNKNOWN_RETURN_TAG
    ;; Catch block in the Reactor will extract the payload and resume
    ;; execution at the correct location (via return_call or dispatcher)
  end
end
```

## 4. Exception Handler Integration
The `yecta::Reactor` wraps speculative calls in `TryTable`/catch regions and handles escapes.
- **Tag**: `$UNKNOWN_RETURN_TAG`.
- **Payload**: The unexpected `$pc` and any configured parameter payloads (the crate's `ret` helper loads params onto the exception payload before `throw`).
- **Behavior**: The catch block in the `Reactor` unpacks the payload, validates it, and performs the appropriate resume (for example via `return_call` or the module dispatcher) so execution continues at the correct guest location.

## 5. Benefits for Megabinary Environment
- **Performance**: Standard WASM `call` instructions are highly optimized by host engines compared to indirect jumps or dispatcher lookups.
- **Traceability**: Guest call stacks partially map to host WASM call stacks, aiding debugging.
- **Security**: The exception-based escape ensures that any deviation from the expected control flow is intercepted and re-validated by the vkernel-backed dispatcher.
