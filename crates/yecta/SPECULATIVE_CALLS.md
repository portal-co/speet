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
**RISC-V:**
1. **Set Return Metadata**: Store the expected return address in both:
   - The guest `ra` register (x1) - visible to the callee
   - A hidden `expected_ra` local via the fixups mechanism - isolated per call
2. **Native Call**: Emit a WASM `call` wrapped in a `TryTable`/catch block

**x86_64:**
1. **Set Return Metadata**: Store the expected return address in:
   - Push return address to the guest stack (normal x86_64 behavior)
   - A hidden `expected_ra` local via the fixups mechanism - isolated per call
2. **Native Call**: Emit a WASM `call` wrapped in a `TryTable`/catch block

### Lowering Pattern (Return Site):
**RISC-V:**
When an ABI-compliant return is detected (`jalr x0, ra, 0`):
1. **Compare Return Addresses**: Check if `ra` matches `expected_ra`
2. **ABI-Compliant Path**: If they match, use direct WASM `Return` instruction
3. **Non-ABI Path**: If they don't match, throw the escape tag for validation
4. **Caller Handles**: The caller either receives a normal return or catches the exception

**x86_64:**
When an ABI-compliant return is detected (`ret` or `ret <imm>`):
1. **Compare Return Addresses**: Check if stack top matches `expected_ra`
2. **ABI-Compliant Path**: If they match, pop stack and use direct WASM `Return` instruction
3. **Non-ABI Path**: If they don't match, throw the escape tag for validation
4. **Caller Handles**: The caller either receives a normal return or catches the exception

## 3. Speculative Return & Exception Escapes
The mechanism uses WebAssembly exceptions for call/return matching:

### RISC-V Implementation

#### On Call (e.g., `jal x1, target`):
```wasm
;; Set ra to return address (normal register)
i32.const $return_addr
local.set $ra        ;; Guest register x1

;; Native call with fixups and exception handling
;; expected_ra set via fixup mechanism (parameter 65)
block (type $ty_idx)
  try_table (catch $ESCAPE_TAG 0)
    ;; Parameters 0-64 passed normally
    ;; Parameter 65 (expected_ra) set via fixup to $return_addr
    call $target_func_idx
    return
  end
end
;; Exception caught (non-ABI return): state restored from payload, continue here
```

#### On Return (e.g., `jalr x0, ra, 0`):
```wasm
;; Check if return is ABI-compliant
local.get $ra           ;; Current return address
local.get $expected_ra  ;; Expected return address from call
i32.eq                  ;; Compare

if                      ;; If they match
  return                ;; ABI-compliant: direct WASM return
else                    ;; If they don't match  
  ;; Non-ABI return: throw exception with all register state
  local.get $x0
  local.get $x1
  ...
  local.get $pc
  local.get $expected_ra
  throw $ESCAPE_TAG
end
```

### x86_64 Implementation

#### On Call (e.g., `call target`):
```wasm
;; Push return address to guest stack (normal x86_64 behavior)
local.get $rsp
i64.const 8
i64.sub
local.tee $rsp         ;; Decrement stack pointer

i64.const $return_addr
i64.store              ;; Push return address to stack

;; Native call with fixups and exception handling
;; expected_ra set via fixup mechanism (parameter N)
block (type $ty_idx)
  try_table (catch $ESCAPE_TAG 0)
    ;; All parameters passed normally
    ;; Parameter N (expected_ra) set via fixup to $return_addr
    call $target_func_idx
    return
  end
end
;; Exception caught (non-ABI return): state restored from payload, continue here
```

#### On Return (e.g., `ret`):
```wasm
;; Load return address from stack top
local.get $rsp
i64.load               ;; Stack top value (should be return address)
local.get $expected_ra ;; Expected return address from call
i64.eq                 ;; Compare

if                     ;; If they match
  ;; ABI-compliant return: pop stack and return
  local.get $rsp
  i64.const 8
  i64.add
  local.set $rsp       ;; Pop stack
  return               ;; Direct WASM return
else                   ;; If they don't match
  ;; Non-ABI return: throw exception with all register/stack state
  local.get $rax
  local.get $rbx
  ...
  local.get $rsp
  local.get $expected_ra
  throw $ESCAPE_TAG
end
```

## 4. Exception Handler Integration
The `yecta::Reactor` wraps speculative calls in `TryTable`/catch regions with fixup parameter handling:

- **Tag**: `$ESCAPE_TAG` (configured via `EscapeTag`)
- **Payload**: All guest registers (x0-x31, f0-f31, PC, expected_ra) for non-ABI returns
- **Fixups**: The `expected_ra` parameter (index 65) is set via yecta's fixup mechanism
- **Behavior**: 
  1. **ABI-compliant return** (`ra` matches `expected_ra`) → direct WASM `Return` → normal function return
  2. **Non-ABI return** (`ra` doesn't match) → throws exception → caught by caller → state restored
  3. **Normal execution**: continues after the call instruction

### Why This Works
**RISC-V:**
- **ABI-compliant code**: Call sets `ra` via fixup, return compares and uses direct return → optimal performance path
- **Non-standard returns** (longjmp, corrupted ra, etc.): Comparison fails → exception thrown → handled by dispatcher
- **Nested calls**: Each call has its own fixup-isolated `expected_ra` → exceptions propagate correctly through the WASM call stack
- **Performance**: Most returns are ABI-compliant and use direct WASM returns instead of exceptions

**x86_64:**
- **ABI-compliant code**: Call pushes return address and sets `expected_ra` via fixup, return compares stack top and uses direct return → optimal performance path
- **Non-standard returns** (longjmp, stack manipulation, etc.): Comparison fails → exception thrown → handled by dispatcher
- **Nested calls**: Each call has its own fixup-isolated `expected_ra` → exceptions propagate correctly through the WASM call stack
- **Performance**: Most returns are ABI-compliant and use direct WASM returns instead of exceptions
- **Stack integrity**: Stack operations continue normally, comparison ensures return address wasn't corrupted

## 5. Hidden State: `expected_ra` Local and Fixups
Each generated function includes a hidden local (`expected_ra`) that tracks the return address set by speculative calls:

**RISC-V:**
- **Local index**: 65 (after x0-x31, f0-f31, PC)
- **Purpose**: Validates that returns come from the expected callee by comparing with `ra` register
- **Set by**: Fixups mechanism during speculative call instructions (`jal x1`, `jalr x1`)
- **Isolation**: Each call gets a fresh `expected_ra` via fixups, ensuring proper isolation
- **Used by**: ABI-compliant returns (`jalr x0, ra, 0`) to compare with actual `ra` register

**x86_64:**
- **Local index**: Varies (after all registers, flags, PC, stack pointer, etc.)
- **Purpose**: Validates that returns come from the expected callee by comparing with stack top
- **Set by**: Fixups mechanism during speculative call instructions (`call`)
- **Isolation**: Each call gets a fresh `expected_ra` via fixups, ensuring proper isolation
- **Used by**: ABI-compliant returns (`ret`, `ret <imm>`) to compare with stack top value

### Fixups Mechanism
The yecta fixups system ensures `expected_ra` is set only for calls and isolated per call site:

**RISC-V Example:**
```rust
// JAL example: use fixups to set expected_ra (local 65) to return address
let expected_ra_snippet = ExpectedRaSnippet { return_addr, enable_rv64 };
let params = JumpCallParams::call(target_func, 66, escape_tag, pool)
    .with_fixup(65, &expected_ra_snippet);
reactor.ji_with_params(ctx, params)?;
```

**x86_64 Example:**
```rust
// CALL example: use fixups to set expected_ra (local N) to return address
let expected_ra_snippet = ExpectedRaSnippet { return_addr };
let params = JumpCallParams::call(target_func, total_locals, escape_tag, pool)
    .with_fixup(expected_ra_local_idx, &expected_ra_snippet);
reactor.ji_with_params(ctx, params)?;
```

This approach provides better isolation than manual local setting and ensures `expected_ra` is fresh for each call.

## 7. Architecture-Specific Implementation Details

### RISC-V Implementation
- **Call Detection**: `jal x1, <offset>` and `jalr x1, rs2, <offset>` where destination is `x1` (ra register)
- **Return Detection**: `jalr x0, ra, 0` (jump to ra with no link register saved)
- **Return Address Storage**: Guest register `x1` (ra) - visible to callee
- **Validation**: Compare `ra` register with `expected_ra` local
- **Stack**: No stack manipulation needed for basic call/return

### x86_64 Implementation  
- **Call Detection**: `call <offset>` (E8) and `call <reg>` (FF /2) instructions
- **Return Detection**: `ret` (C3) and `ret <imm>` (C2) instructions
- **Return Address Storage**: Guest stack (pushed by call, popped by return)
- **Validation**: Compare stack top with `expected_ra` local
- **Stack Management**: 
  - Call: Decrement RSP, store return address at [RSP]
  - Return: Compare [RSP] with expected, then increment RSP if ABI-compliant

### Key Differences
| Aspect | RISC-V | x86_64 |
|--------|---------|---------|
| **Return Address** | Register (`ra`/`x1`) | Stack top |
| **Call Overhead** | Set register | Push to stack |
| **Return Check** | Register comparison | Memory load + comparison |
| **ABI Compliance** | `jalr x0, ra, 0` | `ret` instruction |
| **Non-ABI Examples** | Modified `ra`, longjmp | Stack manipulation, longjmp |
| **Performance** | Register access | Memory access (cached) |

## 6. Benefits for Megabinary Environment
- **Performance**: 
  - Standard WASM `call` instructions are highly optimized by host engines
  - ABI-compliant returns use direct WASM `Return` instead of exceptions (both architectures)
  - Only non-ABI returns incur exception overhead
- **Traceability**: Guest call stacks map 1:1 to host WASM call stacks for ABI-compliant code, aiding debugging.
- **Security**: 
  - **RISC-V**: The comparison-based escape ensures `ra` register integrity
  - **x86_64**: The comparison-based escape ensures stack-based return address integrity
  - Any deviation from expected control flow is intercepted and handled by the vkernel-backed dispatcher
- **Compatibility**: 
  - **RISC-V**: Non-ABI code (longjmp, setjmp, computed gotos) still works through the exception path and standard dispatcher
  - **x86_64**: Non-ABI code (longjmp, setjmp, stack manipulation) still works through the exception path and standard dispatcher
- **Isolation**: The fixups mechanism ensures each call site gets a fresh, isolated `expected_ra` value without global state pollution.
- **Architecture Agnostic**: The same yecta fixups and exception mechanisms work for both register-based (RISC-V) and stack-based (x86_64) calling conventions.
