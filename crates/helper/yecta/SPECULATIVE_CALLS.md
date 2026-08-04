# Yecta Control Flow Integration: Static Speculative Call Lowering

## Overview
This document outlines the strategy for statically detecting and lowering ABI-compliant calls (x86_64 `call`, RISC-V `jal`/`jalr`) to native WebAssembly function calls within the megabinary environment.

## 1. Static ABI Call Detection
The recompiler toolchain analyzes binary patterns to identify standard function calls:
- **x86_64**: `call <offset>` (E8) and `call <reg>` (FF /2) following the System V ABI (return address pushed to stack).
- **RISC-V**: `jal x1, <offset>` and `jalr x1, x2, <offset>` following the standard calling convention (return address in `ra`/`x1`).
- **AArch64**: `BL` / `BLR` (link address in `LR`/`x30`) and ABI `RET` (`RET` / `RET X30`).

## 2. Lowering to WASM Function Calls
When an ABI call is detected, it is lowered to a standard WASM `call` (instead of a `return_call` or jump). This allows the megabinary to utilize the host's native call stack.

Escape of a return-address mismatch is selected by [`CallEscape`](src/lib.rs):

| `CallEscape` | Call site | Mismatch signal | Function results |
|--------------|-----------|-----------------|------------------|
| `Jump` | `return_call` / jump (no native stack) | n/a | existing jump ABI |
| `Exception(EscapeTag)` | `call` inside hoisted `TryTable` | `throw` tag payload | `(register_file)` |
| `Flag` | bare `call` (no `TagSection`) | trailing `i32` flag (`0` match / `1` mismatch) | `(register_file, i32)` |

`SpeculativeEscape { escape, enable }` is the recompiler knob; `FLAG_SPEC` and `exception_spec(tag)` enable native-stack calls. AArch64 supports Flag (and Exception when a tag is wired); MIPS stays on `CallEscape::Jump` until speculative wiring exists.

### Lowering Pattern (Call Site):
**RISC-V / x86_64 (shared shape):**
1. **Set Return Metadata**: Store the expected return address in the guest ABI location (`ra` / stack) *and* a hidden `expected_ra` local via fixups.
2. **Native Call**:
   - **Exception**: WASM `call` inside a hoisted `TryTable`/catch region.
   - **Flag**: bare WASM `call`; after results, branch on the trailing `i32` flag (both arms empty), then restore the register file — same fallthrough-after-restore shape as EH catch.

### Lowering Pattern (Return Site):
When an ABI-compliant return is detected (`jalr x0, ra, 0` / `ret`):
1. **Compare** guest return address with `expected_ra`.
2. **Match**: push register file (+ `i32.const 0` in Flag mode) and `Return`.
3. **Mismatch**:
   - **Exception**: `throw` with the register-file payload.
   - **Flag**: push register file + `i32.const 1` and `Return` (no throw).
4. **Caller**: Exception catch / Flag `if` both restore and fall through after the call.

## 3. Speculative Return & Escape Strategies
Exception mode uses WebAssembly exception tags; Flag mode uses a trailing result:

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

## 4. Escape Integration (`CallEscape`)
The `yecta::Reactor` implements both escape strategies with shared fixup/`expected_ra` plumbing:

**Exception (`CallEscape::Exception`):**
- **Tag**: `$ESCAPE_TAG` (configured via `EscapeTag`)
- **Payload**: Guest register file for non-ABI returns
- **Call**: hoisted `TryTable` region; catch restores and falls through
- Requires a `TagSection` and a runtime with the exception-handling proposal (wasmi gap → wasmtime in e2e)

**Flag (`CallEscape::Flag`):**
- **No tags / no `TryTable`** — runs under plain wasmi
- **ABI**: `(register_file) -> (register_file, i32)`; flag `0` = match, `1` = mismatch
- **Call**: bare `call`, then empty `if`/`else` on the flag, then restore (mirrors EH catch fallthrough)
- Nested frames see the flag at the nearest call site (same “nearest handler” shape as `Catch::One`)

**Behavior (both modes):**
1. ABI-compliant return → `Return` with flag `0` / no throw
2. Non-ABI return → Flag `1` or `throw` → caller restores and continues after the call
3. Fixups isolate `expected_ra` per call site on the WASM call stack

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
// Or Flag mode (no TagSection):
// let params = JumpCallParams::call_flag(target_func, 66, pool)
//     .with_fixup(65, &expected_ra_snippet);
reactor.ji_with_params(ctx, params)?;
```

**x86_64 Example:**
```rust
// CALL example: use fixups to set expected_ra (local N) to return address
let expected_ra_snippet = ExpectedRaSnippet { return_addr };
let params = JumpCallParams::call(target_func, total_locals, escape_tag, pool)
    .with_fixup(expected_ra_local_idx, &expected_ra_snippet);
// Or: JumpCallParams::call_flag(target_func, total_locals, pool)...
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
- **Compatibility**: Non-ABI code (longjmp, setjmp, computed gotos / stack manipulation) still works through the mismatch path (throw or flag=`1`) and the standard dispatcher.
- **Runtime portability**: Flag mode does not need the exception-handling proposal — preferred for wasmi Lane A and thin-runtime assemble paths that omit `TagSection`.
- **Isolation**: The fixups mechanism ensures each call site gets a fresh, isolated `expected_ra` value without global state pollution.
- **Architecture Agnostic**: The same yecta fixups + `CallEscape` policy work for both register-based (RISC-V) and stack-based (x86_64) calling conventions.
