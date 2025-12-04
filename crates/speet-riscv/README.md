# speet-riscv - RISC-V to WebAssembly Recompiler

A `no_std` compatible RISC-V to WebAssembly static recompiler that translates RISC-V machine code to WebAssembly using the [yecta](../yecta) control flow library.

## Features

- **RV32I Base Integer Instruction Set**: Complete support for the base integer instructions
- **RV64I Base Integer Instruction Set**: Runtime-gated support for RV64 instructions (64-bit integers)
- **M Extension**: Integer multiplication and division instructions (RV32M and RV64M)
- **A Extension**: Atomic memory operations (load-reserved/store-conditional)
- **F Extension**: Single-precision floating-point instructions
- **D Extension**: Double-precision floating-point instructions
- **Zicsr Extension**: Control and Status Register instructions (stubbed for runtime support)
- **Memory64 Support**: Optional i64 address calculations for WebAssembly memory64 mode
- **HINT Instruction Tracking**: Optional runtime-gated tracking of RISC-V HINT instructions (e.g., `addi x0, x0, N`) used in rv-corpus test markers
- **Comprehensive Coverage**: Includes fused multiply-add, sign-injection, conversions, and more

## Architecture

The recompiler uses a register mapping approach where RISC-V registers are mapped to WebAssembly local variables:

- **Locals 0-31**: Integer registers `x0`-`x31` (i32 for RV32, i64 for RV64)
- **Locals 32-63**: Floating-point registers `f0`-`f31` (stored as f64)
- **Local 64**: Program counter (PC)
- **Locals 65+**: Temporary variables for complex operations

### RV64 Mode

When RV64 support is enabled at runtime, the recompiler uses i64 locals for integer registers instead of i32. This allows proper handling of 64-bit integer operations. When memory64 is also enabled, memory operations use i64 addresses instead of i32 addresses, enabling access to memory beyond 4GB.

## RISC-V Specification Compliance

This implementation follows the [RISC-V Unprivileged Specification](https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html). Key specification quotes are included as documentation comments throughout the code.

The instruction decoding is handled by the [rv-asm](https://github.com/portal-co/rv-utils) library, which provides a robust, panic-free decoder tested exhaustively on all 32-bit values.

## Usage

### Basic Translation

```rust
use speet_riscv::RiscVRecompiler;
use rv_asm::{Inst, Xlen};

// Create a recompiler instance
let mut recompiler = RiscVRecompiler::new();

// Initialize a function with locals for registers
recompiler.init_function(8); // 8 additional temp locals

// Decode and translate instructions
let instruction_bytes: u32 = 0x00a50533; // add a0, a0, a0
let (inst, _is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
recompiler.translate_instruction(&inst, 0x1000);

// Add more instructions...

// Finalize the function
recompiler.seal();

// Get the reactor to generate WebAssembly
let reactor = recompiler.into_reactor();
```

### RV64 Support

To enable RV64 (64-bit) instruction support:

```rust
use speet_riscv::RiscVRecompiler;
use yecta::{Pool, TableIdx, TypeIdx};

// Create a recompiler with RV64 enabled
let mut recompiler = RiscVRecompiler::new_with_full_config(
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    None,
    0x1000,  // base_pc
    false,   // disable HINT tracking
    true,    // enable RV64
    false,   // disable memory64 (use i32 addresses)
);

// Now you can translate RV64 instructions like LD, SD, ADDIW, etc.
```

When RV64 is enabled, integer registers use i64 locals instead of i32. You can optionally enable memory64 to use i64 addresses for memory operations (required for accessing memory beyond 4GB in WebAssembly):

```rust
// Create a recompiler with both RV64 and memory64 enabled
let mut recompiler = RiscVRecompiler::new_with_full_config(
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    None,
    0x1000,  // base_pc
    false,   // disable HINT tracking
    true,    // enable RV64
    true,    // enable memory64
);
```

You can also control these settings dynamically:

```rust
// Create with default settings (RV32)
let mut recompiler = RiscVRecompiler::new();

// Enable RV64 support
recompiler.set_rv64_support(true);

// Enable memory64 mode
recompiler.set_memory64(true);

// Check current settings
if recompiler.is_rv64_enabled() {
    println!("RV64 is enabled");
}

if recompiler.is_memory64_enabled() {
    println!("Memory64 is enabled");
}
```

### HINT Instruction Tracking

RISC-V HINT instructions are special instructions that write to register `x0` (which is hardwired to zero) and thus have no architectural effect. In the [rv-corpus](https://github.com/portal-co/rv-corpus) test suite, these instructions (typically `addi x0, x0, N`) are used as markers to indicate test case boundaries, where `N` is the test case number.

#### Collecting HINTs

The recompiler can optionally track these HINT instructions to aid in debugging and test case identification:

```rust
use speet_riscv::RiscVRecompiler;
use yecta::{Pool, TableIdx, TypeIdx};

// Create a recompiler with HINT tracking enabled
let mut recompiler = RiscVRecompiler::new_with_full_config(
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    None,
    0x1000,  // base_pc
    true,    // enable HINT tracking
    false,   // disable RV64
    false,   // disable memory64
);

// Translate some code containing HINT markers...
// translate_instruction(...);

// Retrieve collected HINT information
for hint in recompiler.get_hints() {
    println!("Test case {} at PC 0x{:x}", hint.value, hint.pc);
}

// Clear collected hints if needed
recompiler.clear_hints();

// Toggle tracking on/off dynamically
recompiler.set_hint_tracking(false);
```

#### HINT Callbacks

For real-time processing of HINTs during translation, you can set a callback function. The callback uses the `HintCallback` trait, which is automatically implemented for all `FnMut` closures with the appropriate signature. The callback receives both the HINT information and a context for generating WebAssembly instructions:

```rust
use speet_riscv::{RiscVRecompiler, HintInfo, HintContext};
use wasm_encoder::Instruction;

let mut recompiler = RiscVRecompiler::new();

// Set a callback for inline HINT processing with code generation capability
// The HintCallback trait is automatically implemented for FnMut closures
let mut my_callback = |hint: &HintInfo, ctx: &mut HintContext| {
    println!("Encountered test case {} at PC 0x{:x}", hint.value, hint.pc);
    
    // Optionally emit WebAssembly instructions based on the HINT
    // For example, emit a NOP or custom marker instruction
    ctx.emit(&Instruction::Nop).ok();
};

recompiler.set_hint_callback(&mut my_callback);

// The callback will be invoked immediately when HINTs are encountered
// translate_instruction(...);

// Callbacks work independently of tracking - you can use both together or separately
```

You can also implement the `HintCallback` trait for custom types to create more complex callback handlers.

**Note**: HINT tracking is disabled by default for performance. Enable it only when debugging or analyzing test programs. Callbacks have minimal overhead and can be used independently. The callback's `HintContext` parameter provides access to emit WebAssembly instructions, allowing you to generate custom code in response to test markers.

#### ECALL and EBREAK Callbacks

Similar callback systems are available for ECALL (environment call) and EBREAK (breakpoint) instructions. These use the same trait-based design with dual lifetimes:

```rust
use speet_riscv::{RiscVRecompiler, EcallInfo, EbreakInfo, HintContext};
use wasm_encoder::Instruction;

let mut recompiler = RiscVRecompiler::new();

// Set an ECALL callback
let mut ecall_handler = |ecall: &EcallInfo, ctx: &mut HintContext| {
    println!("ECALL at PC 0x{:x}", ecall.pc);
    // Generate custom WebAssembly code for the environment call
    ctx.emit(&Instruction::Call(42)).ok();  // Example: call a WebAssembly function
};
recompiler.set_ecall_callback(&mut ecall_handler);

// Set an EBREAK callback
let mut ebreak_handler = |ebreak: &EbreakInfo, ctx: &mut HintContext| {
    println!("EBREAK at PC 0x{:x}", ebreak.pc);
    // Generate custom WebAssembly code for the breakpoint
    ctx.emit(&Instruction::Nop).ok();
};
recompiler.set_ebreak_callback(&mut ebreak_handler);

// Without callbacks, ECALL and EBREAK default to emitting Unreachable instructions
```

Both `EcallCallback` and `EbreakCallback` traits are automatically implemented for `FnMut` closures with the appropriate signature. The callbacks receive the instruction information and a context for emitting WebAssembly instructions, providing full flexibility to handle these system instructions according to your runtime requirements.

## Instruction Set Extensions

### RV32I - Base Integer Instructions

All base integer instructions are fully supported:
- Arithmetic: `ADD`, `SUB`, `ADDI`, etc.
- Logical: `AND`, `OR`, `XOR`, `ANDI`, `ORI`, `XORI`
- Shifts: `SLL`, `SRL`, `SRA`, `SLLI`, `SRLI`, `SRAI`
- Comparisons: `SLT`, `SLTU`, `SLTI`, `SLTIU`
- Branches: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`
- Jumps: `JAL`, `JALR`
- Loads: `LB`, `LH`, `LW`, `LBU`, `LHU`
- Stores: `SB`, `SH`, `SW`
- Upper immediates: `LUI`, `AUIPC`

**RISC-V Specification Quote:**
> "RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments."

### M Extension - Integer Multiplication and Division

**RISC-V Specification Quote:**
> "This chapter describes the standard integer multiplication and division instruction-set extension, which is named 'M' and contains instructions that multiply or divide values held in two integer registers."

Supported instructions:
- `MUL`: Multiply (lower 32 bits)
- `MULH`, `MULHSU`, `MULHU`: Multiply high (upper 32 bits)
- `DIV`, `DIVU`: Signed and unsigned division
- `REM`, `REMU`: Signed and unsigned remainder

### A Extension - Atomic Instructions

**RISC-V Specification Quote:**
> "The atomic instruction set is divided into two subsets: the standard atomic instructions (AMO) and load-reserved/store-conditional (LR/SC) instructions."

Supported instructions:
- `LR.W`: Load-reserved word
- `SC.W`: Store-conditional word
- Atomic memory operations (AMO) - stubbed for special runtime support

### F Extension - Single-Precision Floating-Point

**RISC-V Specification Quote:**
> "This chapter describes the standard instruction-set extension for single-precision floating-point, which is named 'F'"

Fully supported:
- Arithmetic: `FADD.S`, `FSUB.S`, `FMUL.S`, `FDIV.S`, `FSQRT.S`
- Fused multiply-add: `FMADD.S`, `FMSUB.S`, `FNMADD.S`, `FNMSUB.S`
- Sign injection: `FSGNJ.S`, `FSGNJN.S`, `FSGNJX.S`
- Min/max: `FMIN.S`, `FMAX.S`
- Comparisons: `FEQ.S`, `FLT.S`, `FLE.S`
- Conversions: `FCVT.W.S`, `FCVT.WU.S`, `FCVT.S.W`, `FCVT.S.WU`
- Moves: `FMV.X.W`, `FMV.W.X`
- Load/store: `FLW`, `FSW`

### D Extension - Double-Precision Floating-Point

**RISC-V Specification Quote:**
> "This chapter describes the standard double-precision floating-point instruction-set extension, which is named 'D' and adds double-precision floating-point computational instructions compliant with the IEEE 754-2008 arithmetic standard."

Fully supported (similar to F extension but for double-precision):
- All arithmetic, fused multiply-add, sign injection operations
- Conversions between single and double precision
- Integer conversions: `FCVT.W.D`, `FCVT.WU.D`, `FCVT.D.W`, `FCVT.D.WU`
- Load/store: `FLD`, `FSD`

### Zicsr Extension - CSR Instructions

**RISC-V Specification Quote:**
> "The SYSTEM major opcode is used to encode all privileged instructions, as well as the ECALL and EBREAK instructions and CSR instructions."

CSR instructions are stubbed to return zero and require runtime support for actual CSR management:
- `CSRRW`, `CSRRS`, `CSRRC`: CSR atomic read/write operations
- `CSRRWI`, `CSRRSI`, `CSRRCI`: CSR immediate operations

## Testing

The recompiler includes comprehensive unit tests covering:
- Basic instruction translation
- Register mapping
- Multiple instruction sequences
- All major instruction categories
- Integration with the rv-asm decoder

Run tests with:
```bash
cargo test -p speet-riscv
```

## Implementation Notes

### Sign-Injection Operations

Sign-injection instructions (`FSGNJ`, `FSGNJN`, `FSGNJX`) are implemented using bit manipulation:
- Convert floating-point values to integers
- Manipulate sign bits using bitwise operations
- Convert back to floating-point

This approach ensures correct behavior according to the RISC-V specification.

### Fused Multiply-Add

WebAssembly doesn't have native fused multiply-add instructions. The implementation uses separate multiply and add/subtract operations. While this may have slightly different rounding behavior, it maintains functional correctness.

### Atomic Operations

WebAssembly's atomic operations are used where possible. The LR/SC (load-reserved/store-conditional) implementation is simplified and may need enhancement for full correctness in multi-threaded environments.

### RV64 Support

RV64 support is now available and can be enabled at runtime using configuration flags. When enabled, the recompiler:

- Uses i64 locals for integer registers instead of i32
- Supports all RV64I instructions (LD, SD, LWU, ADDIW, ADDW, SUBW, etc.)
- Supports RV64M multiplication and division instructions (MULW, DIVW, REMW, etc.)
- Optionally uses i64 addresses for memory operations when memory64 is enabled

To enable RV64 support:

```rust
use speet_riscv::RiscVRecompiler;
use yecta::{Pool, TableIdx, TypeIdx};

// Create a recompiler with RV64 enabled
let mut recompiler = RiscVRecompiler::new_with_full_config(
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    None,
    0x1000,  // base_pc
    false,   // disable HINT tracking
    true,    // enable RV64
    false,   // disable memory64 (use i32 addresses)
);

// For memory64 support (i64 addresses):
let mut recompiler_mem64 = RiscVRecompiler::new_with_full_config(
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    None,
    0x1000,  // base_pc
    false,   // disable HINT tracking
    true,    // enable RV64
    true,    // enable memory64 (use i64 addresses)
);
```

You can also enable/disable RV64 and memory64 dynamically:

```rust
// Enable RV64 support
recompiler.set_rv64_support(true);

// Enable memory64 mode
recompiler.set_memory64(true);

// Check current settings
if recompiler.is_rv64_enabled() {
    println!("RV64 is enabled");
}
```

**Note**: RV64 floating-point conversion instructions (FCVT.L.S, FCVT.D.L, etc.) are currently stubbed with `unreachable` and require additional implementation.

## References

- [RISC-V Unprivileged Specification](https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html)
- [rv-asm Decoder Library](https://github.com/portal-co/rv-utils)
- [rv-corpus Test Suite](https://github.com/portal-co/rv-corpus)

## License

This crate is part of the Speet project and is dual-licensed under:

- [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) for open source use
- Proprietary licenses available from Portal Solutions LLC for commercial use

See the [main repository](../../COPYING.md) for details.
