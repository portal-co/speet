# speet-riscv - RISC-V to WebAssembly Recompiler

A `no_std` compatible RISC-V to WebAssembly static recompiler that translates RISC-V machine code to WebAssembly using the [yecta](../yecta) control flow library.

## Features

- **RV32I Base Integer Instruction Set**: Complete support for the base integer instructions
- **M Extension**: Integer multiplication and division instructions
- **A Extension**: Atomic memory operations (load-reserved/store-conditional)
- **F Extension**: Single-precision floating-point instructions
- **D Extension**: Double-precision floating-point instructions
- **Zicsr Extension**: Control and Status Register instructions (stubbed for runtime support)
- **Comprehensive Coverage**: Includes fused multiply-add, sign-injection, conversions, and more

## Architecture

The recompiler uses a register mapping approach where RISC-V registers are mapped to WebAssembly local variables:

- **Locals 0-31**: Integer registers `x0`-`x31`
- **Locals 32-63**: Floating-point registers `f0`-`f31` (stored as f64)
- **Local 64**: Program counter (PC)
- **Locals 65+**: Temporary variables for complex operations

## RISC-V Specification Compliance

This implementation follows the [RISC-V Unprivileged Specification](https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html). Key specification quotes are included as documentation comments throughout the code.

The instruction decoding is handled by the [rv-asm](https://github.com/portal-co/rv-utils) library, which provides a robust, panic-free decoder tested exhaustively on all 32-bit values.

## Usage

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

RV64-specific instructions are recognized but stubbed with `unreachable` since this implementation targets RV32. Future versions may add full RV64 support.

## References

- [RISC-V Unprivileged Specification](https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html)
- [rv-asm Decoder Library](https://github.com/portal-co/rv-utils)
- [rv-corpus Test Suite](https://github.com/portal-co/rv-corpus)

## License

This crate is part of the Speet project and is dual-licensed under:

- [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) for open source use
- Proprietary licenses available from Portal Solutions LLC for commercial use

See the [main repository](../../COPYING.md) for details.
