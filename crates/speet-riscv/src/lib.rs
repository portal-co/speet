//! # RISC-V to WebAssembly Recompiler
//!
//! This crate provides a RISC-V to WebAssembly static recompiler that translates
//! RISC-V machine code to WebAssembly using the yecta control flow library.
//!
//! ## Supported Extensions
//!
//! - **RV32I**: Base integer instruction set
//! - **M**: Integer multiplication and division
//! - **A**: Atomic instructions
//! - **C**: Compressed instructions
//! - **Zicsr**: Control and Status Register instructions
//! - **F**: Single-precision floating-point
//! - **D**: Double-precision floating-point
//!
//! ## Architecture
//!
//! The recompiler uses a register mapping approach where RISC-V registers are mapped
//! to WebAssembly local variables:
//! - Locals 0-31: Integer registers x0-x31
//! - Locals 32-63: Floating-point registers f0-f31
//! - Local 64: Program counter (PC)
//! - Locals 65+: Temporary variables for complex operations
//!
//! ## Usage
//!
//! ```no_run
//! use speet_riscv::RiscVRecompiler;
//! use rv_asm::{Inst, Xlen};
//!
//! // Create a recompiler instance
//! let mut recompiler = RiscVRecompiler::new();
//!
//! // Decode and translate instructions
//! let instruction_bytes: u32 = 0x00a50533; // add a0, a0, a0
//! let (inst, _is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
//! recompiler.translate_instruction(&inst, 0x1000);
//! ```
//!
//! ## RISC-V Specification Compliance
//!
//! This implementation follows the RISC-V Unprivileged Specification:
//! https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html
//!
//! Key specification quotes are included as documentation comments throughout the code.

#![no_std]
extern crate alloc;

use core::convert::Infallible;
use rv_asm::{Inst, Reg, FReg, Imm};
use wasm_encoder::{Function, Instruction, ValType};
use yecta::{Reactor, Pool, EscapeTag, TableIdx, TypeIdx};

/// RISC-V to WebAssembly recompiler
///
/// This structure manages the translation of RISC-V instructions to WebAssembly,
/// using the yecta reactor for control flow management.
pub struct RiscVRecompiler {
    reactor: Reactor<Infallible, Function>,
    #[allow(dead_code)]
    pool: Pool,
    #[allow(dead_code)]
    escape_tag: Option<EscapeTag>,
}

impl RiscVRecompiler {
    /// Create a new RISC-V recompiler instance
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    pub fn new_with_config(pool: Pool, escape_tag: Option<EscapeTag>) -> Self {
        Self {
            reactor: Reactor::default(),
            pool,
            escape_tag,
        }
    }

    /// Create a new RISC-V recompiler with default configuration
    pub fn new() -> Self {
        Self::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
        )
    }

    /// Initialize a function for translating a basic block
    ///
    /// Sets up locals for:
    /// - 32 integer registers (x0-x31)
    /// - 32 floating-point registers (f0-f31)
    /// - 1 program counter register
    /// - Additional temporary registers as needed
    pub fn init_function(&mut self, num_temps: u32) {
        // Integer registers: locals 0-31 (i32)
        // Float registers: locals 32-63 (f64)
        // PC: local 64 (i32)
        // Temps: locals 65+ (mixed types)
        let locals = [
            (32, ValType::I32),  // x0-x31
            (32, ValType::F64),  // f0-f31 (using F64 for both F and D)
            (1, ValType::I32),   // PC
            (num_temps, ValType::I32), // Temporary registers
        ];
        self.reactor.next(locals.into_iter(), 0);
    }

    /// Get the local index for an integer register
    fn reg_to_local(reg: Reg) -> u32 {
        reg.0 as u32
    }

    /// Get the local index for a floating-point register
    fn freg_to_local(freg: FReg) -> u32 {
        32 + freg.0 as u32
    }

    /// Get the local index for the program counter
    const fn pc_local() -> u32 {
        64
    }

    /// Emit instructions to load an immediate value
    fn emit_imm(&mut self, imm: Imm) -> Result<(), Infallible> {
        self.reactor.feed(&Instruction::I32Const(imm.as_i32()))
    }

    /// Translate a single RISC-V instruction to WebAssembly
    ///
    /// # Arguments
    /// * `inst` - The decoded RISC-V instruction
    /// * `pc` - Current program counter value
    pub fn translate_instruction(&mut self, inst: &Inst, pc: u32) -> Result<(), Infallible> {
        // Update PC
        self.reactor.feed(&Instruction::I32Const(pc as i32))?;
        self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;

        match inst {
            // RV32I Base Integer Instruction Set
            
            // Lui: Load Upper Immediate
            // RISC-V Specification Quote:
            // "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format.
            // LUI places the 32-bit U-immediate value into the destination register rd, filling in the
            // lowest 12 bits with zeros."
            Inst::Lui { uimm, dest } => {
                self.emit_imm(*uimm)?;
                self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
            }

            // Auipc: Add Upper Immediate to PC
            // RISC-V Specification Quote:
            // "AUIPC (add upper immediate to pc) is used to build pc-relative addresses and uses the
            // U-type format. AUIPC forms a 32-bit offset from the U-immediate, filling in the lowest
            // 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places
            // the result in register rd."
            Inst::Auipc { uimm, dest } => {
                self.reactor.feed(&Instruction::I32Const(pc as i32))?;
                self.emit_imm(*uimm)?;
                self.reactor.feed(&Instruction::I32Add)?;
                self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
            }

            // Jal: Jump And Link
            // RISC-V Specification Quote:
            // "The jump and link (JAL) instruction uses the J-type format, where the J-immediate encodes
            // a signed offset in multiples of 2 bytes. The offset is sign-extended and added to the
            // address of the jump instruction to form the jump target address."
            Inst::Jal { offset, dest } => {
                // Save return address (PC + 4) to dest
                if dest.0 != 0 {  // x0 is hardwired to zero
                    self.reactor.feed(&Instruction::I32Const(pc as i32 + 4))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Jump to PC + offset
                let target_pc = (pc as i32).wrapping_add(offset.as_i32()) as u32;
                // Here we would need to jump to the function representing target_pc
                // For now, we'll update the PC and let the caller handle the jump
                self.reactor.feed(&Instruction::I32Const(target_pc as i32))?;
                self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;
            }

            // Jalr: Jump And Link Register
            // RISC-V Specification Quote:
            // "The indirect jump instruction JALR (jump and link register) uses the I-type encoding.
            // The target address is obtained by adding the sign-extended 12-bit I-immediate to the
            // register rs1, then setting the least-significant bit of the result to zero."
            Inst::Jalr { offset, base, dest } => {
                // Save return address
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(pc as i32 + 4))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Compute target: (base + offset) & ~1
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*base)))?;
                self.emit_imm(*offset)?;
                self.reactor.feed(&Instruction::I32Add)?;
                self.reactor.feed(&Instruction::I32Const(-2))?; // ~1 in two's complement
                self.reactor.feed(&Instruction::I32And)?;
                self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;
            }

            // Branch Instructions
            // RISC-V Specification Quote:
            // "All branch instructions use the B-type instruction format. The 12-bit B-immediate encodes
            // signed offsets in multiples of 2 bytes. The offset is sign-extended and added to the address
            // of the branch instruction to give the target address."

            Inst::Beq { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::Eq)?;
            }

            Inst::Bne { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::Ne)?;
            }

            Inst::Blt { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::LtS)?;
            }

            Inst::Bge { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::GeS)?;
            }

            Inst::Bltu { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::LtU)?;
            }

            Inst::Bgeu { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, BranchOp::GeU)?;
            }

            // Load Instructions
            // RISC-V Specification Quote:
            // "Load and store instructions transfer a value between the registers and memory.
            // Loads are encoded in the I-type format and stores are S-type."

            Inst::Lb { offset, dest, base } => {
                self.translate_load(*base, *offset, *dest, LoadOp::I8)?;
            }

            Inst::Lh { offset, dest, base } => {
                self.translate_load(*base, *offset, *dest, LoadOp::I16)?;
            }

            Inst::Lw { offset, dest, base } => {
                self.translate_load(*base, *offset, *dest, LoadOp::I32)?;
            }

            Inst::Lbu { offset, dest, base } => {
                self.translate_load(*base, *offset, *dest, LoadOp::U8)?;
            }

            Inst::Lhu { offset, dest, base } => {
                self.translate_load(*base, *offset, *dest, LoadOp::U16)?;
            }

            // Store Instructions
            Inst::Sb { offset, src, base } => {
                self.translate_store(*base, *offset, *src, StoreOp::I8)?;
            }

            Inst::Sh { offset, src, base } => {
                self.translate_store(*base, *offset, *src, StoreOp::I16)?;
            }

            Inst::Sw { offset, src, base } => {
                self.translate_store(*base, *offset, *src, StoreOp::I32)?;
            }

            // Integer Computational Instructions
            // RISC-V Specification Quote:
            // "Integer computational instructions are either encoded as register-immediate operations
            // using the I-type format or as register-register operations using the R-type format."

            Inst::Addi { imm, dest, src1 } => {
                if src1.0 == 0 && dest.0 == 0 {
                    // nop - do nothing
                } else if src1.0 == 0 {
                    // li (load immediate) pseudoinstruction
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32Add)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slti { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32LtS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltiu { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32LtU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32Xor)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Ori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32Or)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Andi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32And)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32Shl)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32ShrU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srai { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32ShrS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Register-Register Operations
            Inst::Add { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Add)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sub { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Sub)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sll { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Shl)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slt { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32LtS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32LtU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xor { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Xor)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srl { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32ShrU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sra { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32ShrS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Or { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Or)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::And { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32And)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Fence: Memory ordering
            // RISC-V Specification Quote:
            // "The FENCE instruction is used to order device I/O and memory accesses as viewed by
            // other RISC-V harts and external devices or coprocessors."
            Inst::Fence { .. } => {
                // WebAssembly has a different memory model; in a single-threaded environment
                // or with WebAssembly's built-in atomics, explicit fences may not be needed
                // For now, we emit a no-op
            }

            // System calls
            Inst::Ecall => {
                // Environment call - implementation specific
                // Would need to be handled by runtime
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            Inst::Ebreak => {
                // Breakpoint - implementation specific
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // M Extension: Integer Multiplication and Division
            // RISC-V Specification Quote:
            // "This chapter describes the standard integer multiplication and division instruction-set
            // extension, which is named 'M' and contains instructions that multiply or divide values
            // held in two integer registers."

            Inst::Mul { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32Mul)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulh { dest, src1, src2 } => {
                // Multiply high signed-signed: returns upper 32 bits of 64-bit product
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    self.reactor.feed(&Instruction::I64Mul)?;
                    self.reactor.feed(&Instruction::I64Const(32))?;
                    self.reactor.feed(&Instruction::I64ShrS)?;
                    self.reactor.feed(&Instruction::I32WrapI64)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulhsu { dest, src1, src2 } => {
                // Multiply high signed-unsigned
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32U)?;
                    self.reactor.feed(&Instruction::I64Mul)?;
                    self.reactor.feed(&Instruction::I64Const(32))?;
                    self.reactor.feed(&Instruction::I64ShrS)?;
                    self.reactor.feed(&Instruction::I32WrapI64)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulhu { dest, src1, src2 } => {
                // Multiply high unsigned-unsigned
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32U)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I64ExtendI32U)?;
                    self.reactor.feed(&Instruction::I64Mul)?;
                    self.reactor.feed(&Instruction::I64Const(32))?;
                    self.reactor.feed(&Instruction::I64ShrU)?;
                    self.reactor.feed(&Instruction::I32WrapI64)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Div { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32DivS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Divu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32DivU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Rem { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32RemS)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Remu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32RemU)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-Point Single-Precision (F Extension)
            // RISC-V Specification Quote:
            // "This chapter describes the standard instruction-set extension for single-precision
            // floating-point, which is named 'F'"

            Inst::Flw { offset, dest, base } => {
                self.translate_fload(*base, *offset, *dest, FLoadOp::F32)?;
            }

            Inst::Fsw { offset, src, base } => {
                self.translate_fstore(*base, *offset, *src, FStoreOp::F32)?;
            }

            Inst::FaddS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Div)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtS { dest, src, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Sqrt)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-Point Double-Precision (D Extension)
            Inst::Fld { offset, dest, base } => {
                self.translate_fload(*base, *offset, *dest, FLoadOp::F64)?;
            }

            Inst::Fsd { offset, src, base } => {
                self.translate_fstore(*base, *offset, *src, FStoreOp::F64)?;
            }

            Inst::FaddD { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Add)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubD { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulD { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivD { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Div)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtD { dest, src, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64Sqrt)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point min/max operations
            Inst::FminS { dest, src1, src2 } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Min)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxS { dest, src1, src2 } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Max)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FminD { dest, src1, src2 } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Min)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxD { dest, src1, src2 } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Max)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point comparison operations
            Inst::FeqS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::F32Eq)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::F32Lt)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::F32Le)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FeqD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Eq)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Lt)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Le)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-point conversion operations
            // RISC-V Specification Quote:
            // "Floating-point-to-integer and integer-to-floating-point conversion instructions
            // are encoded in the OP-FP major opcode space."

            Inst::FcvtWS { dest, src, .. } => {
                // Convert single to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::I32TruncF32S)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuS { dest, src, .. } => {
                // Convert single to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::I32TruncF32U)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtSW { dest, src, .. } => {
                // Convert signed 32-bit integer to single
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32S)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to single
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32U)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtWD { dest, src, .. } => {
                // Convert double to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::I32TruncF64S)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuD { dest, src, .. } => {
                // Convert double to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::I32TruncF64U)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtDW { dest, src, .. } => {
                // Convert signed 32-bit integer to double
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64ConvertI32S)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to double
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64ConvertI32U)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSD { dest, src, .. } => {
                // Convert double to single
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDS { dest, src, .. } => {
                // Convert single to double (already stored as double, just move)
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point move operations
            Inst::FmvXW { dest, src } => {
                // Move bits from float register to integer register
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F32DemoteF64)?;
                    self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FmvWX { dest, src } => {
                // Move bits from integer register to float register
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ReinterpretI32)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Sign-injection operations for single-precision
            // RISC-V Specification Quote:
            // "FSGNJ.S, FSGNJN.S, and FSGNJX.S produce a result that takes all bits except
            // the sign bit from rs1."
            Inst::FsgnjS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src2)
                // We'll use a simple implementation using bit manipulation
                self.emit_fsgnj_s(*dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnS { dest, src1, src2 } => {
                // Result = magnitude(src1) with NOT(sign(src2))
                self.emit_fsgnj_s(*dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxS { dest, src1, src2 } => {
                // Result = magnitude(src1) with sign(src1) XOR sign(src2)
                self.emit_fsgnj_s(*dest, *src1, *src2, FsgnjOp::Sgnjx)?;
            }

            Inst::FsgnjD { dest, src1, src2 } => {
                self.emit_fsgnj_d(*dest, *src1, *src2, FsgnjOp::Sgnj)?;
            }

            Inst::FsgnjnD { dest, src1, src2 } => {
                self.emit_fsgnj_d(*dest, *src1, *src2, FsgnjOp::Sgnjn)?;
            }

            Inst::FsgnjxD { dest, src1, src2 } => {
                self.emit_fsgnj_d(*dest, *src1, *src2, FsgnjOp::Sgnjx)?;
            }

            // Fused multiply-add operations
            // Note: WebAssembly doesn't have fused multiply-add, so we emulate with separate ops
            Inst::FmaddS { dest, src1, src2, src3, .. } => {
                // dest = (src1 * src2) + src3
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubS { dest, src1, src2, src3, .. } => {
                // dest = (src1 * src2) - src3
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubS { dest, src1, src2, src3, .. } => {
                // dest = -(src1 * src2) + src3 = src3 - (src1 * src2)
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddS { dest, src1, src2, src3, .. } => {
                // dest = -(src1 * src2) - src3
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Neg)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaddD { dest, src1, src2, src3, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Add)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubD { dest, src1, src2, src3, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubD { dest, src1, src2, src3, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddD { dest, src1, src2, src3, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::F64Neg)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Atomic operations (A extension)
            // RISC-V Specification Quote:
            // "The atomic instruction set is divided into two subsets: the standard atomic
            // instructions (AMO) and load-reserved/store-conditional (LR/SC) instructions."
            // Note: WebAssembly atomics require special handling
            Inst::LrW { dest, addr, .. } => {
                // Load-reserved word
                // In WebAssembly, we'll implement this as a regular atomic load
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                    self.reactor.feed(&Instruction::I32AtomicLoad(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::ScW { dest, addr, src, .. } => {
                // Store-conditional word
                // In a simplified model, always succeed (return 0)
                // A full implementation would track reservations
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::I32AtomicStore(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(0))?; // Success
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // CSR Instructions (Zicsr extension)
            // RISC-V Specification Quote:
            // "The SYSTEM major opcode is used to encode all privileged instructions, as well
            // as the ECALL and EBREAK instructions and CSR instructions."
            // Note: CSR operations are system-specific and may need special runtime support
            Inst::Csrrw { dest, src, .. } |
            Inst::Csrrs { dest, src, .. } |
            Inst::Csrrc { dest, src, .. } => {
                // For now, we'll stub these out as they require system support
                // A real implementation would need to call into a CSR handler
                if dest.0 != 0 {
                    // Return zero as placeholder
                    self.reactor.feed(&Instruction::I32Const(0))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Silently ignore the write for now
                _ = src;
            }

            Inst::Csrrwi { dest, .. } |
            Inst::Csrrsi { dest, .. } |
            Inst::Csrrci { dest, .. } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(0))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // RV64 instructions - these are supported but require different handling
            Inst::Lwu { .. } |
            Inst::Ld { .. } |
            Inst::Sd { .. } |
            Inst::AddiW { .. } |
            Inst::SlliW { .. } |
            Inst::SrliW { .. } |
            Inst::SraiW { .. } |
            Inst::AddW { .. } |
            Inst::SubW { .. } |
            Inst::SllW { .. } |
            Inst::SrlW { .. } |
            Inst::SraW { .. } |
            Inst::MulW { .. } |
            Inst::DivW { .. } |
            Inst::DivuW { .. } |
            Inst::RemW { .. } |
            Inst::RemuW { .. } => {
                // RV64-specific instructions
                // For RV32 mode, these should not be executed
                // Stub them out with unreachable for now
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // RV64 floating-point conversions
            Inst::FcvtLS { .. } |
            Inst::FcvtLuS { .. } |
            Inst::FcvtSL { .. } |
            Inst::FcvtSLu { .. } |
            Inst::FcvtLD { .. } |
            Inst::FcvtLuD { .. } |
            Inst::FmvXD { .. } |
            Inst::FcvtDL { .. } |
            Inst::FcvtDLu { .. } |
            Inst::FmvDX { .. } => {
                // RV64 floating-point operations
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Atomic memory operations
            Inst::AmoW { .. } => {
                // Atomic memory operations would need special WebAssembly atomic support
                // Stub for now
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Floating-point classify
            Inst::FclassS { .. } |
            Inst::FclassD { .. } => {
                // Floating-point classify requires special handling
                // Stub for now
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Catch-all for any unhandled instructions
            _ => {
                // Unimplemented or unrecognized instruction
                // In a production system, this might log or report the error
                self.reactor.feed(&Instruction::Unreachable)?;
            }
        }

        Ok(())
    }

    /// Helper to translate branch instructions
    fn translate_branch(
        &mut self,
        src1: Reg,
        src2: Reg,
        offset: Imm,
        pc: u32,
        op: BranchOp,
    ) -> Result<(), Infallible> {
        // Load operands
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(src1)))?;
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(src2)))?;
        
        // Compare
        match op {
            BranchOp::Eq => self.reactor.feed(&Instruction::I32Eq)?,
            BranchOp::Ne => self.reactor.feed(&Instruction::I32Ne)?,
            BranchOp::LtS => self.reactor.feed(&Instruction::I32LtS)?,
            BranchOp::GeS => self.reactor.feed(&Instruction::I32GeS)?,
            BranchOp::LtU => self.reactor.feed(&Instruction::I32LtU)?,
            BranchOp::GeU => self.reactor.feed(&Instruction::I32GeU)?,
        }

        // Conditional update of PC
        self.reactor.feed(&Instruction::If(wasm_encoder::BlockType::Empty))?;
        let target_pc = (pc as i32).wrapping_add(offset.as_i32()) as u32;
        self.reactor.feed(&Instruction::I32Const(target_pc as i32))?;
        self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;
        self.reactor.feed(&Instruction::Else)?;
        self.reactor.feed(&Instruction::I32Const((pc + 4) as i32))?;
        self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;
        self.reactor.feed(&Instruction::End)?;

        Ok(())
    }

    /// Helper to translate load instructions
    fn translate_load(
        &mut self,
        base: Reg,
        offset: Imm,
        dest: Reg,
        op: LoadOp,
    ) -> Result<(), Infallible> {
        if dest.0 == 0 {
            return Ok(()); // x0 is hardwired to zero
        }

        // Compute address: base + offset
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load from memory
        match op {
            LoadOp::I8 => {
                self.reactor.feed(&Instruction::I32Load8S(wasm_encoder::MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }))?;
            }
            LoadOp::U8 => {
                self.reactor.feed(&Instruction::I32Load8U(wasm_encoder::MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }))?;
            }
            LoadOp::I16 => {
                self.reactor.feed(&Instruction::I32Load16S(wasm_encoder::MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }))?;
            }
            LoadOp::U16 => {
                self.reactor.feed(&Instruction::I32Load16U(wasm_encoder::MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }))?;
            }
            LoadOp::I32 => {
                self.reactor.feed(&Instruction::I32Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
            }
        }

        self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate store instructions
    fn translate_store(
        &mut self,
        base: Reg,
        offset: Imm,
        src: Reg,
        op: StoreOp,
    ) -> Result<(), Infallible> {
        // Compute address: base + offset
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load value to store
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(src)))?;

        // Store to memory
        match op {
            StoreOp::I8 => {
                self.reactor.feed(&Instruction::I32Store8(wasm_encoder::MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                }))?;
            }
            StoreOp::I16 => {
                self.reactor.feed(&Instruction::I32Store16(wasm_encoder::MemArg {
                    offset: 0,
                    align: 1,
                    memory_index: 0,
                }))?;
            }
            StoreOp::I32 => {
                self.reactor.feed(&Instruction::I32Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
            }
        }

        Ok(())
    }

    /// Helper to translate floating-point load instructions
    fn translate_fload(
        &mut self,
        base: Reg,
        offset: Imm,
        dest: FReg,
        op: FLoadOp,
    ) -> Result<(), Infallible> {
        // Compute address: base + offset
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load from memory
        match op {
            FLoadOp::F32 => {
                self.reactor.feed(&Instruction::F32Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
            }
            FLoadOp::F64 => {
                self.reactor.feed(&Instruction::F64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }))?;
            }
        }

        self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate floating-point store instructions
    fn translate_fstore(
        &mut self,
        base: Reg,
        offset: Imm,
        src: FReg,
        op: FStoreOp,
    ) -> Result<(), Infallible> {
        // Compute address: base + offset
        self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load value to store
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src)))?;

        // Store to memory
        match op {
            FStoreOp::F32 => {
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::F32Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }))?;
            }
            FStoreOp::F64 => {
                self.reactor.feed(&Instruction::F64Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }))?;
            }
        }

        Ok(())
    }

    /// Helper to emit sign-injection for single-precision floats
    fn emit_fsgnj_s(
        &mut self,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), Infallible> {
        // Sign injection uses bit manipulation on the float representation
        // Get magnitude from src1, sign from src2 (possibly modified)
        
        // Convert src1 to i32 to manipulate bits
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.reactor.feed(&Instruction::F32DemoteF64)?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        
        // Mask to keep only magnitude (clear sign bit): 0x7FFFFFFF
        self.reactor.feed(&Instruction::I32Const(0x7FFFFFFF))?;
        self.reactor.feed(&Instruction::I32And)?;
        
        // Get sign bit from src2
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.reactor.feed(&Instruction::F32DemoteF64)?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        
        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly: mask with 0x80000000
                self.reactor.feed(&Instruction::I32Const(i32::MIN))?; // 0x80000000
                self.reactor.feed(&Instruction::I32And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor.feed(&Instruction::I32Const(i32::MIN))?;
                self.reactor.feed(&Instruction::I32And)?;
                self.reactor.feed(&Instruction::I32Const(i32::MIN))?;
                self.reactor.feed(&Instruction::I32Xor)?; // Flip the sign bit
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits of src1 and src2
                // Need original src1 sign
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                self.reactor.feed(&Instruction::I32Xor)?;
                self.reactor.feed(&Instruction::I32Const(i32::MIN))?;
                self.reactor.feed(&Instruction::I32And)?;
            }
        }
        
        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I32Or)?;
        self.reactor.feed(&Instruction::F32ReinterpretI32)?;
        self.reactor.feed(&Instruction::F64PromoteF32)?;
        self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;
        
        Ok(())
    }

    /// Helper to emit sign-injection for double-precision floats
    fn emit_fsgnj_d(
        &mut self,
        dest: FReg,
        src1: FReg,
        src2: FReg,
        op: FsgnjOp,
    ) -> Result<(), Infallible> {
        // Similar to single-precision but using i64
        // Convert src1 to i64 to manipulate bits
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.reactor.feed(&Instruction::I64ReinterpretF64)?;
        
        // Mask to keep only magnitude (clear sign bit)
        self.reactor.feed(&Instruction::I64Const(0x7FFFFFFFFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?;
        
        // Get sign bit from src2
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.reactor.feed(&Instruction::I64ReinterpretF64)?;
        
        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly
                self.reactor.feed(&Instruction::I64Const(i64::MIN))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor.feed(&Instruction::I64Const(i64::MIN))?;
                self.reactor.feed(&Instruction::I64And)?;
                self.reactor.feed(&Instruction::I64Const(i64::MIN))?;
                self.reactor.feed(&Instruction::I64Xor)?;
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.reactor.feed(&Instruction::I64ReinterpretF64)?;
                self.reactor.feed(&Instruction::I64Xor)?;
                self.reactor.feed(&Instruction::I64Const(i64::MIN))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
        }
        
        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I64Or)?;
        self.reactor.feed(&Instruction::F64ReinterpretI64)?;
        self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;
        
        Ok(())
    }

    /// Finalize the function
    pub fn seal(&mut self) -> Result<(), Infallible> {
        self.reactor.seal(&Instruction::Unreachable)
    }

    /// Get the reactor (consumes self)
    pub fn into_reactor(self) -> Reactor<Infallible, Function> {
        self.reactor
    }
}

impl Default for RiscVRecompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Branch operation types
#[derive(Debug, Clone, Copy)]
enum BranchOp {
    Eq,
    Ne,
    LtS,
    GeS,
    LtU,
    GeU,
}

/// Load operation types
#[derive(Debug, Clone, Copy)]
enum LoadOp {
    I8,   // Load byte (sign-extended)
    U8,   // Load byte (zero-extended)
    I16,  // Load halfword (sign-extended)
    U16,  // Load halfword (zero-extended)
    I32,  // Load word
}

/// Store operation types
#[derive(Debug, Clone, Copy)]
enum StoreOp {
    I8,   // Store byte
    I16,  // Store halfword
    I32,  // Store word
}

/// Floating-point load operation types
#[derive(Debug, Clone, Copy)]
enum FLoadOp {
    F32,  // Load single-precision float
    F64,  // Load double-precision float
}

/// Floating-point store operation types
#[derive(Debug, Clone, Copy)]
enum FStoreOp {
    F32,  // Store single-precision float
    F64,  // Store double-precision float
}

/// Sign-injection operation types
#[derive(Debug, Clone, Copy)]
enum FsgnjOp {
    Sgnj,   // Copy sign from src2
    Sgnjn,  // Copy negated sign from src2
    Sgnjx,  // XOR signs of src1 and src2
}