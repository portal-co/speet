//! # RISC-V to WebAssembly Recompiler
//!
//! This crate provides a RISC-V to WebAssembly static recompiler that translates
//! RISC-V machine code to WebAssembly using the yecta control flow library.
//!
//! ## Supported Extensions
//!
//! - **RV32I**: Base integer instruction set
//! - **M**: Integer multiplication and division
//! - **A**: Atomic instructions (LR/SC subset)
//! - **Zicsr**: Control and Status Register instructions (stubbed for runtime)
//! - **F**: Single-precision floating-point
//! - **D**: Double-precision floating-point
//!
//! Note: Compressed instructions (C extension) are automatically handled by the rv-asm
//! decoder, which transparently expands them to their full-length equivalents before
//! translation.
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
//! let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
//!
//! // Decode and translate instructions
//! let instruction_bytes: u32 = 0x00a50533; // add a0, a0, a0
//! let (inst, is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
//! recompiler.translate_instruction(&inst, 0x1000, is_compressed);
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
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use core::convert::Infallible;
use rv_asm::{Inst, Reg, FReg, Imm, Xlen, IsCompressed};
use wasm_encoder::{Function, Instruction, ValType};
use yecta::{Reactor, Pool, EscapeTag, TableIdx, TypeIdx, FuncIdx};

/// Information about a detected HINT instruction
///
/// RISC-V HINT instructions are instructions that write to x0 (which is hardwired
/// to zero) and thus have no architectural effect. In the rv-corpus test suite,
/// these are used as markers to indicate test case boundaries.
///
/// Common pattern: `addi x0, x0, N` where N is the test case number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HintInfo {
    /// Program counter where the HINT was encountered
    pub pc: u32,
    /// The immediate value from the HINT instruction (typically the test case number)
    pub value: i32,
}

/// Information about an encountered ECALL instruction
///
/// RISC-V ECALL (environment call) is used to make a request to the
/// execution environment (e.g., operating system).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcallInfo {
    /// Program counter where the ECALL was encountered
    pub pc: u32,
}

/// Information about an encountered EBREAK instruction
///
/// RISC-V EBREAK (environment break) is used for debugging,
/// typically to transfer control to a debugger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EbreakInfo {
    /// Program counter where the EBREAK was encountered
    pub pc: u32,
}

/// Context provided to HINT callbacks for code generation
///
/// This struct provides access to the WebAssembly instruction emitter,
/// allowing callbacks to generate code in response to HINT instructions.
pub struct HintContext<'a> {
    /// Reference to the reactor for emitting WebAssembly instructions
    pub reactor: &'a mut Reactor<Infallible, Function>,
}

impl<'a> HintContext<'a> {
    /// Emit a WebAssembly instruction
    ///
    /// # Example
    /// ```no_run
    /// # use speet_riscv::HintContext;
    /// # use wasm_encoder::Instruction;
    /// fn my_callback(hint: &speet_riscv::HintInfo, ctx: &mut HintContext) {
    ///     // Emit a NOP or other instruction based on the hint
    ///     ctx.emit(&Instruction::Nop).ok();
    /// }
    /// ```
    pub fn emit(&mut self, instruction: &Instruction) -> Result<(), Infallible> {
        self.reactor.feed(instruction)
    }
}

/// Trait for HINT instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when HINT
/// instructions are encountered during translation. Implementations receive
/// both the HINT information and a context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait HintCallback {
    /// Process a HINT instruction
    ///
    /// # Arguments
    /// * `hint` - Information about the detected HINT instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, hint: &HintInfo, ctx: &mut HintContext);
}

/// Blanket implementation of HintCallback for FnMut closures
impl<F> HintCallback for F
where
    F: FnMut(&HintInfo, &mut HintContext),
{
    fn call(&mut self, hint: &HintInfo, ctx: &mut HintContext) {
        self(hint, ctx)
    }
}

/// Trait for ECALL instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when ECALL
/// instructions are encountered during translation. Implementations receive
/// both the ECALL information and a context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait EcallCallback {
    /// Process an ECALL instruction
    ///
    /// # Arguments
    /// * `ecall` - Information about the detected ECALL instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, ecall: &EcallInfo, ctx: &mut HintContext);
}

/// Blanket implementation of EcallCallback for FnMut closures
impl<F> EcallCallback for F
where
    F: FnMut(&EcallInfo, &mut HintContext),
{
    fn call(&mut self, ecall: &EcallInfo, ctx: &mut HintContext) {
        self(ecall, ctx)
    }
}

/// Trait for EBREAK instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when EBREAK
/// instructions are encountered during translation. Implementations receive
/// both the EBREAK information and a context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait EbreakCallback {
    /// Process an EBREAK instruction
    ///
    /// # Arguments
    /// * `ebreak` - Information about the detected EBREAK instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, ebreak: &EbreakInfo, ctx: &mut HintContext);
}

/// Blanket implementation of EbreakCallback for FnMut closures
impl<F> EbreakCallback for F
where
    F: FnMut(&EbreakInfo, &mut HintContext),
{
    fn call(&mut self, ebreak: &EbreakInfo, ctx: &mut HintContext) {
        self(ebreak, ctx)
    }
}

/// RISC-V to WebAssembly recompiler
///
/// This structure manages the translation of RISC-V instructions to WebAssembly,
/// using the yecta reactor for control flow management.
/// 
/// Each instruction gets its own function (at 2-byte boundaries), and control flow
/// is managed through jumps between these functions using the yecta reactor.
/// PC values are used directly as function indices, offset by base_pc.
/// 
/// The lifetime parameters:
/// - `'cb` represents the lifetime of the callback reference
/// - `'ctx` represents the lifetime of data the callback may capture
pub struct RiscVRecompiler<'cb, 'ctx> {
    reactor: Reactor<Infallible, Function>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    /// Base PC address - subtracted from PC values to compute function indices
    base_pc: u32,
    /// Whether to track HINT instructions (disabled by default)
    track_hints: bool,
    /// Collected HINT instructions (when tracking is enabled)
    hints: Vec<HintInfo>,
    /// Optional callback for inline HINT processing
    hint_callback: Option<&'cb mut (dyn HintCallback + 'ctx)>,
}

impl<'cb, 'ctx> RiscVRecompiler<'cb, 'ctx> {
    /// Create a new RISC-V recompiler instance with full configuration
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    /// * `track_hints` - Whether to track HINT instructions for debugging/testing
    pub fn new_with_full_config(
        pool: Pool,
        escape_tag: Option<EscapeTag>,
        base_pc: u32,
        track_hints: bool,
    ) -> Self {
        Self {
            reactor: Reactor::default(),
            pool,
            escape_tag,
            base_pc,
            track_hints,
            hints: Vec::new(),
            hint_callback: None,
        }
    }

    /// Create a new RISC-V recompiler instance
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    pub fn new_with_config(pool: Pool, escape_tag: Option<EscapeTag>, base_pc: u32) -> Self {
        Self::new_with_full_config(pool, escape_tag, base_pc, false)
    }

    /// Create a new RISC-V recompiler with default configuration
    /// Uses base_pc of 0
    pub fn new() -> Self {
        Self::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0,
        )
    }

    /// Create a new RISC-V recompiler with a specified base PC
    ///
    /// # Arguments
    /// * `base_pc` - Base PC address - this is subtracted from instruction PCs to compute function indices
    pub fn new_with_base_pc(base_pc: u32) -> Self {
        Self::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            base_pc,
        )
    }

    /// Enable or disable HINT instruction tracking
    ///
    /// When enabled, the recompiler will collect information about HINT instructions
    /// (instructions that write to x0). This is useful for debugging and understanding
    /// test case boundaries in rv-corpus test files.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable HINT tracking
    pub fn set_hint_tracking(&mut self, enable: bool) {
        self.track_hints = enable;
        if !enable {
            self.hints.clear();
        }
    }

    /// Get the collected HINT instructions
    ///
    /// Returns a slice of all HINT instructions encountered during translation
    /// when tracking is enabled.
    pub fn get_hints(&self) -> &[HintInfo] {
        &self.hints
    }

    /// Clear the collected HINT instructions
    pub fn clear_hints(&mut self) {
        self.hints.clear();
    }

    /// Set a callback for inline HINT processing
    ///
    /// When a callback is set, it will be invoked immediately when a HINT instruction
    /// is encountered during translation. This allows for real-time processing of
    /// test case markers without needing to collect them all first.
    ///
    /// The callback receives both the HINT information and a context for code generation.
    ///
    /// # Arguments
    /// * `callback` - A mutable reference to a closure or function that takes a `&HintInfo` and `&mut HintContext`
    ///
    /// # Example
    /// ```no_run
    /// # use speet_riscv::{RiscVRecompiler, HintInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    /// let mut recompiler = RiscVRecompiler::new();
    /// let mut my_callback = |hint: &HintInfo, ctx: &mut HintContext| {
    ///     println!("Test case {} at PC 0x{:x}", hint.value, hint.pc);
    ///     // Optionally emit WebAssembly instructions
    ///     ctx.emit(&Instruction::Nop).ok();
    /// };
    /// recompiler.set_hint_callback(&mut my_callback);
    /// ```
    pub fn set_hint_callback(&mut self, callback: &'cb mut (dyn HintCallback + 'ctx)) {
        self.hint_callback = Some(callback);
    }

    /// Clear the HINT callback
    ///
    /// Removes any previously set HINT callback.
    pub fn clear_hint_callback(&mut self) {
        self.hint_callback = None;
    }

    /// Convert a PC value to a function index
    /// PC values are offset by base_pc and then divided by 2 for 2-byte alignment
    fn pc_to_func_idx(&self, pc: u32) -> FuncIdx {
        let offset_pc = pc.wrapping_sub(self.base_pc);
        FuncIdx(offset_pc / 2)
    }

    /// Initialize a function for a single instruction at the given PC
    ///
    /// Sets up locals for:
    /// - 32 integer registers (x0-x31)
    /// - 32 floating-point registers (f0-f31)
    /// - 1 program counter register
    /// - Additional temporary registers as needed
    ///
    /// # Arguments
    /// * `_pc` - Program counter for this instruction (used for documentation)
    /// * `_inst_len` - Length of instruction in 2-byte increments (1 for compressed, 2 for normal)
    /// * `num_temps` - Number of additional temporary registers needed
    fn init_function(&mut self, _pc: u32, _inst_len: u32, num_temps: u32) {
        // Integer registers: locals 0-31 (i32)
        // Float registers: locals 32-63 (f64)
        // PC: local 64 (i32)
        // Temps: locals 65+ (mixed types)
        let locals = [
            (32, ValType::I32),  // x0-x31
            (32, ValType::F64),  // f0-f31 (using F64 for both F and D with NaN-boxing)
            (1, ValType::I32),   // PC
            (num_temps, ValType::I32), // Temporary registers
        ];
        // The second argument is the length in 2-byte increments
        // Set to 2 to prevent infinite looping (yecta handles automatic fallthrough)
        self.reactor.next(locals.into_iter(), 2);
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

    /// Perform a jump to a target PC using yecta's jump API
    fn jump_to_pc(&mut self, target_pc: u32, params: u32) -> Result<(), Infallible> {
        let target_func = self.pc_to_func_idx(target_pc);
        self.reactor.jmp(target_func, params)
    }

    /// NaN-box a single-precision float value for storage in a double-precision register
    /// 
    /// RISC-V Specification Quote:
    /// "When multiple floating-point precisions are supported, then valid values of
    /// narrower n-bit types, n < FLEN, are represented in the lower n bits of an
    /// FLEN-bit NaN value, with the upper bits all 1s. We call this a NaN-boxed value."
    ///
    /// For F32 values in F64 registers: set upper 32 bits to all 1s
    fn nan_box_f32(&mut self) -> Result<(), Infallible> {
        // Convert F32 to I32, then to I64, OR with 0xFFFFFFFF00000000, reinterpret as F64
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        self.reactor.feed(&Instruction::I64ExtendI32U)?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF00000000_u64 as i64))?;
        self.reactor.feed(&Instruction::I64Or)?;
        self.reactor.feed(&Instruction::F64ReinterpretI64)?;
        Ok(())
    }

    /// Unbox a NaN-boxed single-precision value from a double-precision register
    ///
    /// Extract the F32 value from the lower 32 bits of the NaN-boxed F64 value
    fn unbox_f32(&mut self) -> Result<(), Infallible> {
        // Reinterpret F64 as I64, wrap to I32 (takes lower 32 bits), reinterpret as F32
        self.reactor.feed(&Instruction::I64ReinterpretF64)?;
        self.reactor.feed(&Instruction::I32WrapI64)?;
        self.reactor.feed(&Instruction::F32ReinterpretI32)?;
        Ok(())
    }

    /// Translate a single RISC-V instruction to WebAssembly
    ///
    /// This creates a separate function for the instruction at the given PC and
    /// handles jumps to other instructions using the yecta reactor's jump APIs.
    ///
    /// # Arguments
    /// * `inst` - The decoded RISC-V instruction
    /// * `pc` - Current program counter value
    /// * `is_compressed` - Whether the instruction is compressed (2 bytes vs 4 bytes)
    pub fn translate_instruction(&mut self, inst: &Inst, pc: u32, is_compressed: IsCompressed) -> Result<(), Infallible> {
        // Calculate instruction length in 2-byte increments
        let inst_len = match is_compressed {
            IsCompressed::Yes => 1, // 2 bytes
            IsCompressed::No => 2,  // 4 bytes
        };
        
        // Initialize function for this instruction
        self.init_function(pc, inst_len, 8);
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
                // Save return address (PC + inst_len * 2) to dest
                if dest.0 != 0 {  // x0 is hardwired to zero
                    let return_addr = pc + (inst_len * 2);
                    self.reactor.feed(&Instruction::I32Const(return_addr as i32))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Jump to PC + offset using yecta's jump API with PC-based indexing
                let target_pc = (pc as i32).wrapping_add(offset.as_i32()) as u32;
                self.jump_to_pc(target_pc, 65)?; // Pass all registers as parameters
                return Ok(()); // JAL handles control flow, no fallthrough
            }

            // Jalr: Jump And Link Register
            // RISC-V Specification Quote:
            // "The indirect jump instruction JALR (jump and link register) uses the I-type encoding.
            // The target address is obtained by adding the sign-extended 12-bit I-immediate to the
            // register rs1, then setting the least-significant bit of the result to zero."
            Inst::Jalr { offset, base, dest } => {
                // Save return address
                if dest.0 != 0 {
                    let return_addr = pc + (inst_len * 2);
                    self.reactor.feed(&Instruction::I32Const(return_addr as i32))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // JALR is indirect, so we need to compute the target dynamically
                // For now, we'll use the computed target and update PC
                // A full implementation would need dynamic dispatch through a table
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*base)))?;
                self.emit_imm(*offset)?;
                self.reactor.feed(&Instruction::I32Add)?;
                self.reactor.feed(&Instruction::I32Const(0xFFFFFFFE_u32 as i32))?; // ~1 mask
                self.reactor.feed(&Instruction::I32And)?;
                self.reactor.feed(&Instruction::LocalSet(Self::pc_local()))?;
                // For indirect jumps, we seal with unreachable as we can't statically determine target
                self.reactor.seal(&Instruction::Unreachable)?;
                return Ok(()); // JALR handles control flow, no fallthrough
            }

            // Branch Instructions
            // RISC-V Specification Quote:
            // "All branch instructions use the B-type instruction format. The 12-bit B-immediate encodes
            // signed offsets in multiples of 2 bytes. The offset is sign-extended and added to the address
            // of the branch instruction to give the target address."

            Inst::Beq { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::Eq)?;
                return Ok(()); // Branch handles control flow
            }

            Inst::Bne { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::Ne)?;
                return Ok(());
            }

            Inst::Blt { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::LtS)?;
                return Ok(());
            }

            Inst::Bge { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::GeS)?;
                return Ok(());
            }

            Inst::Bltu { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::LtU)?;
                return Ok(());
            }

            Inst::Bgeu { offset, src1, src2 } => {
                self.translate_branch(*src1, *src2, *offset, pc, inst_len, BranchOp::GeU)?;
                return Ok(());
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
                    // HINT instruction: addi x0, x0, imm
                    // This has no architectural effect since x0 is hardwired to zero.
                    // In rv-corpus, this is used to mark test case boundaries.
                    
                    let hint_info = HintInfo {
                        pc,
                        value: imm.as_i32(),
                    };
                    
                    // Track the HINT if tracking is enabled
                    if self.track_hints {
                        self.hints.push(hint_info);
                    }
                    
                    // Invoke callback if set
                    if let Some(ref mut callback) = self.hint_callback {
                        let mut ctx = HintContext {
                            reactor: &mut self.reactor,
                        };
                        callback.call(&hint_info, &mut ctx);
                    }
                    
                    // No WebAssembly code generation needed - this is a true no-op
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
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivS { dest, src1, src2, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Div)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtS { dest, src, .. } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sqrt)?;
                self.nan_box_f32()?;
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
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Min)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxS { dest, src1, src2 } => {
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Max)?;
                self.nan_box_f32()?;
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
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::F32Eq)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::F32Lt)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
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
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32TruncF32S)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuS { dest, src, .. } => {
                // Convert single to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32TruncF32U)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtSW { dest, src, .. } => {
                // Convert signed 32-bit integer to single
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32S)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to single
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32U)?;
                self.nan_box_f32()?;
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
                // RISC-V Specification Quote:
                // "FCVT.S.D converts double-precision float to single-precision float,
                // rounding according to the dynamic rounding mode."
                // Convert double to single with proper NaN-boxing
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDS { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.D.S converts single-precision float to double-precision float."
                // Unbox the NaN-boxed single value, then promote to double
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point move operations
            Inst::FmvXW { dest, src } => {
                // Move bits from float register to integer register
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FmvWX { dest, src } => {
                // Move bits from integer register to float register
                self.reactor.feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ReinterpretI32)?;
                self.nan_box_f32()?;
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
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubS { dest, src1, src2, src3, .. } => {
                // dest = (src1 * src2) - src3
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubS { dest, src1, src2, src3, .. } => {
                // dest = -(src1 * src2) + src3 = src3 - (src1 * src2)
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddS { dest, src1, src2, src3, .. } => {
                // dest = -(src1 * src2) - src3
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Neg)?;
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
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
                let _ = src;
            }

            Inst::Csrrwi { dest, .. } |
            Inst::Csrrsi { dest, .. } |
            Inst::Csrrci { dest, .. } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(0))?;
                    self.reactor.feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // RV64 instructions
            // These are RV64-specific and not supported in RV32 mode.
            // We emit a trap instruction to signal unsupported operation.
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
            Inst::RemuW { .. } |
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
                // RV64-specific instructions are not supported in this RV32 implementation.
                // In a real system, this could trigger an illegal instruction exception.
                // For now, we emit unreachable which will trap if executed.
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Advanced atomic memory operations
            // These require more sophisticated atomic support than simple LR/SC
            Inst::AmoW { .. } => {
                // AMO operations (AMOSWAP, AMOADD, etc.) need WebAssembly atomic RMW operations
                // Future implementation should map these to appropriate wasm atomic instructions
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Floating-point classify
            // These require examining the floating-point value's bit pattern
            Inst::FclassS { .. } |
            Inst::FclassD { .. } => {
                // FCLASS returns a 10-bit mask indicating the class of the floating-point number
                // (positive/negative infinity, normal, subnormal, zero, NaN, etc.)
                // This requires complex bit pattern analysis not yet implemented
                self.reactor.feed(&Instruction::Unreachable)?;
            }

            // Catch-all for any other unhandled instructions
            _ => {
                // This should ideally never be reached if all instruction variants are handled.
                // If it is reached, it indicates an instruction type that was added to rv-asm
                // but not yet implemented in this recompiler.
                self.reactor.feed(&Instruction::Unreachable)?;
                return Ok(()); // Don't fallthrough for unimplemented instructions
            }
        }

        // For most instructions that don't explicitly handle control flow,
        // yecta automatically handles fallthrough based on the len parameter in init_function
        Ok(())
    }

    /// Helper to translate branch instructions using yecta's ji API with custom Snippet
    fn translate_branch(
        &mut self,
        src1: Reg,
        src2: Reg,
        offset: Imm,
        pc: u32,
        _inst_len: u32,
        op: BranchOp,
    ) -> Result<(), Infallible> {
        let target_pc = (pc as i32).wrapping_add(offset.as_i32()) as u32;
        
        // Create a custom Snippet for the branch condition using a closure
        // The closure captures the registers and operation
        struct BranchCondition {
            src1: u32,
            src2: u32,
            op: BranchOp,
        }
        
        impl wax_core::build::InstructionOperatorSource<Infallible> for BranchCondition {
            fn emit(&self, sink: &mut (dyn wax_core::build::InstructionOperatorSink<Infallible> + '_)) -> Result<(), Infallible> {
                // Emit the same instructions as emit_instruction
                sink.instruction(&Instruction::LocalGet(self.src1))?;
                sink.instruction(&Instruction::LocalGet(self.src2))?;
                sink.instruction(&match self.op {
                    BranchOp::Eq => Instruction::I32Eq,
                    BranchOp::Ne => Instruction::I32Ne,
                    BranchOp::LtS => Instruction::I32LtS,
                    BranchOp::GeS => Instruction::I32GeS,
                    BranchOp::LtU => Instruction::I32LtU,
                    BranchOp::GeU => Instruction::I32GeU,
                })?;
                Ok(())
            }
        }
        
        impl wax_core::build::InstructionSource<Infallible> for BranchCondition {
            fn emit_instruction(
                &self,
                sink: &mut (dyn wax_core::build::InstructionSink<Infallible> + '_),
            ) -> Result<(), Infallible> {
                sink.instruction(&Instruction::LocalGet(self.src1))?;
                sink.instruction(&Instruction::LocalGet(self.src2))?;
                sink.instruction(&match self.op {
                    BranchOp::Eq => Instruction::I32Eq,
                    BranchOp::Ne => Instruction::I32Ne,
                    BranchOp::LtS => Instruction::I32LtS,
                    BranchOp::GeS => Instruction::I32GeS,
                    BranchOp::LtU => Instruction::I32LtU,
                    BranchOp::GeU => Instruction::I32GeU,
                })?;
                Ok(())
            }
        }
        
        let condition = BranchCondition {
            src1: Self::reg_to_local(src1),
            src2: Self::reg_to_local(src2),
            op,
        };
        
        // Use ji with condition for branch taken path
        // When condition is true, jump to target; yecta handles else/end automatically
        let target_func = self.pc_to_func_idx(target_pc);
        let target = yecta::Target::Static { func: target_func };
        
        self.reactor.ji(
            65,                     // params: pass all registers
            &BTreeMap::new(),       // fixups: none needed
            target,                 // target: branch target
            None,                   // call: not an escape call
            self.pool,              // pool: for indirect calls
            Some(&condition),       // condition: branch condition
        )?;

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
        self.unbox_f32()?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        
        // Mask to keep only magnitude (clear sign bit): 0x7FFFFFFF
        self.reactor.feed(&Instruction::I32Const(0x7FFFFFFF))?;
        self.reactor.feed(&Instruction::I32And)?;
        
        // Get sign bit from src2
        self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.unbox_f32()?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        
        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly: mask with 0x80000000
                self.reactor.feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor.feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
                self.reactor.feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32Xor)?; // Flip the sign bit
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits of src1 and src2
                // Need original src1 sign
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                self.reactor.feed(&Instruction::I32Xor)?;
                self.reactor.feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
            }
        }
        
        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I32Or)?;
        self.reactor.feed(&Instruction::F32ReinterpretI32)?;
        self.nan_box_f32()?;
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
                self.reactor.feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor.feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
                self.reactor.feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64Xor)?;
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits
                self.reactor.feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.reactor.feed(&Instruction::I64ReinterpretF64)?;
                self.reactor.feed(&Instruction::I64Xor)?;
                self.reactor.feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
        }
        
        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I64Or)?;
        self.reactor.feed(&Instruction::F64ReinterpretI64)?;
        self.reactor.feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;
        
        Ok(())
    }

    /// Translate a block of RISC-V bytecode starting at the given address
    ///
    /// This method decodes and translates multiple instructions, creating separate
    /// functions for each instruction and linking them with jumps.
    ///
    /// # Arguments
    /// * `bytes` - The bytecode to translate
    /// * `start_pc` - The starting program counter address
    /// * `xlen` - The XLEN mode (RV32 or RV64)
    ///
    /// # Returns
    /// The number of bytes successfully translated
    pub fn translate_bytes(&mut self, bytes: &[u8], start_pc: u32, xlen: Xlen) -> Result<usize, ()> {
        let mut offset = 0;
        
        while offset < bytes.len() {
            // Need at least 2 bytes for a compressed instruction
            if offset + 1 >= bytes.len() {
                break;
            }
            
            // Read instruction bytes (little-endian)
            let inst_word = if offset + 3 < bytes.len() {
                u32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ])
            } else {
                // Might be a compressed instruction at the end
                u32::from_le_bytes([bytes[offset], bytes[offset + 1], 0, 0])
            };
            
            // Decode the instruction
            let (inst, is_compressed) = match Inst::decode(inst_word, xlen) {
                Ok(result) => result,
                Err(_) => break, // Stop on decode error
            };
            
            let pc = start_pc + offset as u32;
            
            // Translate the instruction
            if let Err(_) = self.translate_instruction(&inst, pc, is_compressed) {
                break;
            }
            
            // Advance by instruction size
            offset += match is_compressed {
                IsCompressed::Yes => 2,
                IsCompressed::No => 4,
            };
        }
        
        Ok(offset)
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

impl<'cb, 'ctx> Default for RiscVRecompiler<'cb, 'ctx> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rv_asm::{Inst, Xlen};

    #[test]
    fn test_recompiler_creation() {
        let _recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        // Just ensure it can be created without panicking
    }

    #[test]
    fn test_addi_instruction() {
        // Test ADDI instruction: addi x1, x0, 42
        // Use base_pc to offset high addresses
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(42),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_add_instruction() {
        // Test ADD instruction: add x3, x1, x2
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_load_instruction() {
        // Test LW instruction: lw x1, 0(x2)
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Lw {
            offset: rv_asm::Imm::new_i32(0),
            dest: rv_asm::Reg(1),
            base: rv_asm::Reg(2),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_store_instruction() {
        // Test SW instruction: sw x1, 4(x2)
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Sw {
            offset: rv_asm::Imm::new_i32(4),
            src: rv_asm::Reg(1),
            base: rv_asm::Reg(2),
        };
        let inst = Inst::Sw {
            offset: rv_asm::Imm::new_i32(4),
            src: rv_asm::Reg(1),
            base: rv_asm::Reg(2),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_branch_instruction() {
        // Test BEQ instruction: beq x1, x2, offset
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Beq {
            offset: rv_asm::Imm::new_i32(8),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_mul_instruction() {
        // Test MUL instruction from M extension
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::Mul {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_fadd_instruction() {
        // Test FADD.S instruction from F extension
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let inst = Inst::FaddS {
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
            dest: rv_asm::FReg(1),
            src1: rv_asm::FReg(2),
            src2: rv_asm::FReg(3),
        };
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_decode_and_translate() {
        // Test decoding a real instruction and translating it
        // This is "addi a0, a0, 0" which is a common NOP-like instruction
        let instruction_bytes: u32 = 0x00050513;
        let (inst, is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
        
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        assert!(recompiler.translate_instruction(&inst, 0x1000, is_compressed).is_ok());
    }

    #[test]
    fn test_multiple_instructions() {
        // Test translating multiple instructions in sequence
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        // addi x1, x0, 5
        let inst1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&inst1, 0x1000, IsCompressed::No).is_ok());
        
        // addi x2, x0, 3
        let inst2 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(3),
            dest: rv_asm::Reg(2),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&inst2, 0x1004, IsCompressed::No).is_ok());
        
        // add x3, x1, x2  (should compute 5 + 3 = 8)
        let inst3 = Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };
        assert!(recompiler.translate_instruction(&inst3, 0x1008, IsCompressed::No).is_ok());
    }

    #[test]
    fn test_translate_from_bytes() {
        // Test translating from raw bytecode
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        // Simple program: addi x1, x0, 5 (0x00500093)
        let bytes = [0x93, 0x00, 0x50, 0x00]; // Little-endian
        
        let result = recompiler.translate_bytes(&bytes, 0x1000, Xlen::Rv32);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4); // Should have translated 4 bytes
    }

    #[test]
    fn test_translate_compressed_from_bytes() {
        // Test translating compressed instructions from bytes
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        // c.addi x1, 5 (0x0095) - compressed instruction
        let bytes = [0x95, 0x00];
        
        let result = recompiler.translate_bytes(&bytes, 0x1000, Xlen::Rv32);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2); // Should have translated 2 bytes
    }

    #[test]
    fn test_register_mapping() {
        // Test that register indices map correctly
        assert_eq!(RiscVRecompiler::reg_to_local(rv_asm::Reg(0)), 0);
        assert_eq!(RiscVRecompiler::reg_to_local(rv_asm::Reg(31)), 31);
        assert_eq!(RiscVRecompiler::freg_to_local(rv_asm::FReg(0)), 32);
        assert_eq!(RiscVRecompiler::freg_to_local(rv_asm::FReg(31)), 63);
        assert_eq!(RiscVRecompiler::pc_local(), 64);
    }

    #[test]
    fn test_hint_tracking_disabled_by_default() {
        // HINT tracking should be disabled by default
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        // Translate a HINT instruction: addi x0, x0, 1
        let hint_inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        
        assert!(recompiler.translate_instruction(&hint_inst, 0x1000, IsCompressed::No).is_ok());
        
        // Should not have collected any hints since tracking is disabled
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_tracking_enabled() {
        // Test HINT tracking when explicitly enabled
        let mut recompiler = RiscVRecompiler::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true, // Enable HINT tracking
        );
        
        // Translate a HINT instruction: addi x0, x0, 1
        let hint1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&hint1, 0x1000, IsCompressed::No).is_ok());
        
        // Translate another HINT: addi x0, x0, 2
        let hint2 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(2),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&hint2, 0x1004, IsCompressed::No).is_ok());
        
        // Should have collected both hints
        let hints = recompiler.get_hints();
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].pc, 0x1000);
        assert_eq!(hints[0].value, 1);
        assert_eq!(hints[1].pc, 0x1004);
        assert_eq!(hints[1].value, 2);
    }

    #[test]
    fn test_hint_vs_regular_addi() {
        // Test that regular ADDI instructions are not tracked as HINTs
        let mut recompiler = RiscVRecompiler::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true, // Enable HINT tracking
        );
        
        // Regular addi x1, x0, 5 (not a HINT)
        let regular_addi = Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&regular_addi, 0x1000, IsCompressed::No).is_ok());
        
        // Should not be tracked as a HINT
        assert_eq!(recompiler.get_hints().len(), 0);
        
        // Now translate a real HINT: addi x0, x0, 1
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&hint, 0x1004, IsCompressed::No).is_ok());
        
        // Should only have the HINT
        let hints = recompiler.get_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].pc, 0x1004);
        assert_eq!(hints[0].value, 1);
    }

    #[test]
    fn test_hint_clear() {
        // Test clearing collected HINTs
        let mut recompiler = RiscVRecompiler::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true,
        );
        
        // Collect some hints
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(recompiler.translate_instruction(&hint, 0x1000, IsCompressed::No).is_ok());
        assert_eq!(recompiler.get_hints().len(), 1);
        
        // Clear hints
        recompiler.clear_hints();
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_tracking_toggle() {
        // Test toggling HINT tracking on and off
        let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
        
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        
        // Initially disabled
        assert!(recompiler.translate_instruction(&hint, 0x1000, IsCompressed::No).is_ok());
        assert_eq!(recompiler.get_hints().len(), 0);
        
        // Enable tracking
        recompiler.set_hint_tracking(true);
        assert!(recompiler.translate_instruction(&hint, 0x1004, IsCompressed::No).is_ok());
        assert_eq!(recompiler.get_hints().len(), 1);
        
        // Disable tracking (should clear existing hints)
        recompiler.set_hint_tracking(false);
        assert_eq!(recompiler.get_hints().len(), 0);
        assert!(recompiler.translate_instruction(&hint, 0x1008, IsCompressed::No).is_ok());
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_from_rv_corpus_pattern() {
        // Test the actual pattern used in rv-corpus test files
        let mut recompiler = RiscVRecompiler::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true,
        );
        
        // Simulate test case markers from rv-corpus
        for test_case in 1..=5 {
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(test_case),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            let pc = 0x1000 + (test_case as u32 * 4);
            assert!(recompiler.translate_instruction(&hint, pc, IsCompressed::No).is_ok());
        }
        
        // Verify all test case markers were collected
        let hints = recompiler.get_hints();
        assert_eq!(hints.len(), 5);
        for (i, hint) in hints.iter().enumerate() {
            assert_eq!(hint.value, (i as i32) + 1);
        }
    }

    #[test]
    fn test_hint_callback_basic() {
        // Test basic callback functionality
        use alloc::vec::Vec;
        
        let mut collected = Vec::new();
        
        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
            
            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext| {
                collected.push(*hint);
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            // Translate some HINT instructions
            let hint1 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(1),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&hint1, 0x1000, IsCompressed::No).is_ok());
            
            let hint2 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(2),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&hint2, 0x1004, IsCompressed::No).is_ok());
            
            // Drop recompiler before checking results
        }
        
        // Verify callback was invoked
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].pc, 0x1000);
        assert_eq!(collected[0].value, 1);
        assert_eq!(collected[1].pc, 0x1004);
        assert_eq!(collected[1].value, 2);
    }

    #[test]
    fn test_hint_callback_with_tracking() {
        // Test that callback and tracking work together
        use alloc::vec::Vec;
        
        let mut callback_hints = Vec::new();
        let tracked_hints_result;
        
        {
            let mut recompiler = RiscVRecompiler::new_with_full_config(
                Pool {
                    table: TableIdx(0),
                    ty: TypeIdx(0),
                },
                None,
                0x1000,
                true, // Enable tracking
            );
            
            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext| {
                callback_hints.push(*hint);
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            // Translate a HINT
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(42),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&hint, 0x2000, IsCompressed::No).is_ok());
            
            // Clear callback and get tracked hints before dropping
            recompiler.clear_hint_callback();
            tracked_hints_result = recompiler.get_hints().to_vec();
            
            // Drop recompiler
        }
        
        // Both callback and tracking should have captured it
        assert_eq!(callback_hints.len(), 1);
        assert_eq!(callback_hints[0].value, 42);
        
        assert_eq!(tracked_hints_result.len(), 1);
        assert_eq!(tracked_hints_result[0].value, 42);
        assert_eq!(tracked_hints_result[0].pc, 0x2000);
    }

    #[test]
    fn test_hint_callback_clear() {
        // Test clearing the callback
        use alloc::vec::Vec;
        
        let mut collected = Vec::new();
        
        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext| {
                collected.push(hint.value);
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(1),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            
            // First HINT should invoke callback
            assert!(recompiler.translate_instruction(&hint, 0x1000, IsCompressed::No).is_ok());
            
            // Clear callback
            recompiler.clear_hint_callback();
            
            // Translate another HINT - callback should not be invoked
            let hint2 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(2),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&hint2, 0x1004, IsCompressed::No).is_ok());
            
            // Drop recompiler
        }
        
        // Verify callback was invoked only once
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], 1);
    }

    #[test]
    fn test_hint_callback_without_tracking() {
        // Test that callback works even when tracking is disabled
        use alloc::vec::Vec;
        
        let mut callback_values = Vec::new();
        
        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
            // Tracking is disabled by default
            
            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext| {
                assert_eq!(hint.value, 99);
                callback_values.push(hint.value);
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(99),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&hint, 0x1000, IsCompressed::No).is_ok());
            
            // Get hints before dropping
            assert_eq!(recompiler.get_hints().len(), 0);
            
            // Drop recompiler
        }
        
        // Callback should have been invoked
        assert_eq!(callback_values.len(), 1);
        assert_eq!(callback_values[0], 99);
    }

    #[test]
    fn test_hint_callback_no_invoke_for_regular_addi() {
        // Test that callback is NOT invoked for regular ADDI instructions
        let mut invoked = false;
        
        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
            
            let mut callback = |_hint: &HintInfo, _ctx: &mut HintContext| {
                invoked = true;
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            // Regular addi x1, x0, 5 (not a HINT)
            let regular_addi = Inst::Addi {
                imm: rv_asm::Imm::new_i32(5),
                dest: rv_asm::Reg(1),
                src1: rv_asm::Reg(0),
            };
            assert!(recompiler.translate_instruction(&regular_addi, 0x1000, IsCompressed::No).is_ok());
            
            // Drop recompiler
        }
        
        // Callback should NOT have been invoked
        assert!(!invoked);
    }

    #[test]
    fn test_hint_callback_with_code_generation() {
        // Test that callback can generate WebAssembly instructions
        use alloc::vec::Vec;
        
        let mut hint_values = Vec::new();
        
        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);
            
            let mut callback = |hint: &HintInfo, ctx: &mut HintContext| {
                hint_values.push(hint.value);
                // Generate a NOP instruction for each HINT
                ctx.emit(&Instruction::Nop).ok();
            };
            
            recompiler.set_hint_callback(&mut callback);
            
            // Translate HINTs
            for i in 1..=3 {
                let hint = Inst::Addi {
                    imm: rv_asm::Imm::new_i32(i),
                    dest: rv_asm::Reg(0),
                    src1: rv_asm::Reg(0),
                };
                assert!(recompiler.translate_instruction(&hint, 0x1000 + (i as u32 * 4), IsCompressed::No).is_ok());
            }
            
            // Drop recompiler
        }
        
        // Verify callback was invoked for all HINTs
        assert_eq!(hint_values.len(), 3);
        assert_eq!(hint_values[0], 1);
        assert_eq!(hint_values[1], 2);
        assert_eq!(hint_values[2], 3);
    }
}