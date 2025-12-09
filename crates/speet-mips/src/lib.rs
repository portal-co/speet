//! # MIPS to WebAssembly Recompiler
//!
//! This crate provides a MIPS to WebAssembly static recompiler that translates
//! MIPS machine code to WebAssembly using the yecta control flow library.
//!
//! ## Supported Architectures
//!
//! - **MIPS32**: 32-bit MIPS instruction set
//! - **MIPS64**: 64-bit MIPS instruction set (when enabled)
//!
//! ## Supported Instructions
//!
//! - **Base integer operations**: ADD, SUB, MUL, DIV, AND, OR, XOR, NOR, shifts
//! - **Load/Store**: LW, SW, LH, SH, LB, SB, and their unsigned variants
//! - **Branch/Jump**: BEQ, BNE, BLEZ, BGTZ, J, JAL, JR, JALR
//! - **Immediate operations**: ADDI, ADDIU, ANDI, ORI, XORI, LUI
//! - **Special registers**: MFHI, MTHI, MFLO, MTLO
//! - **System calls**: SYSCALL, BREAK
//!
//! ## Architecture
//!
//! The recompiler uses a register mapping approach where MIPS registers are mapped
//! to WebAssembly local variables:
//! - Locals 0-31: General-purpose registers $0-$31
//! - Locals 32-33: HI/LO registers for multiplication/division results
//! - Local 34: Program counter (PC)
//! - Locals 35+: Temporary variables for complex operations
//!
//! ## Usage
//!
//! ```ignore
//! use speet_mips::MipsRecompiler;
//! use rabbitizer::Instruction;
//!
//! // Create a recompiler instance
//! let mut recompiler = MipsRecompiler::new_with_base_pc(0x1000);
//!
//! // Decode and translate instructions
//! let instruction_bytes: u32 = 0x21290001; // addi $t1, $t0, 1
//! let instruction = Instruction::new(instruction_bytes, 0x1000, rabbitizer::instr_category_enum::InstrCategory::CPU);
//! recompiler.translate_instruction(&instruction, &mut |a| Function::new(a.collect::<Vec<_>>()));
//! ```

#![no_std]

extern crate alloc;
use wax_core::build::InstructionSink;

use rabbitizer::{Instruction, InstrId, registers::GprO32};
use wasm_encoder::{Instruction as WasmInstruction, ValType};
use yecta::{EscapeTag, FuncIdx, Pool, Reactor, TableIdx, TypeIdx};

// Shared snippet to compute table index for indirect jumps
// idx = ((reg_value & ~3) - base_pc) >> 2
struct TableIndexSnippet {
    rs_local: u32,
    base_pc: u32,
}

impl<E> wax_core::build::InstructionOperatorSource<E> for TableIndexSnippet {
    fn emit(
        &self,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<E> + '_),
    ) -> Result<(), E> {
        sink.instruction(&WasmInstruction::LocalGet(self.rs_local))?;
        sink.instruction(&WasmInstruction::I32Const(0xFFFFFFFC_u32 as i32))?;
        sink.instruction(&WasmInstruction::I32And)?;
        sink.instruction(&WasmInstruction::I32Const(self.base_pc as i32))?;
        sink.instruction(&WasmInstruction::I32Sub)?;
        sink.instruction(&WasmInstruction::I32Const(2))?;
        sink.instruction(&WasmInstruction::I32ShrU)?;
        Ok(())
    }
}

impl<E> wax_core::build::InstructionSource<E> for TableIndexSnippet {
    fn emit_instruction(
        &self,
        sink: &mut (dyn wax_core::build::InstructionSink<E> + '_),
    ) -> Result<(), E> {
        sink.instruction(&WasmInstruction::LocalGet(self.rs_local))?;
        sink.instruction(&WasmInstruction::I32Const(0xFFFFFFFC_u32 as i32))?;
        sink.instruction(&WasmInstruction::I32And)?;
        sink.instruction(&WasmInstruction::I32Const(self.base_pc as i32))?;
        sink.instruction(&WasmInstruction::I32Sub)?;
        sink.instruction(&WasmInstruction::I32Const(2))?;
        sink.instruction(&WasmInstruction::I32ShrU)?;
        Ok(())
    }
}

/// Information about an encountered SYSCALL instruction
///
/// MIPS SYSCALL is used to make a request to the operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyscallInfo {
    /// Program counter where the SYSCALL was encountered
    pub pc: u32,
    /// System call number (typically in $v0/$2)
    pub syscall_number: u32,
}

/// Information about an encountered BREAK instruction
///
/// MIPS BREAK is used for debugging and software breakpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakInfo {
    /// Program counter where the BREAK was encountered
    pub pc: u32,
    /// Break code (immediate value)
    pub code: u32,
}

/// Unified context for all callbacks
///
/// This struct provides access to the WebAssembly instruction emitter and other
/// compilation state. It is passed to all callbacks (SYSCALL, BREAK, mapper).
pub struct CallbackContext<'a, E, F: InstructionSink<E>> {
    /// Reference to the reactor for emitting WebAssembly instructions
    pub reactor: &'a mut Reactor<E, F>,
}

impl<'a, E, F: InstructionSink<E>> CallbackContext<'a, E, F> {
    /// Emit a WebAssembly instruction
    pub fn emit(&mut self, instruction: &WasmInstruction) -> Result<(), E> {
        self.reactor.feed(instruction)
    }
}

/// Trait for address mapping callbacks (paging support)
///
/// This trait defines interface for callbacks that translate virtual addresses
/// to physical addresses. The callback receives the virtual address on the Wasm stack
/// and should leave the physical address on the stack.
pub trait MapperCallback<E, F: InstructionSink<E>> {
    /// Translate a virtual address to a physical address
    ///
    /// # Stack State
    /// - Input: Virtual address (i32)
    /// - Output: Physical address (i32)
    fn call(&mut self, ctx: &mut CallbackContext<E, F>) -> Result<(), E>;
}

/// Blanket implementation of MapperCallback for FnMut closures
impl<E, F: InstructionSink<E>, T> MapperCallback<E, F> for T
where
    T: FnMut(&mut CallbackContext<E, F>) -> Result<(), E>,
{
    fn call(&mut self, ctx: &mut CallbackContext<E, F>) -> Result<(), E> {
        self(ctx)
    }
}

/// Trait for SYSCALL instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when SYSCALL
/// instructions are encountered during translation.
pub trait SyscallCallback<E, F: InstructionSink<E>> {
    /// Process a SYSCALL instruction
    ///
    /// # Arguments
    /// * `syscall` - Information about the detected SYSCALL instruction
    /// * `ctx` - Unified context for emitting WebAssembly instructions
    fn call(&mut self, syscall: &SyscallInfo, ctx: &mut CallbackContext<E, F>);
}

/// Blanket implementation of SyscallCallback for FnMut closures
impl<E, G: InstructionSink<E>, F> SyscallCallback<E, G> for F
where
    F: FnMut(&SyscallInfo, &mut CallbackContext<E, G>),
{
    fn call(&mut self, syscall: &SyscallInfo, ctx: &mut CallbackContext<E, G>) {
        self(syscall, ctx)
    }
}

/// Trait for BREAK instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when BREAK
/// instructions are encountered during translation.
pub trait BreakCallback<E, F: InstructionSink<E>> {
    /// Process a BREAK instruction
    ///
    /// # Arguments
    /// * `break_info` - Information about the detected BREAK instruction
    /// * `ctx` - Unified context for emitting WebAssembly instructions
    fn call(&mut self, break_info: &BreakInfo, ctx: &mut CallbackContext<E, F>);
}

/// Blanket implementation of BreakCallback for FnMut closures
impl<E, G: InstructionSink<E>, F> BreakCallback<E, G> for F
where
    F: FnMut(&BreakInfo, &mut CallbackContext<E, G>),
{
    fn call(&mut self, break_info: &BreakInfo, ctx: &mut CallbackContext<E, G>) {
        self(break_info, ctx)
    }
}

/// MIPS to WebAssembly recompiler
///
/// This structure manages the translation of MIPS instructions to WebAssembly,
/// using the yecta reactor for control flow management.
///
/// Each instruction gets its own function, and control flow is managed through
/// jumps between these functions using the yecta reactor.
pub struct MipsRecompiler<'cb, 'ctx, E, F: InstructionSink<E>> {
    reactor: Reactor<E, F>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    /// Base PC address - subtracted from PC values to compute function indices
    base_pc: u32,
    /// Optional callback for SYSCALL instructions
    syscall_callback: Option<&'cb mut (dyn SyscallCallback<E, F> + 'ctx)>,
    /// Optional callback for BREAK instructions
    break_callback: Option<&'cb mut (dyn BreakCallback<E, F> + 'ctx)>,
    /// Optional callback for address mapping (paging support)
    mapper_callback: Option<&'cb mut (dyn MapperCallback<E, F> + 'ctx)>,
    /// Whether to enable MIPS64 instruction support (disabled by default)
    enable_mips64: bool,
}

impl<'cb, 'ctx, E, F: InstructionSink<E>> MipsRecompiler<'cb, 'ctx, E, F> {
    /// Create a new MIPS recompiler instance with full configuration
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    /// * `enable_mips64` - Whether to enable MIPS64 instruction support
    pub fn new_with_full_config(
        pool: Pool,
        escape_tag: Option<EscapeTag>,
        base_pc: u32,
        enable_mips64: bool,
    ) -> Self {
        Self {
            reactor: Reactor::default(),
            pool,
            escape_tag,
            base_pc,
            syscall_callback: None,
            break_callback: None,
            mapper_callback: None,
            enable_mips64,
        }
    }

    /// Create a new MIPS recompiler instance with all configuration options including base function offset
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    /// * `base_func_offset` - Offset added to emitted function indices for imports/helpers
    /// * `enable_mips64` - Whether to enable MIPS64 instruction support
    pub fn new_with_all_config(
        pool: Pool,
        escape_tag: Option<EscapeTag>,
        base_pc: u32,
        base_func_offset: u32,
        enable_mips64: bool,
    ) -> Self {
        Self {
            reactor: Reactor::with_base_func_offset(base_func_offset),
            pool,
            escape_tag,
            base_pc,
            syscall_callback: None,
            break_callback: None,
            mapper_callback: None,
            enable_mips64,
        }
    }

    /// Get the current base function offset.
    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    /// Set the base function offset.
    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }

    /// Create a new MIPS recompiler instance
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    pub fn new_with_config(pool: Pool, escape_tag: Option<EscapeTag>, base_pc: u32) -> Self {
        Self::new_with_full_config(pool, escape_tag, base_pc, false)
    }

    /// Create a new MIPS recompiler with default configuration
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

    /// Create a new MIPS recompiler with a specified base PC
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

    /// Set a callback for SYSCALL instructions
    ///
    /// When a callback is set, it will be invoked immediately when a SYSCALL instruction
    /// is encountered during translation.
    pub fn set_syscall_callback(&mut self, callback: &'cb mut (dyn SyscallCallback<E, F> + 'ctx)) {
        self.syscall_callback = Some(callback);
    }

    /// Clear the SYSCALL callback
    pub fn clear_syscall_callback(&mut self) {
        self.syscall_callback = None;
    }

    /// Set a callback for BREAK instructions
    ///
    /// When a callback is set, it will be invoked immediately when a BREAK instruction
    /// is encountered during translation.
    pub fn set_break_callback(&mut self, callback: &'cb mut (dyn BreakCallback<E, F> + 'ctx)) {
        self.break_callback = Some(callback);
    }

    /// Clear the BREAK callback
    pub fn clear_break_callback(&mut self) {
        self.break_callback = None;
    }

    /// Set an address mapping callback for paging support
    ///
    /// When a mapper is set, it will be invoked for every memory load/store operation
    /// to translate virtual addresses to physical addresses.
    pub fn set_mapper_callback(&mut self, callback: &'cb mut (dyn MapperCallback<E, F> + 'ctx)) {
        self.mapper_callback = Some(callback);
    }

    /// Clear the address mapping callback
    pub fn clear_mapper_callback(&mut self) {
        self.mapper_callback = None;
    }

    /// Enable or disable MIPS64 instruction support
    ///
    /// When enabled, MIPS64-specific instructions will be translated instead of
    /// emitting unreachable.
    pub fn set_mips64_support(&mut self, enable: bool) {
        self.enable_mips64 = enable;
    }

    /// Check if MIPS64 support is enabled
    pub fn is_mips64_enabled(&self) -> bool {
        self.enable_mips64
    }

    /// Convert a PC value to a function index
    /// PC values are offset by base_pc and then divided by 4 for 4-byte alignment
    fn pc_to_func_idx(&self, pc: u32) -> FuncIdx {
        let offset_pc = pc.wrapping_sub(self.base_pc);
        FuncIdx(offset_pc / 4)
    }

    /// Initialize a function for a single instruction at given PC
    ///
    /// Sets up locals for:
    /// - 32 general-purpose registers ($0-$31)
    /// - 2 special registers (HI/LO)
    /// - 1 program counter register
    /// - Additional temporary registers as needed
    ///
    /// # Arguments
    /// * `_pc` - Program counter for this instruction (used for documentation)
    /// * `num_temps` - Number of additional temporary registers needed
    fn init_function(
        &mut self,
        _pc: u32,
        num_temps: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) {
        // General-purpose registers: locals 0-31 (i32 for MIPS32, i64 for MIPS64)
        // HI/LO registers: locals 32-33 (same type as GPRs)
        // PC: local 34 (i32)
        // Temps: locals 35+ (mixed types)
        let gpr_type = if self.enable_mips64 {
            ValType::I64
        } else {
            ValType::I32
        };
        let locals = [
            (32, gpr_type),            // $0-$31
            (2, gpr_type),             // HI/LO
            (1, ValType::I32),         // PC
            (num_temps, gpr_type),     // Temporary registers (match GPR type)
        ];
        // Set to 1 to prevent infinite looping (yecta handles automatic fallthrough)
        self.reactor.next_with(f(&mut locals.into_iter()), 1);
    }

    /// Get the local index for a general-purpose register
    fn gpr_to_local(reg: GprO32) -> u32 {
        reg as u32
    }

    /// Get the local index for the HI register
    const fn hi_local() -> u32 {
        32
    }

    /// Get the local index for the LO register
    const fn lo_local() -> u32 {
        33
    }

    /// Get the local index for the program counter
    const fn pc_local() -> u32 {
        34
    }

    /// Emit an integer constant (i32 or i64 depending on MIPS64 mode)
    fn emit_int_const(&mut self, value: i32) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Const(value as i64))
        } else {
            self.reactor.feed(&WasmInstruction::I32Const(value))
        }
    }

    /// Emit an unsigned integer constant (i32 or i64 depending on MIPS64 mode)
    fn emit_uint_const(&mut self, value: u32) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Const(value as i64))
        } else {
            self.reactor.feed(&WasmInstruction::I32Const(value as i32))
        }
    }

    /// Emit an add instruction (I32Add or I64Add depending on MIPS64 mode)
    fn emit_add(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Add)
        } else {
            self.reactor.feed(&WasmInstruction::I32Add)
        }
    }

    /// Emit a sub instruction (I32Sub or I64Sub depending on MIPS64 mode)
    fn emit_sub(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Sub)
        } else {
            self.reactor.feed(&WasmInstruction::I32Sub)
        }
    }

    /// Emit a multiply instruction (I32Mul or I64Mul depending on MIPS64 mode)
    fn emit_mul(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Mul)
        } else {
            self.reactor.feed(&WasmInstruction::I32Mul)
        }
    }

    /// Emit a logical and instruction (I32And or I64And depending on MIPS64 mode)
    fn emit_and(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64And)
        } else {
            self.reactor.feed(&WasmInstruction::I32And)
        }
    }

    /// Emit a logical or instruction (I32Or or I64Or depending on MIPS64 mode)
    fn emit_or(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Or)
        } else {
            self.reactor.feed(&WasmInstruction::I32Or)
        }
    }

    /// Emit a logical xor instruction (I32Xor or I64Xor depending on MIPS64 mode)
    fn emit_xor(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Xor)
        } else {
            self.reactor.feed(&WasmInstruction::I32Xor)
        }
    }

    /// Emit a shift left instruction (I32Shl or I64Shl depending on MIPS64 mode)
    fn emit_shl(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64Shl)
        } else {
            self.reactor.feed(&WasmInstruction::I32Shl)
        }
    }

    /// Emit a logical shift right instruction (I32ShrU or I64ShrU depending on MIPS64 mode)
    fn emit_shr_u(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64ShrU)
        } else {
            self.reactor.feed(&WasmInstruction::I32ShrU)
        }
    }

    /// Emit an arithmetic shift right instruction (I32ShrS or I64ShrS depending on MIPS64 mode)
    fn emit_shr_s(&mut self) -> Result<(), E> {
        if self.enable_mips64 {
            self.reactor.feed(&WasmInstruction::I64ShrS)
        } else {
            self.reactor.feed(&WasmInstruction::I32ShrS)
        }
    }

    /// Perform a jump to a target PC using yecta's jump API
    fn jump_to_pc(&mut self, target_pc: u32, params: u32) -> Result<(), E> {
        let target_func = self.pc_to_func_idx(target_pc);
        self.reactor.jmp(target_func, params)
    }

    /// Translate a single MIPS instruction to WebAssembly
    ///
    /// This creates a separate function for instruction at given PC and
    /// handles jumps to other instructions using the yecta reactor's jump APIs.
    ///
    /// # Arguments
    /// * `instruction` - The decoded MIPS instruction
    /// * `f` - Function to create the instruction sink
    pub fn translate_instruction(
        &mut self,
        instruction: &Instruction,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        let pc = instruction.vram;

        // Initialize function for this instruction
        self.init_function(pc, 8, f);
        
        // Update PC
        self.reactor.feed(&WasmInstruction::I32Const(pc as i32))?;
        self.reactor.feed(&WasmInstruction::LocalSet(Self::pc_local()))?;

        let opcode = instruction.unique_id;

        match opcode {
            // Arithmetic instructions
            InstrId::cpu_add => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_add()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_addu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_add()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_addi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(imm))?;
                    self.emit_add()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_addiu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(imm))?;
                    self.emit_add()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_sub => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_sub()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_subu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_sub()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            // Logical operations
            InstrId::cpu_and => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_and()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_or => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_or()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_xor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_xor()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_nor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_or()?;
                    self.emit_int_const(-1)?;
                    self.emit_xor()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_andi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(imm as i32))?;
                    self.emit_and()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_ori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(imm as i32))?;
                    self.emit_or()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_xori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(imm as i32))?;
                    self.emit_xor()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Shift operations
            InstrId::cpu_sll => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shl()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_srl => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_u()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_sra => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();
                
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.reactor.feed(&WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_s()?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            // Load upper immediate
            InstrId::cpu_lui => {
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;
                
                if rt != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::I32Const((imm << 16) as i32))?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Jump instructions
            InstrId::cpu_j => {
                // J uses a 26-bit instruction index field (instr_index)
                let target = (instruction.get_instr_index() as u32) << 2;
                let target_pc = (pc & 0xF0000000) | target;
                
                self.jump_to_pc(target_pc, 35)?; // Pass all registers as parameters
                return Ok(()); // J handles control flow, no fallthrough
            }

            InstrId::cpu_jal => {
                // JAL uses the same 26-bit instruction index field
                let target = (instruction.get_instr_index() as u32) << 2;
                let target_pc = (pc & 0xF0000000) | target;
                
                // Save return address in $ra ($31)
                let return_addr = pc + 8; // JAL has a delay slot
                self.reactor.feed(&WasmInstruction::I32Const(return_addr as i32))?;
                self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(GprO32::ra)))?;
                
                self.jump_to_pc(target_pc, 35)?; // Pass all registers as parameters
                return Ok(()); // JAL handles control flow, no fallthrough
            }

            InstrId::cpu_jr => {
                let rs: GprO32 = instruction.get_rs_o32();

                // Use shared TableIndexSnippet to compute table index
                let snippet = TableIndexSnippet {
                    rs_local: Self::gpr_to_local(rs),
                    base_pc: self.base_pc,
                };

                // Use yecta indirect jump params: pass all regs as parameters
                let params = yecta::JumpCallParams::indirect_jump(&snippet, 35, self.pool);
                self.reactor.ji_with_params(params)?;
                return Ok(()); // JR handles control flow, no fallthrough
            }

            InstrId::cpu_jalr => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                // Save return address
                let return_addr = pc + 8; // JALR has a delay slot
                if rd != GprO32::zero {
                    self.reactor.feed(&WasmInstruction::I32Const(return_addr as i32))?;
                    self.reactor.feed(&WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }

                // Use shared TableIndexSnippet to compute table index
                let snippet = TableIndexSnippet {
                    rs_local: Self::gpr_to_local(rs),
                    base_pc: self.base_pc,
                };

                // Use yecta indirect jump params: pass all regs as parameters
                let params = yecta::JumpCallParams::indirect_jump(&snippet, 35, self.pool);
                self.reactor.ji_with_params(params)?;
                return Ok(()); // JALR handles control flow, no fallthrough
            }

            // System instructions
            InstrId::cpu_syscall => {
                let syscall_info = SyscallInfo {
                    pc,
                    syscall_number: 0, // Would need to extract from $v0
                };

                // Invoke callback if set
                if let Some(ref mut callback) = self.syscall_callback {
                    let mut ctx = CallbackContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&syscall_info, &mut ctx);
                } else {
                    // Default behavior: system call - implementation specific
                    self.reactor.feed(&WasmInstruction::Unreachable)?;
                }
            }

            InstrId::cpu_break => {
                let code = instruction.get_code() as u32;
                let break_info = BreakInfo { pc, code };

                // Invoke callback if set
                if let Some(ref mut callback) = self.break_callback {
                    let mut ctx = CallbackContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&break_info, &mut ctx);
                } else {
                    // Default behavior: breakpoint - implementation specific
                    self.reactor.feed(&WasmInstruction::Unreachable)?;
                }
            }

            // Unsupported or unimplemented instructions
            _ => {
                // Emit unreachable for unsupported instructions
                self.reactor.feed(&WasmInstruction::Unreachable)?;
            }
        }

        Ok(())
    }
}

