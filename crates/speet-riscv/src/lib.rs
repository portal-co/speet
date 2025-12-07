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
use wax_core::build::InstructionSink;

use core::convert::Infallible;
use rv_asm::{FReg, Imm, Inst, IsCompressed, Reg, Xlen};
use wasm_encoder::{Function, Instruction, ValType};
use yecta::{EscapeTag, FuncIdx, Pool, Reactor, TableIdx, TypeIdx};

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
pub struct HintContext<'a, E, F: InstructionSink<E>> {
    /// Reference to the reactor for emitting WebAssembly instructions
    pub reactor: &'a mut Reactor<E, F>,
}

impl<'a, E, F: InstructionSink<E>> HintContext<'a, E, F> {
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
    pub fn emit(&mut self, instruction: &Instruction) -> Result<(), E> {
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
pub trait HintCallback<E, F: InstructionSink<E>> {
    /// Process a HINT instruction
    ///
    /// # Arguments
    /// * `hint` - Information about the detected HINT instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, hint: &HintInfo, ctx: &mut HintContext<E, F>);
}

/// Blanket implementation of HintCallback for FnMut closures
impl<E, G: InstructionSink<E>, F> HintCallback<E, G> for F
where
    F: FnMut(&HintInfo, &mut HintContext<E, G>),
{
    fn call(&mut self, hint: &HintInfo, ctx: &mut HintContext<E, G>) {
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
pub trait EcallCallback<E, F: InstructionSink<E>> {
    /// Process an ECALL instruction
    ///
    /// # Arguments
    /// * `ecall` - Information about the detected ECALL instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, ecall: &EcallInfo, ctx: &mut HintContext<E, F>);
}

/// Blanket implementation of EcallCallback for FnMut closures
impl<E, G: InstructionSink<E>, F> EcallCallback<E, G> for F
where
    F: FnMut(&EcallInfo, &mut HintContext<E, G>),
{
    fn call(&mut self, ecall: &EcallInfo, ctx: &mut HintContext<E, G>) {
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
pub trait EbreakCallback<E, F: InstructionSink<E>> {
    /// Process an EBREAK instruction
    ///
    /// # Arguments
    /// * `ebreak` - Information about the detected EBREAK instruction
    /// * `ctx` - Context for emitting WebAssembly instructions
    fn call(&mut self, ebreak: &EbreakInfo, ctx: &mut HintContext<E, F>);
}

/// Blanket implementation of EbreakCallback for FnMut closures
impl<E, G: InstructionSink<E>, F> EbreakCallback<E, G> for F
where
    F: FnMut(&EbreakInfo, &mut HintContext<E, G>),
{
    fn call(&mut self, ebreak: &EbreakInfo, ctx: &mut HintContext<E, G>) {
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
pub struct RiscVRecompiler<'cb, 'ctx, E, F: InstructionSink<E>> {
    reactor: Reactor<E, F>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,
    /// Base PC address - subtracted from PC values to compute function indices
    /// For RV64, this is a 64-bit address
    base_pc: u64,
    /// Whether to track HINT instructions (disabled by default)
    track_hints: bool,
    /// Collected HINT instructions (when tracking is enabled)
    hints: Vec<HintInfo>,
    /// Optional callback for inline HINT processing
    hint_callback: Option<&'cb mut (dyn HintCallback<E, F> + 'ctx)>,
    /// Optional callback for ECALL instructions
    ecall_callback: Option<&'cb mut (dyn EcallCallback<E, F> + 'ctx)>,
    /// Optional callback for EBREAK instructions
    ebreak_callback: Option<&'cb mut (dyn EbreakCallback<E, F> + 'ctx)>,
    /// Whether to enable RV64 instruction support (disabled by default)
    enable_rv64: bool,
    /// Whether to use memory64 (i64 addresses) instead of memory32 (i32 addresses)
    /// Only relevant when enable_rv64 is true
    use_memory64: bool,
}

impl<'cb, 'ctx, E, F: InstructionSink<E>> RiscVRecompiler<'cb, 'ctx, E, F> {
    /// Create a new RISC-V recompiler instance with full configuration
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices (64-bit for RV64 support)
    /// * `track_hints` - Whether to track HINT instructions for debugging/testing
    /// * `enable_rv64` - Whether to enable RV64 instruction support
    /// * `use_memory64` - Whether to use memory64 (i64 addresses) for memory operations
    pub fn new_with_full_config(
        pool: Pool,
        escape_tag: Option<EscapeTag>,
        base_pc: u64,
        track_hints: bool,
        enable_rv64: bool,
        use_memory64: bool,
    ) -> Self {
        Self {
            reactor: Reactor::default(),
            pool,
            escape_tag,
            base_pc,
            track_hints,
            hints: Vec::new(),
            hint_callback: None,
            ecall_callback: None,
            ebreak_callback: None,
            enable_rv64,
            use_memory64,
        }
    }

    /// Create a new RISC-V recompiler instance with all configuration options including base function offset
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices (64-bit for RV64 support)
    /// * `base_func_offset` - Offset added to emitted function indices for imports/helpers
    /// * `track_hints` - Whether to track HINT instructions for debugging/testing
    /// * `enable_rv64` - Whether to enable RV64 instruction support
    /// * `use_memory64` - Whether to use memory64 (i64 addresses) for memory operations
    pub fn new_with_all_config(
        pool: Pool,
        escape_tag: Option<EscapeTag>,
        base_pc: u64,
        base_func_offset: u32,
        track_hints: bool,
        enable_rv64: bool,
        use_memory64: bool,
    ) -> Self {
        Self {
            reactor: Reactor::with_base_func_offset(base_func_offset),
            pool,
            escape_tag,
            base_pc,
            track_hints,
            hints: Vec::new(),
            hint_callback: None,
            ecall_callback: None,
            ebreak_callback: None,
            enable_rv64,
            use_memory64,
        }
    }

    /// Get the current base function offset.
    ///
    /// The offset is added to all emitted function indices. This is useful when
    /// the WebAssembly module has imported functions or helper functions that
    /// precede the generated functions.
    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    /// Set the base function offset.
    ///
    /// The offset is added to all emitted function indices. This is useful when
    /// the WebAssembly module has imported functions or helper functions that
    /// precede the generated functions.
    ///
    /// # Arguments
    /// * `offset` - Offset added to function indices in emitted instructions
    ///
    /// # Example
    /// If the module has 10 imports and 5 helper functions, use `offset = 15`
    /// so that generated function 0 emits as WebAssembly function 15.
    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }

    /// Create a new RISC-V recompiler instance
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices (64-bit for RV64 support)
    pub fn new_with_config(pool: Pool, escape_tag: Option<EscapeTag>, base_pc: u64) -> Self {
        Self::new_with_full_config(pool, escape_tag, base_pc, false, false, false)
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
    /// * `base_pc` - Base PC address - this is subtracted from instruction PCs to compute function indices (64-bit for RV64 support)
    pub fn new_with_base_pc(base_pc: u64) -> Self {
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
    pub fn set_hint_callback(&mut self, callback: &'cb mut (dyn HintCallback<E, F> + 'ctx)) {
        self.hint_callback = Some(callback);
    }

    /// Clear the HINT callback
    ///
    /// Removes any previously set HINT callback.
    pub fn clear_hint_callback(&mut self) {
        self.hint_callback = None;
    }

    /// Set a callback for ECALL instructions
    ///
    /// When a callback is set, it will be invoked immediately when an ECALL instruction
    /// is encountered during translation. This allows for custom handling of environment
    /// calls, including generating custom WebAssembly code.
    ///
    /// # Arguments
    /// * `callback` - A mutable reference to a closure or function that takes an `&EcallInfo` and `&mut HintContext`
    ///
    /// # Example
    /// ```no_run
    /// # use speet_riscv::{RiscVRecompiler, EcallInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    /// let mut recompiler = RiscVRecompiler::new();
    /// let mut my_callback = |ecall: &EcallInfo, ctx: &mut HintContext| {
    ///     println!("ECALL at PC 0x{:x}", ecall.pc);
    ///     // Optionally emit WebAssembly instructions for the ecall
    ///     ctx.emit(&Instruction::Nop).ok();
    /// };
    /// recompiler.set_ecall_callback(&mut my_callback);
    /// ```
    pub fn set_ecall_callback(&mut self, callback: &'cb mut (dyn EcallCallback<E, F> + 'ctx)) {
        self.ecall_callback = Some(callback);
    }

    /// Clear the ECALL callback
    ///
    /// Removes any previously set ECALL callback.
    pub fn clear_ecall_callback(&mut self) {
        self.ecall_callback = None;
    }

    /// Set a callback for EBREAK instructions
    ///
    /// When a callback is set, it will be invoked immediately when an EBREAK instruction
    /// is encountered during translation. This allows for custom handling of breakpoints,
    /// including generating custom WebAssembly code.
    ///
    /// # Arguments
    /// * `callback` - A mutable reference to a closure or function that takes an `&EbreakInfo` and `&mut HintContext`
    ///
    /// # Example
    /// ```no_run
    /// # use speet_riscv::{RiscVRecompiler, EbreakInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    /// let mut recompiler = RiscVRecompiler::new();
    /// let mut my_callback = |ebreak: &EbreakInfo, ctx: &mut HintContext| {
    ///     println!("EBREAK at PC 0x{:x}", ebreak.pc);
    ///     // Optionally emit WebAssembly instructions for the ebreak
    ///     ctx.emit(&Instruction::Nop).ok();
    /// };
    /// recompiler.set_ebreak_callback(&mut my_callback);
    /// ```
    pub fn set_ebreak_callback(&mut self, callback: &'cb mut (dyn EbreakCallback<E, F> + 'ctx)) {
        self.ebreak_callback = Some(callback);
    }

    /// Clear the EBREAK callback
    ///
    /// Removes any previously set EBREAK callback.
    pub fn clear_ebreak_callback(&mut self) {
        self.ebreak_callback = None;
    }

    /// Enable or disable RV64 instruction support
    ///
    /// When enabled, RV64-specific instructions will be translated instead of
    /// emitting unreachable. This includes 64-bit loads/stores and W-suffix
    /// instructions (ADDIW, ADDW, etc.).
    ///
    /// # Arguments
    /// * `enable` - Whether to enable RV64 support
    pub fn set_rv64_support(&mut self, enable: bool) {
        self.enable_rv64 = enable;
    }

    /// Check if RV64 support is enabled
    pub fn is_rv64_enabled(&self) -> bool {
        self.enable_rv64
    }

    /// Enable or disable memory64 mode
    ///
    /// When enabled (and RV64 is enabled), memory operations will use i64
    /// addresses instead of i32 addresses. This is required for accessing
    /// memory beyond 4GB in WebAssembly.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable memory64 mode
    pub fn set_memory64(&mut self, enable: bool) {
        self.use_memory64 = enable;
    }

    /// Check if memory64 mode is enabled
    pub fn is_memory64_enabled(&self) -> bool {
        self.use_memory64
    }

    /// Convert a PC value to a function index
    /// PC values are offset by base_pc and then divided by 2 for 2-byte alignment
    /// For RV64, accepts 64-bit PC values
    fn pc_to_func_idx(&self, pc: u64) -> FuncIdx {
        let offset_pc = pc.wrapping_sub(self.base_pc);
        FuncIdx((offset_pc / 2) as u32)
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
    fn init_function(
        &mut self,
        _pc: u32,
        _inst_len: u32,
        num_temps: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) {
        // Integer registers: locals 0-31 (i32 for RV32, i64 for RV64)
        // Float registers: locals 32-63 (f64)
        // PC: local 64 (i32)
        // Temps: locals 65+ (mixed types)
        let int_type = if self.enable_rv64 {
            ValType::I64
        } else {
            ValType::I32
        };
        let locals = [
            (32, int_type),            // x0-x31
            (32, ValType::F64),        // f0-f31 (using F64 for both F and D with NaN-boxing)
            (1, ValType::I32),         // PC
            (num_temps, int_type),     // Temporary registers (match integer register type)
        ];
        // The second argument is the length in 2-byte increments
        // Set to 2 to prevent infinite looping (yecta handles automatic fallthrough)
        self.reactor.next_with(f(&mut locals.into_iter()), 2);
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
    fn emit_imm(&mut self, imm: Imm) -> Result<(), E> {
        if self.enable_rv64 {
            // Sign-extend the 32-bit immediate to 64 bits
            self.reactor.feed(&Instruction::I64Const(imm.as_i32() as i64))
        } else {
            self.reactor.feed(&Instruction::I32Const(imm.as_i32()))
        }
    }

    /// Emit an integer constant (i32 or i64 depending on RV64 mode)
    fn emit_int_const(&mut self, value: i32) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Const(value as i64))
        } else {
            self.reactor.feed(&Instruction::I32Const(value))
        }
    }

    /// Emit an add instruction (I32Add or I64Add depending on RV64 mode)
    fn emit_add(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Add)
        } else {
            self.reactor.feed(&Instruction::I32Add)
        }
    }

    /// Emit a sub instruction (I32Sub or I64Sub depending on RV64 mode)
    fn emit_sub(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Sub)
        } else {
            self.reactor.feed(&Instruction::I32Sub)
        }
    }

    /// Emit a multiply instruction (I32Mul or I64Mul depending on RV64 mode)
    fn emit_mul(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Mul)
        } else {
            self.reactor.feed(&Instruction::I32Mul)
        }
    }

    /// Emit a logical and instruction (I32And or I64And depending on RV64 mode)
    fn emit_and(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64And)
        } else {
            self.reactor.feed(&Instruction::I32And)
        }
    }

    /// Emit a logical or instruction (I32Or or I64Or depending on RV64 mode)
    fn emit_or(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Or)
        } else {
            self.reactor.feed(&Instruction::I32Or)
        }
    }

    /// Emit a logical xor instruction (I32Xor or I64Xor depending on RV64 mode)
    fn emit_xor(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Xor)
        } else {
            self.reactor.feed(&Instruction::I32Xor)
        }
    }

    /// Emit a shift left instruction (I32Shl or I64Shl depending on RV64 mode)
    fn emit_shl(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Shl)
        } else {
            self.reactor.feed(&Instruction::I32Shl)
        }
    }

    /// Emit a logical shift right instruction (I32ShrU or I64ShrU depending on RV64 mode)
    fn emit_shr_u(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64ShrU)
        } else {
            self.reactor.feed(&Instruction::I32ShrU)
        }
    }

    /// Emit an arithmetic shift right instruction (I32ShrS or I64ShrS depending on RV64 mode)
    fn emit_shr_s(&mut self) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64ShrS)
        } else {
            self.reactor.feed(&Instruction::I32ShrS)
        }
    }

    /// Perform a jump to a target PC using yecta's jump API
    fn jump_to_pc(&mut self, target_pc: u64, params: u32) -> Result<(), E> {
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
    fn nan_box_f32(&mut self) -> Result<(), E> {
        // Convert F32 to I32, then to I64, OR with 0xFFFFFFFF00000000, reinterpret as F64
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;
        self.reactor.feed(&Instruction::I64ExtendI32U)?;
        self.reactor
            .feed(&Instruction::I64Const(0xFFFFFFFF00000000_u64 as i64))?;
        self.reactor.feed(&Instruction::I64Or)?;
        self.reactor.feed(&Instruction::F64ReinterpretI64)?;
        Ok(())
    }

    /// Unbox a NaN-boxed single-precision value from a double-precision register
    ///
    /// Extract the F32 value from the lower 32 bits of the NaN-boxed F64 value
    fn unbox_f32(&mut self) -> Result<(), E> {
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
    pub fn translate_instruction(
        &mut self,
        inst: &Inst,
        pc: u32,
        is_compressed: IsCompressed,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        // Calculate instruction length in 2-byte increments
        let inst_len = match is_compressed {
            IsCompressed::Yes => 1, // 2 bytes
            IsCompressed::No => 2,  // 4 bytes
        };

        // Initialize function for this instruction
        self.init_function(pc, inst_len, 8, f);
        // Update PC
        self.reactor.feed(&Instruction::I32Const(pc as i32))?;
        self.reactor
            .feed(&Instruction::LocalSet(Self::pc_local()))?;

        match inst {
            // RV32I Base Integer Instruction Set

            // Lui: Load Upper Immediate
            // RISC-V Specification Quote:
            // "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format.
            // LUI places the 32-bit U-immediate value into the destination register rd, filling in the
            // lowest 12 bits with zeros."
            Inst::Lui { uimm, dest } => {
                self.emit_imm(*uimm)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
            }

            // Auipc: Add Upper Immediate to PC
            // RISC-V Specification Quote:
            // "AUIPC (add upper immediate to pc) is used to build pc-relative addresses and uses the
            // U-type format. AUIPC forms a 32-bit offset from the U-immediate, filling in the lowest
            // 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places
            // the result in register rd."
            Inst::Auipc { uimm, dest } => {
                if self.enable_rv64 {
                    // RV64: Use full 64-bit PC
                    self.reactor.feed(&Instruction::I64Const(pc as i64))?;
                    self.emit_imm(*uimm)?;
                    self.emit_add()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else {
                    // RV32: Use 32-bit PC
                    self.reactor.feed(&Instruction::I32Const(pc as i32))?;
                    self.emit_imm(*uimm)?;
                    self.reactor.feed(&Instruction::I32Add)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Jal: Jump And Link
            // RISC-V Specification Quote:
            // "The jump and link (JAL) instruction uses the J-type format, where the J-immediate encodes
            // a signed offset in multiples of 2 bytes. The offset is sign-extended and added to the
            // address of the jump instruction to form the jump target address."
            Inst::Jal { offset, dest } => {
                // Save return address (PC + inst_len * 2) to dest
                if dest.0 != 0 {
                    // x0 is hardwired to zero
                    let return_addr = pc as u64 + (inst_len * 2) as u64;
                    if self.enable_rv64 {
                        // RV64: Store full 64-bit return address
                        self.reactor
                            .feed(&Instruction::I64Const(return_addr as i64))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // RV32: Store 32-bit return address
                        self.reactor
                            .feed(&Instruction::I32Const(return_addr as i32))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
                // Jump to PC + offset using yecta's jump API with PC-based indexing
                let target_pc = if self.enable_rv64 {
                    // RV64: Use 64-bit PC arithmetic
                    (pc as i64).wrapping_add(offset.as_i32() as i64) as u64
                } else {
                    // RV32: Use 32-bit PC arithmetic  
                    (pc as i32).wrapping_add(offset.as_i32()) as u32 as u64
                };
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
                    let return_addr = pc as u64 + (inst_len * 2) as u64;
                    if self.enable_rv64 {
                        // RV64: Store full 64-bit return address
                        self.reactor
                            .feed(&Instruction::I64Const(return_addr as i64))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // RV32: Store 32-bit return address
                        self.reactor
                            .feed(&Instruction::I32Const(return_addr as i32))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
                // JALR is indirect, so we need to compute the target dynamically
                // For now, we'll use the computed target and update PC
                // A full implementation would need dynamic dispatch through a table
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*base)))?;
                self.emit_imm(*offset)?;
                self.emit_add()?;
                if self.enable_rv64 {
                    // RV64: Clear LSB with 64-bit mask
                    self.reactor
                        .feed(&Instruction::I64Const(0xFFFFFFFFFFFFFFFE_u64 as i64))?; // ~1 mask
                    self.reactor.feed(&Instruction::I64And)?;
                } else {
                    // RV32: Clear LSB with 32-bit mask
                    self.reactor
                        .feed(&Instruction::I32Const(0xFFFFFFFE_u32 as i32))?; // ~1 mask
                    self.reactor.feed(&Instruction::I32And)?;
                }
                self.reactor
                    .feed(&Instruction::LocalSet(Self::pc_local()))?;
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
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                } else if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_add()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slti { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32LtS)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltiu { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.reactor.feed(&Instruction::I32LtU)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_xor()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Ori { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_or()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Andi { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_and()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_shl()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srli { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_shr_u()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srai { imm, dest, src1 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.emit_imm(*imm)?;
                    self.emit_shr_s()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Register-Register Operations
            Inst::Add { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_add()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sub { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_sub()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sll { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shl()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Slt { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32LtS)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sltu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32LtU)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Xor { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_xor()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Srl { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_u()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Sra { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_shr_s()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Or { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_or()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::And { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_and()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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
                let ecall_info = EcallInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ecall_callback {
                    let mut ctx = HintContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&ecall_info, &mut ctx);
                } else {
                    // Default behavior: environment call - implementation specific
                    // Would need to be handled by runtime
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::Ebreak => {
                let ebreak_info = EbreakInfo { pc };

                // Invoke callback if set
                if let Some(ref mut callback) = self.ebreak_callback {
                    let mut ctx = HintContext {
                        reactor: &mut self.reactor,
                    };
                    callback.call(&ebreak_info, &mut ctx);
                } else {
                    // Default behavior: breakpoint - implementation specific
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            // M Extension: Integer Multiplication and Division
            // RISC-V Specification Quote:
            // "This chapter describes the standard integer multiplication and division instruction-set
            // extension, which is named 'M' and contains instructions that multiply or divide values
            // held in two integer registers."
            Inst::Mul { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.emit_mul()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Mulh { dest, src1, src2 } => {
                // Multiply high signed-signed: returns upper bits of product
                // RV32: upper 32 bits of 64-bit product
                // RV64: upper 64 bits of 128-bit product
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit signed multiplication
                        self.emit_mulh_signed(Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor.feed(&Instruction::I64Mul)?;
                        self.reactor.feed(&Instruction::I64Const(32))?;
                        self.reactor.feed(&Instruction::I64ShrS)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Mulhsu { dest, src1, src2 } => {
                // Multiply high signed-unsigned
                // RV32: upper 32 bits of 64-bit product (src1 signed, src2 unsigned)
                // RV64: upper 64 bits of 128-bit product (src1 signed, src2 unsigned)
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit signed-unsigned multiplication
                        self.emit_mulh_signed_unsigned(Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply with mixed sign extension
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32U)?;
                        self.reactor.feed(&Instruction::I64Mul)?;
                        self.reactor.feed(&Instruction::I64Const(32))?;
                        self.reactor.feed(&Instruction::I64ShrS)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Mulhu { dest, src1, src2 } => {
                // Multiply high unsigned-unsigned
                // RV32: upper 32 bits of 64-bit product
                // RV64: upper 64 bits of 128-bit product
                if dest.0 != 0 {
                    if self.enable_rv64 {
                        // For RV64: compute high 64 bits of 128-bit unsigned multiplication
                        self.emit_mulh_unsigned(Self::reg_to_local(*src1), Self::reg_to_local(*src2))?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    } else {
                        // For RV32: use i64 multiply and shift
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32U)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64ExtendI32U)?;
                        self.reactor.feed(&Instruction::I64Mul)?;
                        self.reactor.feed(&Instruction::I64Const(32))?;
                        self.reactor.feed(&Instruction::I64ShrU)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                }
            }

            Inst::Div { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32DivS)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Divu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32DivU)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Rem { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32RemS)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::Remu { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::I32RemU)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
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

            Inst::FaddS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivS {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Div)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtS { dest, src, .. } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sqrt)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-Point Double-Precision (D Extension)
            Inst::Fld { offset, dest, base } => {
                self.translate_fload(*base, *offset, *dest, FLoadOp::F64)?;
            }

            Inst::Fsd { offset, src, base } => {
                self.translate_fstore(*base, *offset, *src, FStoreOp::F64)?;
            }

            Inst::FaddD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Add)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsubD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmulD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FdivD {
                dest, src1, src2, ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Div)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FsqrtD { dest, src, .. } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64Sqrt)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point min/max operations
            Inst::FminS { dest, src1, src2 } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Min)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxS { dest, src1, src2 } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Max)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FminD { dest, src1, src2 } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Min)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaxD { dest, src1, src2 } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Max)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point comparison operations
            Inst::FeqS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32()?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::F32Eq)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32()?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::F32Lt)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleS { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.unbox_f32()?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::F32Le)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FeqD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Eq)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FltD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Lt)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FleD { dest, src1, src2 } => {
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                    self.reactor.feed(&Instruction::F64Le)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // Floating-point conversion operations
            // RISC-V Specification Quote:
            // "Floating-point-to-integer and integer-to-floating-point conversion instructions
            // are encoded in the OP-FP major opcode space."
            Inst::FcvtWS { dest, src, .. } => {
                // Convert single to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32TruncF32S)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuS { dest, src, .. } => {
                // Convert single to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32TruncF32U)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtSW { dest, src, .. } => {
                // Convert signed 32-bit integer to single
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32S)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to single
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ConvertI32U)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtWD { dest, src, .. } => {
                // Convert double to signed 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::I32TruncF64S)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtWuD { dest, src, .. } => {
                // Convert double to unsigned 32-bit integer
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::I32TruncF64U)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FcvtDW { dest, src, .. } => {
                // Convert signed 32-bit integer to double
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64ConvertI32S)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDWu { dest, src, .. } => {
                // Convert unsigned 32-bit integer to double
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F64ConvertI32U)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtSD { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.S.D converts double-precision float to single-precision float,
                // rounding according to the dynamic rounding mode."
                // Convert double to single with proper NaN-boxing
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FcvtDS { dest, src, .. } => {
                // RISC-V Specification Quote:
                // "FCVT.D.S converts single-precision float to double-precision float."
                // Unbox the NaN-boxed single value, then promote to double
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            // Floating-point move operations
            Inst::FmvXW { dest, src } => {
                // Move bits from float register to integer register
                if dest.0 != 0 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                    self.unbox_f32()?;
                    self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::FmvWX { dest, src } => {
                // Move bits from integer register to float register
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor.feed(&Instruction::F32ReinterpretI32)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
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
            Inst::FmaddS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = (src1 * src2) + src3
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Add)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = (src1 * src2) - src3
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) + src3 = src3 - (src1 * src2)
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddS {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                // dest = -(src1 * src2) - src3
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.unbox_f32()?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Mul)?;
                self.reactor.feed(&Instruction::F32Neg)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::F32Sub)?;
                self.nan_box_f32()?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Add)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmsubD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
            }

            Inst::FnmaddD {
                dest,
                src1,
                src2,
                src3,
                ..
            } => {
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src1)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src2)))?;
                self.reactor.feed(&Instruction::F64Mul)?;
                self.reactor.feed(&Instruction::F64Neg)?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(*src3)))?;
                self.reactor.feed(&Instruction::F64Sub)?;
                self.reactor
                    .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
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
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                    self.reactor
                        .feed(&Instruction::I32AtomicLoad(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            Inst::ScW {
                dest, addr, src, ..
            } => {
                // Store-conditional word
                // In a simplified model, always succeed (return 0)
                // A full implementation would track reservations
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*addr)))?;
                self.reactor
                    .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                self.reactor
                    .feed(&Instruction::I32AtomicStore(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(0))?; // Success
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // CSR Instructions (Zicsr extension)
            // RISC-V Specification Quote:
            // "The SYSTEM major opcode is used to encode all privileged instructions, as well
            // as the ECALL and EBREAK instructions and CSR instructions."
            // Note: CSR operations are system-specific and may need special runtime support
            Inst::Csrrw { dest, src, .. }
            | Inst::Csrrs { dest, src, .. }
            | Inst::Csrrc { dest, src, .. } => {
                // For now, we'll stub these out as they require system support
                // A real implementation would need to call into a CSR handler
                if dest.0 != 0 {
                    // Return zero as placeholder
                    self.reactor.feed(&Instruction::I32Const(0))?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
                // Silently ignore the write for now
                let _ = src;
            }

            Inst::Csrrwi { dest, .. } | Inst::Csrrsi { dest, .. } | Inst::Csrrci { dest, .. } => {
                if dest.0 != 0 {
                    self.reactor.feed(&Instruction::I32Const(0))?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                }
            }

            // RV64 instructions
            // These are RV64-specific. When RV64 is disabled, we emit unreachable.
            Inst::Lwu { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(*base, *offset, *dest, LoadOp::U32)?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::Ld { offset, dest, base } => {
                if self.enable_rv64 {
                    self.translate_load(*base, *offset, *dest, LoadOp::I64)?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::Sd { offset, base, src } => {
                if self.enable_rv64 {
                    self.translate_store(*base, *offset, *src, StoreOp::I64)?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            // RV64I: Word arithmetic instructions (operate on lower 32 bits)
            Inst::AddiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.emit_imm(*imm)?;
                        self.reactor.feed(&Instruction::I64Add)?;
                        // Sign-extend lower 32 bits to 64 bits
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SlliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(&Instruction::I32Shl)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SrliW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(&Instruction::I32ShrU)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SraiW { imm, dest, src1 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32Const(imm.as_i32()))?;
                        self.reactor.feed(&Instruction::I32ShrS)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::AddW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64Add)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SubW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64Sub)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SllW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32Shl)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SrlW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32ShrU)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::SraW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32ShrS)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            // RV64M: Multiplication and division word instructions
            Inst::MulW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I64Mul)?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::DivW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32DivS)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::DivuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32DivU)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::RemW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32RemS)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            Inst::RemuW { dest, src1, src2 } => {
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src1)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::reg_to_local(*src2)))?;
                        self.reactor.feed(&Instruction::I32WrapI64)?;
                        self.reactor.feed(&Instruction::I32RemU)?;
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }
            
            // RV64F/D: Floating-point conversion instructions
            Inst::FcvtLS { dest, src, .. } => {
                // Convert single-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32()?;
                        self.reactor.feed(&Instruction::I64TruncF32S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuS { dest, src, .. } => {
                // Convert single-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.unbox_f32()?;
                        self.reactor.feed(&Instruction::I64TruncF32U)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSL { dest, src, .. } => {
                // Convert signed 64-bit integer to single-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F32ConvertI64S)?;
                    self.nan_box_f32()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtSLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to single-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F32ConvertI64U)?;
                    self.nan_box_f32()?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLD { dest, src, .. } => {
                // Convert double-precision float to signed 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(&Instruction::I64TruncF64S)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtLuD { dest, src, .. } => {
                // Convert double-precision float to unsigned 64-bit integer
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(&Instruction::I64TruncF64U)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDL { dest, src, .. } => {
                // Convert signed 64-bit integer to double-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F64ConvertI64S)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FcvtDLu { dest, src, .. } => {
                // Convert unsigned 64-bit integer to double-precision float
                if self.enable_rv64 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F64ConvertI64U)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FmvXD { dest, src } => {
                // Move bits from double-precision float register to 64-bit integer register
                if self.enable_rv64 {
                    if dest.0 != 0 {
                        self.reactor
                            .feed(&Instruction::LocalGet(Self::freg_to_local(*src)))?;
                        self.reactor.feed(&Instruction::I64ReinterpretF64)?;
                        self.reactor
                            .feed(&Instruction::LocalSet(Self::reg_to_local(*dest)))?;
                    }
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
            }

            Inst::FmvDX { dest, src } => {
                // Move bits from 64-bit integer register to double-precision float register
                if self.enable_rv64 {
                    self.reactor
                        .feed(&Instruction::LocalGet(Self::reg_to_local(*src)))?;
                    self.reactor.feed(&Instruction::F64ReinterpretI64)?;
                    self.reactor
                        .feed(&Instruction::LocalSet(Self::freg_to_local(*dest)))?;
                } else {
                    self.reactor.feed(&Instruction::Unreachable)?;
                }
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
            Inst::FclassS { .. } | Inst::FclassD { .. } => {
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
    ) -> Result<(), E> {
        let target_pc = if self.enable_rv64 {
            // RV64: Use 64-bit PC arithmetic
            (pc as i64).wrapping_add(offset.as_i32() as i64) as u64
        } else {
            // RV32: Use 32-bit PC arithmetic
            (pc as i32).wrapping_add(offset.as_i32()) as u32 as u64
        };

        // Create a custom Snippet for the branch condition using a closure
        // The closure captures the registers and operation
        struct BranchCondition {
            src1: u32,
            src2: u32,
            op: BranchOp,
            enable_rv64: bool,
        }

        impl<E> wax_core::build::InstructionOperatorSource<E> for BranchCondition {
            fn emit(
                &self,
                sink: &mut (dyn wax_core::build::InstructionOperatorSink<E> + '_),
            ) -> Result<(), E> {
                // Emit the same instructions as emit_instruction
                sink.instruction(&Instruction::LocalGet(self.src1))?;
                sink.instruction(&Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(&match self.op {
                        BranchOp::Eq => Instruction::I64Eq,
                        BranchOp::Ne => Instruction::I64Ne,
                        BranchOp::LtS => Instruction::I64LtS,
                        BranchOp::GeS => Instruction::I64GeS,
                        BranchOp::LtU => Instruction::I64LtU,
                        BranchOp::GeU => Instruction::I64GeU,
                    })?;
                } else {
                    sink.instruction(&match self.op {
                        BranchOp::Eq => Instruction::I32Eq,
                        BranchOp::Ne => Instruction::I32Ne,
                        BranchOp::LtS => Instruction::I32LtS,
                        BranchOp::GeS => Instruction::I32GeS,
                        BranchOp::LtU => Instruction::I32LtU,
                        BranchOp::GeU => Instruction::I32GeU,
                    })?;
                }
                Ok(())
            }
        }

        impl<E> wax_core::build::InstructionSource<E> for BranchCondition {
            fn emit_instruction(
                &self,
                sink: &mut (dyn wax_core::build::InstructionSink<E> + '_),
            ) -> Result<(), E> {
                sink.instruction(&Instruction::LocalGet(self.src1))?;
                sink.instruction(&Instruction::LocalGet(self.src2))?;
                if self.enable_rv64 {
                    sink.instruction(&match self.op {
                        BranchOp::Eq => Instruction::I64Eq,
                        BranchOp::Ne => Instruction::I64Ne,
                        BranchOp::LtS => Instruction::I64LtS,
                        BranchOp::GeS => Instruction::I64GeS,
                        BranchOp::LtU => Instruction::I64LtU,
                        BranchOp::GeU => Instruction::I64GeU,
                    })?;
                } else {
                    sink.instruction(&match self.op {
                        BranchOp::Eq => Instruction::I32Eq,
                        BranchOp::Ne => Instruction::I32Ne,
                        BranchOp::LtS => Instruction::I32LtS,
                        BranchOp::GeS => Instruction::I32GeS,
                        BranchOp::LtU => Instruction::I32LtU,
                        BranchOp::GeU => Instruction::I32GeU,
                    })?;
                }
                Ok(())
            }
        }

        let condition = BranchCondition {
            src1: Self::reg_to_local(src1),
            src2: Self::reg_to_local(src2),
            op,
            enable_rv64: self.enable_rv64,
        };

        // Use ji with condition for branch taken path
        // When condition is true, jump to target; yecta handles else/end automatically
        let target_func = self.pc_to_func_idx(target_pc);
        let target = yecta::Target::Static { func: target_func };

        self.reactor.ji(
            65,               // params: pass all registers
            &BTreeMap::new(), // fixups: none needed
            target,           // target: branch target
            None,             // call: not an escape call
            self.pool,        // pool: for indirect calls
            Some(&condition), // condition: branch condition
        )?;

        Ok(())
    }

    /// Helper to compute high 64 bits of signed 64x64 -> 128-bit multiplication
    /// 
    /// Algorithm: For two 64-bit signed numbers a and b, we compute the high 64 bits
    /// of their 128-bit product using the formula:
    /// 
    /// Let a = a_hi * 2^32 + a_lo and b = b_hi * 2^32 + b_lo
    /// Then a * b = (a_hi * b_hi * 2^64) + (a_hi * b_lo * 2^32) + (a_lo * b_hi * 2^32) + (a_lo * b_lo)
    /// 
    /// The high 64 bits are:
    /// - a_hi * b_hi (full result)
    /// - high 32 bits of (a_hi * b_lo)
    /// - high 32 bits of (a_lo * b_hi)  
    /// - carries from the middle terms
    ///
    /// Note: a_lo * b_lo produces at most 64 bits, so it doesn't directly contribute
    /// to the high 64 bits, only through carries.
    fn emit_mulh_signed(&mut self, src1: u32, src2: u32) -> Result<(), E> {
        // Load src1 and src2 to locals for reuse
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67; // for accumulating middle terms
        
        self.reactor.feed(&Instruction::LocalGet(src1))?;
        self.reactor.feed(&Instruction::LocalSet(temp_a))?;
        self.reactor.feed(&Instruction::LocalGet(src2))?;
        self.reactor.feed(&Instruction::LocalSet(temp_b))?;
        
        // Start with a_hi * b_hi
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // a_hi (sign-extended)
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // b_hi (sign-extended)
        self.reactor.feed(&Instruction::I64Mul)?; // a_hi * b_hi
        
        // Compute middle term: a_hi * b_lo (full 64-bit result)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // a_hi
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // b_lo
        self.reactor.feed(&Instruction::I64Mul)?; // a_hi * b_lo (64-bit result)
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?; // save for carry computation
        
        // Add high 32 bits of (a_hi * b_lo) to result
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // arithmetic shift for signed
        self.reactor.feed(&Instruction::I64Add)?;
        
        // Compute other middle term: a_lo * b_hi (full 64-bit result)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // a_lo
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // b_hi
        self.reactor.feed(&Instruction::I64Mul)?; // a_lo * b_hi (64-bit result)
        
        // Add it to the middle term accumulator for carry calculation
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Add)?; // sum of middle terms (low parts)
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?;
        
        // Add high 32 bits of the summed middle terms
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // arithmetic shift
        self.reactor.feed(&Instruction::I64Add)?;
        
        Ok(())
    }

    /// Helper to compute high 64 bits of unsigned 64x64 -> 128-bit multiplication
    fn emit_mulh_unsigned(&mut self, src1: u32, src2: u32) -> Result<(), E> {
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67;
        
        self.reactor.feed(&Instruction::LocalGet(src1))?;
        self.reactor.feed(&Instruction::LocalSet(temp_a))?;
        self.reactor.feed(&Instruction::LocalGet(src2))?;
        self.reactor.feed(&Instruction::LocalSet(temp_b))?;
        
        // Start with a_hi * b_hi (all unsigned)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // a_hi (unsigned)
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(&Instruction::I64Mul)?;
        
        // Compute middle term: a_hi * b_lo
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // a_hi
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // b_lo
        self.reactor.feed(&Instruction::I64Mul)?;
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?;
        
        // Add high 32 bits of (a_hi * b_lo)
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?;
        self.reactor.feed(&Instruction::I64Add)?;
        
        // Compute other middle term: a_lo * b_hi
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // a_lo
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // b_hi
        self.reactor.feed(&Instruction::I64Mul)?;
        
        // Add to middle term for carry calculation
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Add)?;
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?;
        
        // Add high 32 bits of summed middle terms
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?;
        self.reactor.feed(&Instruction::I64Add)?;
        
        Ok(())
    }

    /// Helper to compute high 64 bits of signed-unsigned 64x64 -> 128-bit multiplication
    fn emit_mulh_signed_unsigned(&mut self, src1: u32, src2: u32) -> Result<(), E> {
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67;
        
        self.reactor.feed(&Instruction::LocalGet(src1))?;
        self.reactor.feed(&Instruction::LocalSet(temp_a))?;
        self.reactor.feed(&Instruction::LocalGet(src2))?;
        self.reactor.feed(&Instruction::LocalSet(temp_b))?;
        
        // src1 is signed, src2 is unsigned
        
        // Start with a_hi * b_hi (a_hi signed, b_hi unsigned)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // a_hi (signed)
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(&Instruction::I64Mul)?;
        
        // Compute middle term: a_hi * b_lo (a_hi signed, b_lo unsigned)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?; // a_hi (signed)
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // b_lo
        self.reactor.feed(&Instruction::I64Mul)?;
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?;
        
        // Add high 32 bits of (a_hi * b_lo) - use signed shift
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I64Add)?;
        
        // Compute other middle term: a_lo * b_hi (a_lo unsigned, b_hi unsigned)
        self.reactor.feed(&Instruction::LocalGet(temp_a))?;
        self.reactor.feed(&Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?; // a_lo
        self.reactor.feed(&Instruction::LocalGet(temp_b))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(&Instruction::I64Mul)?;
        
        // Add to middle term for carry calculation
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Add)?;
        self.reactor.feed(&Instruction::LocalSet(temp_mid))?;
        
        // Add high 32 bits of summed middle terms - use signed shift
        self.reactor.feed(&Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(&Instruction::I64Const(32))?;
        self.reactor.feed(&Instruction::I64ShrS)?;
        self.reactor.feed(&Instruction::I64Add)?;
        
        Ok(())
    }

    /// Helper to translate load instructions
    fn translate_load(&mut self, base: Reg, offset: Imm, dest: Reg, op: LoadOp) -> Result<(), E> {
        if dest.0 == 0 {
            return Ok(()); // x0 is hardwired to zero
        }

        // Compute address: base + offset
        self.reactor
            .feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        
        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                self.reactor.feed(&Instruction::I32WrapI64)?;
            }
        } else {
            self.reactor.feed(&Instruction::I32Add)?;
        }

        // Load from memory
        match op {
            LoadOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                    // If RV64 but not memory64, extend to i64
                    if self.enable_rv64 {
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(&Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(&Instruction::I64ExtendI32U)?;
                    }
                }
            }
            LoadOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load32S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    if self.enable_rv64 {
                        self.reactor.feed(&Instruction::I64ExtendI32S)?;
                    }
                }
            }
            LoadOp::U32 => {
                // RV64 LWU instruction - load word unsigned (zero-extended)
                if self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Load32U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                    self.reactor.feed(&Instruction::I64ExtendI32U)?;
                }
            }
            LoadOp::I64 => {
                // RV64 LD instruction - load double-word
                self.reactor
                    .feed(&Instruction::I64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        self.reactor
            .feed(&Instruction::LocalSet(Self::reg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate store instructions
    fn translate_store(&mut self, base: Reg, offset: Imm, src: Reg, op: StoreOp) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        
        // Add instruction depends on whether we're using memory64 and RV64
        if self.enable_rv64 {
            self.reactor.feed(&Instruction::I64Add)?;
            // If not using memory64, wrap to 32-bit address
            if !self.use_memory64 {
                self.reactor.feed(&Instruction::I32WrapI64)?;
            }
        } else {
            self.reactor.feed(&Instruction::I32Add)?;
        }

        // Load value to store
        self.reactor
            .feed(&Instruction::LocalGet(Self::reg_to_local(src)))?;
        
        // If RV64 but not memory64, need to wrap i64 value to i32 for 32-bit stores
        let need_wrap = self.enable_rv64 && !self.use_memory64 && !matches!(op, StoreOp::I64);
        if need_wrap {
            self.reactor.feed(&Instruction::I32WrapI64)?;
        }

        // Store to memory
        match op {
            StoreOp::I8 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I16 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I32 => {
                if self.enable_rv64 && self.use_memory64 {
                    self.reactor
                        .feed(&Instruction::I64Store32(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                } else {
                    self.reactor
                        .feed(&Instruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }))?;
                }
            }
            StoreOp::I64 => {
                // RV64 SD instruction - store double-word
                self.reactor
                    .feed(&Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
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
    ) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load from memory
        match op {
            FLoadOp::F32 => {
                self.reactor
                    .feed(&Instruction::F32Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
                self.reactor.feed(&Instruction::F64PromoteF32)?;
            }
            FLoadOp::F64 => {
                self.reactor
                    .feed(&Instruction::F64Load(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        self.reactor
            .feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;
        Ok(())
    }

    /// Helper to translate floating-point store instructions
    fn translate_fstore(
        &mut self,
        base: Reg,
        offset: Imm,
        src: FReg,
        op: FStoreOp,
    ) -> Result<(), E> {
        // Compute address: base + offset
        self.reactor
            .feed(&Instruction::LocalGet(Self::reg_to_local(base)))?;
        self.emit_imm(offset)?;
        self.reactor.feed(&Instruction::I32Add)?;

        // Load value to store
        self.reactor
            .feed(&Instruction::LocalGet(Self::freg_to_local(src)))?;

        // Store to memory
        match op {
            FStoreOp::F32 => {
                self.reactor.feed(&Instruction::F32DemoteF64)?;
                self.reactor
                    .feed(&Instruction::F32Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    }))?;
            }
            FStoreOp::F64 => {
                self.reactor
                    .feed(&Instruction::F64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }))?;
            }
        }

        Ok(())
    }

    /// Helper to emit sign-injection for single-precision floats
    fn emit_fsgnj_s(&mut self, dest: FReg, src1: FReg, src2: FReg, op: FsgnjOp) -> Result<(), E> {
        // Sign injection uses bit manipulation on the float representation
        // Get magnitude from src1, sign from src2 (possibly modified)

        // Convert src1 to i32 to manipulate bits
        self.reactor
            .feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.unbox_f32()?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;

        // Mask to keep only magnitude (clear sign bit): 0x7FFFFFFF
        self.reactor.feed(&Instruction::I32Const(0x7FFFFFFF))?;
        self.reactor.feed(&Instruction::I32And)?;

        // Get sign bit from src2
        self.reactor
            .feed(&Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.unbox_f32()?;
        self.reactor.feed(&Instruction::I32ReinterpretF32)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly: mask with 0x80000000
                self.reactor
                    .feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor
                    .feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
                self.reactor
                    .feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32Xor)?; // Flip the sign bit
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits of src1 and src2
                // Need original src1 sign
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.unbox_f32()?;
                self.reactor.feed(&Instruction::I32ReinterpretF32)?;
                self.reactor.feed(&Instruction::I32Xor)?;
                self.reactor
                    .feed(&Instruction::I32Const(0x80000000_u32 as i32))?;
                self.reactor.feed(&Instruction::I32And)?;
            }
        }

        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I32Or)?;
        self.reactor.feed(&Instruction::F32ReinterpretI32)?;
        self.nan_box_f32()?;
        self.reactor
            .feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;

        Ok(())
    }

    /// Helper to emit sign-injection for double-precision floats
    fn emit_fsgnj_d(&mut self, dest: FReg, src1: FReg, src2: FReg, op: FsgnjOp) -> Result<(), E> {
        // Similar to single-precision but using i64
        // Convert src1 to i64 to manipulate bits
        self.reactor
            .feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
        self.reactor.feed(&Instruction::I64ReinterpretF64)?;

        // Mask to keep only magnitude (clear sign bit)
        self.reactor
            .feed(&Instruction::I64Const(0x7FFFFFFFFFFFFFFF))?;
        self.reactor.feed(&Instruction::I64And)?;

        // Get sign bit from src2
        self.reactor
            .feed(&Instruction::LocalGet(Self::freg_to_local(src2)))?;
        self.reactor.feed(&Instruction::I64ReinterpretF64)?;

        match op {
            FsgnjOp::Sgnj => {
                // Use sign from src2 directly
                self.reactor
                    .feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
            FsgnjOp::Sgnjn => {
                // Use negated sign from src2
                self.reactor
                    .feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
                self.reactor
                    .feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64Xor)?;
            }
            FsgnjOp::Sgnjx => {
                // XOR sign bits
                self.reactor
                    .feed(&Instruction::LocalGet(Self::freg_to_local(src1)))?;
                self.reactor.feed(&Instruction::I64ReinterpretF64)?;
                self.reactor.feed(&Instruction::I64Xor)?;
                self.reactor
                    .feed(&Instruction::I64Const(0x8000000000000000_u64 as i64))?;
                self.reactor.feed(&Instruction::I64And)?;
            }
        }

        // Combine magnitude and sign
        self.reactor.feed(&Instruction::I64Or)?;
        self.reactor.feed(&Instruction::F64ReinterpretI64)?;
        self.reactor
            .feed(&Instruction::LocalSet(Self::freg_to_local(dest)))?;

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
    pub fn translate_bytes(
        &mut self,
        bytes: &[u8],
        start_pc: u32,
        xlen: Xlen,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<usize, ()> {
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
            if let Err(_) = self.translate_instruction(&inst, pc, is_compressed, f) {
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
    pub fn seal(&mut self) -> Result<(), E> {
        self.reactor.seal(&Instruction::Unreachable)
    }

    /// Get the reactor (consumes self)
    pub fn into_reactor(self) -> Reactor<E, F> {
        self.reactor
    }
}

impl<'cb, 'ctx, E, F: InstructionSink<E>> Default for RiscVRecompiler<'cb, 'ctx, E, F> {
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
    I8,  // Load byte (sign-extended)
    U8,  // Load byte (zero-extended)
    I16, // Load halfword (sign-extended)
    U16, // Load halfword (zero-extended)
    I32, // Load word
    U32, // Load word (zero-extended, for RV64 LWU)
    I64, // Load double-word (for RV64 LD)
}

/// Store operation types
#[derive(Debug, Clone, Copy)]
enum StoreOp {
    I8,  // Store byte
    I16, // Store halfword
    I32, // Store word
    I64, // Store double-word (for RV64 SD)
}

/// Floating-point load operation types
#[derive(Debug, Clone, Copy)]
enum FLoadOp {
    F32, // Load single-precision float
    F64, // Load double-precision float
}

/// Floating-point store operation types
#[derive(Debug, Clone, Copy)]
enum FStoreOp {
    F32, // Store single-precision float
    F64, // Store double-precision float
}

/// Sign-injection operation types
#[derive(Debug, Clone, Copy)]
enum FsgnjOp {
    Sgnj,  // Copy sign from src2
    Sgnjn, // Copy negated sign from src2
    Sgnjx, // XOR signs of src1 and src2
}

#[cfg(test)]
mod tests {
    use super::*;
    use rv_asm::{Inst, Xlen};

    #[test]
    fn test_recompiler_creation() {
        let _recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);
        // Just ensure it can be created without panicking
    }

    #[test]
    fn test_addi_instruction() {
        // Test ADDI instruction: addi x1, x0, 42
        // Use base_pc to offset high addresses
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(42),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_add_instruction() {
        // Test ADD instruction: add x3, x1, x2
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_load_instruction() {
        // Test LW instruction: lw x1, 0(x2)
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::Lw {
            offset: rv_asm::Imm::new_i32(0),
            dest: rv_asm::Reg(1),
            base: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_store_instruction() {
        // Test SW instruction: sw x1, 4(x2)
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

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

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_branch_instruction() {
        // Test BEQ instruction: beq x1, x2, offset
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::Beq {
            offset: rv_asm::Imm::new_i32(8),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_mul_instruction() {
        // Test MUL instruction from M extension
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::Mul {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_fadd_instruction() {
        // Test FADD.S instruction from F extension
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let inst = Inst::FaddS {
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
            dest: rv_asm::FReg(1),
            src1: rv_asm::FReg(2),
            src2: rv_asm::FReg(3),
        };

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_decode_and_translate() {
        // Test decoding a real instruction and translating it
        // This is "addi a0, a0, 0" which is a common NOP-like instruction
        let instruction_bytes: u32 = 0x00050513;
        let (inst, is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();

        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        assert!(
            recompiler
                .translate_instruction(&inst, 0x1000, is_compressed, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_multiple_instructions() {
        // Test translating multiple instructions in sequence
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        // addi x1, x0, 5
        let inst1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&inst1, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // addi x2, x0, 3
        let inst2 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(3),
            dest: rv_asm::Reg(2),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&inst2, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // add x3, x1, x2  (should compute 5 + 3 = 8)
        let inst3 = Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };
        assert!(
            recompiler
                .translate_instruction(&inst3, 0x1008, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_translate_from_bytes() {
        // Test translating from raw bytecode
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        // Simple program: addi x1, x0, 5 (0x00500093)
        let bytes = [0x93, 0x00, 0x50, 0x00]; // Little-endian

        let result = recompiler.translate_bytes(&bytes, 0x1000, Xlen::Rv32, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4); // Should have translated 4 bytes
    }

    #[test]
    fn test_translate_compressed_from_bytes() {
        // Test translating compressed instructions from bytes
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        // c.addi x1, 5 (0x0095) - compressed instruction
        let bytes = [0x95, 0x00];

        let result = recompiler.translate_bytes(&bytes, 0x1000, Xlen::Rv32, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2); // Should have translated 2 bytes
    }

    #[test]
    fn test_register_mapping() {
        // Test that register indices map correctly
        assert_eq!(
            RiscVRecompiler::<Infallible, Function>::reg_to_local(rv_asm::Reg(0)),
            0
        );
        assert_eq!(
            RiscVRecompiler::<Infallible, Function>::reg_to_local(rv_asm::Reg(31)),
            31
        );
        assert_eq!(
            RiscVRecompiler::<Infallible, Function>::freg_to_local(rv_asm::FReg(0)),
            32
        );
        assert_eq!(
            RiscVRecompiler::<Infallible, Function>::freg_to_local(rv_asm::FReg(31)),
            63
        );
        assert_eq!(RiscVRecompiler::<Infallible, Function>::pc_local(), 64);
    }

    #[test]
    fn test_hint_tracking_disabled_by_default() {
        // HINT tracking should be disabled by default
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        // Translate a HINT instruction: addi x0, x0, 1
        let hint_inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };

        assert!(
            recompiler
                .translate_instruction(
                    &hint_inst,
                    0x1000,
                    IsCompressed::No,
                    &mut |a| Function::new(a.collect::<Vec<_>>())
                )
                .is_ok()
        );

        // Should not have collected any hints since tracking is disabled
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_tracking_enabled() {
        // Test HINT tracking when explicitly enabled
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true, // Enable HINT tracking
            false, // Disable RV64
            false, // Disable memory64
        );

        // Translate a HINT instruction: addi x0, x0, 1
        let hint1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&hint1, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Translate another HINT: addi x0, x0, 2
        let hint2 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(2),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&hint2, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

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
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true, // Enable HINT tracking
            false, // Disable RV64
            false, // Disable memory64
        );

        // Regular addi x1, x0, 5 (not a HINT)
        let regular_addi = Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&regular_addi, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );

        // Should not be tracked as a HINT
        assert_eq!(recompiler.get_hints().len(), 0);

        // Now translate a real HINT: addi x0, x0, 1
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&hint, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Should only have the HINT
        let hints = recompiler.get_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].pc, 0x1004);
        assert_eq!(hints[0].value, 1);
    }

    #[test]
    fn test_hint_clear() {
        // Test clearing collected HINTs
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true,
            false, // Disable RV64
            false, // Disable memory64
        );

        // Collect some hints
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&hint, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 1);

        // Clear hints
        recompiler.clear_hints();
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_tracking_toggle() {
        // Test toggling HINT tracking on and off
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);

        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };

        // Initially disabled
        assert!(
            recompiler
                .translate_instruction(&hint, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 0);

        // Enable tracking
        recompiler.set_hint_tracking(true);
        assert!(
            recompiler
                .translate_instruction(&hint, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 1);

        // Disable tracking (should clear existing hints)
        recompiler.set_hint_tracking(false);
        assert_eq!(recompiler.get_hints().len(), 0);
        assert!(
            recompiler
                .translate_instruction(&hint, 0x1008, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[test]
    fn test_hint_from_rv_corpus_pattern() {
        // Test the actual pattern used in rv-corpus test files
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            true,
            false, // Disable RV64
            false, // Disable memory64
        );

        // Simulate test case markers from rv-corpus
        for test_case in 1..=5 {
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(test_case),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            let pc = 0x1000 + (test_case as u32 * 4);
            assert!(
                recompiler
                    .translate_instruction(&hint, pc, IsCompressed::No, &mut |a| Function::new(
                        a.collect::<Vec<_>>()
                    ))
                    .is_ok()
            );
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

            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext<Infallible, Function>| {
                collected.push(*hint);
            };

            recompiler.set_hint_callback(&mut callback);

            // Translate some HINT instructions
            let hint1 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(1),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(
                        &hint1,
                        0x1000,
                        IsCompressed::No,
                        &mut |a| Function::new(a.collect::<Vec<_>>())
                    )
                    .is_ok()
            );

            let hint2 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(2),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(
                        &hint2,
                        0x1004,
                        IsCompressed::No,
                        &mut |a| Function::new(a.collect::<Vec<_>>())
                    )
                    .is_ok()
            );

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
                false, // Disable RV64
                false, // Disable memory64
            );

            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext<Infallible, Function>| {
                callback_hints.push(*hint);
            };

            recompiler.set_hint_callback(&mut callback);

            // Translate a HINT
            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(42),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(&hint, 0x2000, IsCompressed::No, &mut |a| Function::new(
                        a.collect::<Vec<_>>()
                    ))
                    .is_ok()
            );

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
            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext<Infallible, Function>| {
                collected.push(hint.value);
            };

            recompiler.set_hint_callback(&mut callback);

            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(1),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };

            // First HINT should invoke callback
            assert!(
                recompiler
                    .translate_instruction(&hint, 0x1000, IsCompressed::No, &mut |a| Function::new(
                        a.collect::<Vec<_>>()
                    ))
                    .is_ok()
            );

            // Clear callback
            recompiler.clear_hint_callback();

            // Translate another HINT - callback should not be invoked
            let hint2 = Inst::Addi {
                imm: rv_asm::Imm::new_i32(2),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(
                        &hint2,
                        0x1004,
                        IsCompressed::No,
                        &mut |a| Function::new(a.collect::<Vec<_>>())
                    )
                    .is_ok()
            );

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

            let mut callback = |hint: &HintInfo, _ctx: &mut HintContext<Infallible, Function>| {
                assert_eq!(hint.value, 99);
                callback_values.push(hint.value);
            };

            recompiler.set_hint_callback(&mut callback);

            let hint = Inst::Addi {
                imm: rv_asm::Imm::new_i32(99),
                dest: rv_asm::Reg(0),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(&hint, 0x1000, IsCompressed::No, &mut |a| Function::new(
                        a.collect::<Vec<_>>()
                    ))
                    .is_ok()
            );

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

            let mut callback = |_hint: &HintInfo, _ctx: &mut HintContext<Infallible, Function>| {
                invoked = true;
            };

            recompiler.set_hint_callback(&mut callback);

            // Regular addi x1, x0, 5 (not a HINT)
            let regular_addi = Inst::Addi {
                imm: rv_asm::Imm::new_i32(5),
                dest: rv_asm::Reg(1),
                src1: rv_asm::Reg(0),
            };
            assert!(
                recompiler
                    .translate_instruction(&regular_addi, 0x1000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );

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

            let mut callback = |hint: &HintInfo, ctx: &mut HintContext<Infallible, Function>| {
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
                assert!(
                    recompiler
                        .translate_instruction(
                            &hint,
                            0x1000 + (i as u32 * 4),
                            IsCompressed::No,
                            &mut |a| Function::new(a.collect::<Vec<_>>())
                        )
                        .is_ok()
                );
            }

            // Drop recompiler
        }

        // Verify callback was invoked for all HINTs
        assert_eq!(hint_values.len(), 3);
        assert_eq!(hint_values[0], 1);
        assert_eq!(hint_values[1], 2);
        assert_eq!(hint_values[2], 3);
    }

    #[test]
    fn test_ecall_callback_basic() {
        // Test basic ecall callback functionality
        use alloc::vec::Vec;

        let mut ecall_pcs = Vec::new();

        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);

            let mut callback = |ecall: &EcallInfo, _ctx: &mut HintContext<Infallible, Function>| {
                ecall_pcs.push(ecall.pc);
            };

            recompiler.set_ecall_callback(&mut callback);

            // Translate an ECALL instruction
            let ecall = Inst::Ecall;
            assert!(
                recompiler
                    .translate_instruction(
                        &ecall,
                        0x2000,
                        IsCompressed::No,
                        &mut |a| Function::new(a.collect::<Vec<_>>())
                    )
                    .is_ok()
            );

            // Drop recompiler
        }

        // Verify callback was invoked
        assert_eq!(ecall_pcs.len(), 1);
        assert_eq!(ecall_pcs[0], 0x2000);
    }

    #[test]
    fn test_ebreak_callback_basic() {
        // Test basic ebreak callback functionality
        use alloc::vec::Vec;

        let mut ebreak_pcs = Vec::new();

        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);

            let mut callback =
                |ebreak: &EbreakInfo, _ctx: &mut HintContext<Infallible, Function>| {
                    ebreak_pcs.push(ebreak.pc);
                };

            recompiler.set_ebreak_callback(&mut callback);

            // Translate an EBREAK instruction
            let ebreak = Inst::Ebreak;
            assert!(
                recompiler
                    .translate_instruction(&ebreak, 0x3000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );

            // Drop recompiler
        }

        // Verify callback was invoked
        assert_eq!(ebreak_pcs.len(), 1);
        assert_eq!(ebreak_pcs[0], 0x3000);
    }

    #[test]
    fn test_ecall_callback_with_code_generation() {
        // Test that ecall callback can generate WebAssembly instructions
        use alloc::vec::Vec;

        let mut ecall_count = 0;

        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);

            let mut callback = |_ecall: &EcallInfo, ctx: &mut HintContext<Infallible, Function>| {
                ecall_count += 1;
                // Generate a NOP instruction for each ECALL
                ctx.emit(&Instruction::Nop).ok();
            };

            recompiler.set_ecall_callback(&mut callback);

            // Translate multiple ECALL instructions
            for i in 0..3 {
                let ecall = Inst::Ecall;
                let pc = 0x1000 + (i * 4);
                assert!(
                    recompiler
                        .translate_instruction(
                            &ecall,
                            pc,
                            IsCompressed::No,
                            &mut |a| Function::new(a.collect::<Vec<_>>())
                        )
                        .is_ok()
                );
            }

            // Drop recompiler
        }

        // Verify callback was invoked for all ECALLs
        assert_eq!(ecall_count, 3);
    }

    #[test]
    fn test_ebreak_callback_with_code_generation() {
        // Test that ebreak callback can generate WebAssembly instructions
        let mut ebreak_count = 0;

        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);

            let mut callback =
                |_ebreak: &EbreakInfo, ctx: &mut HintContext<Infallible, Function>| {
                    ebreak_count += 1;
                    // Generate a NOP instruction for each EBREAK
                    ctx.emit(&Instruction::Nop).ok();
                };

            recompiler.set_ebreak_callback(&mut callback);

            // Translate multiple EBREAK instructions
            for i in 0..3 {
                let ebreak = Inst::Ebreak;
                let pc = 0x2000 + (i * 4);
                assert!(
                    recompiler
                        .translate_instruction(&ebreak, pc, IsCompressed::No, &mut |a| {
                            Function::new(a.collect::<Vec<_>>())
                        })
                        .is_ok()
                );
            }

            // Drop recompiler
        }

        // Verify callback was invoked for all EBREAKs
        assert_eq!(ebreak_count, 3);
    }

    #[test]
    fn test_ecall_ebreak_callbacks_clear() {
        // Test clearing the ecall and ebreak callbacks
        let mut ecall_count = 0;
        let mut ebreak_count = 0;

        {
            let mut recompiler = RiscVRecompiler::new_with_base_pc(0x1000);

            let mut ecall_cb =
                |_ecall: &EcallInfo, _ctx: &mut HintContext<Infallible, Function>| {
                    ecall_count += 1;
                };

            let mut ebreak_cb =
                |_ebreak: &EbreakInfo, _ctx: &mut HintContext<Infallible, Function>| {
                    ebreak_count += 1;
                };

            recompiler.set_ecall_callback(&mut ecall_cb);
            recompiler.set_ebreak_callback(&mut ebreak_cb);

            // First ECALL and EBREAK should invoke callbacks
            assert!(
                recompiler
                    .translate_instruction(&Inst::Ecall, 0x1000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );
            assert!(
                recompiler
                    .translate_instruction(&Inst::Ebreak, 0x1004, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );

            // Clear callbacks
            recompiler.clear_ecall_callback();
            recompiler.clear_ebreak_callback();

            // Second ECALL and EBREAK should not invoke callbacks (will use default behavior)
            assert!(
                recompiler
                    .translate_instruction(&Inst::Ecall, 0x1008, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );
            assert!(
                recompiler
                    .translate_instruction(&Inst::Ebreak, 0x100c, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                    .is_ok()
            );

            // Drop recompiler
        }

        // Verify callbacks were invoked only once
        assert_eq!(ecall_count, 1);
        assert_eq!(ebreak_count, 1);
    }

    #[test]
    fn test_rv64_instructions() {
        // Test RV64 instructions when RV64 is enabled
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            false, // Disable HINT tracking
            true,  // Enable RV64
            false, // Disable memory64 (use i32 addresses)
        );

        // Test ADDIW (Add Word Immediate)
        let addiw = Inst::AddiW {
            imm: rv_asm::Imm::new_i32(10),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(2),
        };
        assert!(
            recompiler
                .translate_instruction(&addiw, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test ADDW (Add Word)
        let addw = Inst::AddW {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(4),
            src2: rv_asm::Reg(5),
        };
        assert!(
            recompiler
                .translate_instruction(&addw, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test SLLIW (Shift Left Logical Word Immediate)
        let slliw = Inst::SlliW {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(6),
            src1: rv_asm::Reg(7),
        };
        assert!(
            recompiler
                .translate_instruction(&slliw, 0x1008, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test LWU (Load Word Unsigned)
        let lwu = Inst::Lwu {
            offset: rv_asm::Imm::new_i32(0),
            dest: rv_asm::Reg(8),
            base: rv_asm::Reg(9),
        };
        assert!(
            recompiler
                .translate_instruction(&lwu, 0x100c, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test LD (Load Double-word)
        let ld = Inst::Ld {
            offset: rv_asm::Imm::new_i32(8),
            dest: rv_asm::Reg(10),
            base: rv_asm::Reg(11),
        };
        assert!(
            recompiler
                .translate_instruction(&ld, 0x1010, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test SD (Store Double-word)
        let sd = Inst::Sd {
            offset: rv_asm::Imm::new_i32(16),
            base: rv_asm::Reg(12),
            src: rv_asm::Reg(13),
        };
        assert!(
            recompiler
                .translate_instruction(&sd, 0x1014, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_rv64_disabled_by_default() {
        // Test that RV64 instructions are not supported when RV64 is disabled
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new();

        // ADDIW should fail when RV64 is disabled
        let addiw = Inst::AddiW {
            imm: rv_asm::Imm::new_i32(10),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(2),
        };
        
        // When RV64 is disabled, we emit Unreachable, which should still succeed
        // but the generated code will trap if executed
        assert!(
            recompiler
                .translate_instruction(&addiw, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_rv64_with_memory64() {
        // Test RV64 with memory64 enabled
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            false, // Disable HINT tracking
            true,  // Enable RV64
            true,  // Enable memory64 (use i64 addresses)
        );

        // Test LD with memory64
        let ld = Inst::Ld {
            offset: rv_asm::Imm::new_i32(8),
            dest: rv_asm::Reg(10),
            base: rv_asm::Reg(11),
        };
        assert!(
            recompiler
                .translate_instruction(&ld, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test SD with memory64
        let sd = Inst::Sd {
            offset: rv_asm::Imm::new_i32(16),
            base: rv_asm::Reg(12),
            src: rv_asm::Reg(13),
        };
        assert!(
            recompiler
                .translate_instruction(&sd, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_rv64_mulh_instructions() {
        // Test RV64 multiply-high instructions (Mulh, Mulhu, Mulhsu)
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            false, // Disable HINT tracking
            true,  // Enable RV64
            false, // Disable memory64
        );

        // Test MULH (signed x signed)
        let mulh = Inst::Mulh {
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(2),
            src2: rv_asm::Reg(3),
        };
        assert!(
            recompiler
                .translate_instruction(&mulh, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test MULHU (unsigned x unsigned)
        let mulhu = Inst::Mulhu {
            dest: rv_asm::Reg(4),
            src1: rv_asm::Reg(5),
            src2: rv_asm::Reg(6),
        };
        assert!(
            recompiler
                .translate_instruction(&mulhu, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test MULHSU (signed x unsigned)
        let mulhsu = Inst::Mulhsu {
            dest: rv_asm::Reg(7),
            src1: rv_asm::Reg(8),
            src2: rv_asm::Reg(9),
        };
        assert!(
            recompiler
                .translate_instruction(&mulhsu, 0x1008, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Verify that writing to x0 is properly ignored
        let mulh_x0 = Inst::Mulh {
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(2),
            src2: rv_asm::Reg(3),
        };
        assert!(
            recompiler
                .translate_instruction(&mulh_x0, 0x100c, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_rv64_float_conversions() {
        // Test RV64 floating-point conversion instructions
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            false, // Disable HINT tracking
            true,  // Enable RV64
            false, // Disable memory64
        );

        // Test FCVT.L.S (float to signed i64)
        let fcvt_ls = Inst::FcvtLS {
            dest: rv_asm::Reg(1),
            src: rv_asm::FReg(2),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_ls, 0x1000, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.LU.S (float to unsigned i64)
        let fcvt_lus = Inst::FcvtLuS {
            dest: rv_asm::Reg(3),
            src: rv_asm::FReg(4),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_lus, 0x1004, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.S.L (signed i64 to float)
        let fcvt_sl = Inst::FcvtSL {
            dest: rv_asm::FReg(5),
            src: rv_asm::Reg(6),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_sl, 0x1008, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.S.LU (unsigned i64 to float)
        let fcvt_slu = Inst::FcvtSLu {
            dest: rv_asm::FReg(7),
            src: rv_asm::Reg(8),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_slu, 0x100c, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.L.D (double to signed i64)
        let fcvt_ld = Inst::FcvtLD {
            dest: rv_asm::Reg(9),
            src: rv_asm::FReg(10),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_ld, 0x1010, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.LU.D (double to unsigned i64)
        let fcvt_lud = Inst::FcvtLuD {
            dest: rv_asm::Reg(11),
            src: rv_asm::FReg(12),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_lud, 0x1014, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.D.L (signed i64 to double)
        let fcvt_dl = Inst::FcvtDL {
            dest: rv_asm::FReg(13),
            src: rv_asm::Reg(14),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_dl, 0x1018, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FCVT.D.LU (unsigned i64 to double)
        let fcvt_dlu = Inst::FcvtDLu {
            dest: rv_asm::FReg(15),
            src: rv_asm::Reg(16),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_dlu, 0x101c, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FMV.X.D (double register to integer register)
        let fmv_xd = Inst::FmvXD {
            dest: rv_asm::Reg(17),
            src: rv_asm::FReg(18),
        };
        assert!(
            recompiler
                .translate_instruction(&fmv_xd, 0x1020, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Test FMV.D.X (integer register to double register)
        let fmv_dx = Inst::FmvDX {
            dest: rv_asm::FReg(19),
            src: rv_asm::Reg(20),
        };
        assert!(
            recompiler
                .translate_instruction(&fmv_dx, 0x1024, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );

        // Verify that writing to x0 is properly ignored (for conversion to integer)
        let fcvt_ls_x0 = Inst::FcvtLS {
            dest: rv_asm::Reg(0),
            src: rv_asm::FReg(2),
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
        };
        assert!(
            recompiler
                .translate_instruction(&fcvt_ls_x0, 0x1028, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    #[test]
    fn test_base_func_offset() {
        // Test that base_func_offset can be set and retrieved
        let mut recompiler = RiscVRecompiler::<Infallible, Function>::new();
        
        // Default should be 0
        assert_eq!(recompiler.base_func_offset(), 0);
        
        // Set to a new value
        recompiler.set_base_func_offset(15);
        assert_eq!(recompiler.base_func_offset(), 15);
        
        // Create with offset using new_with_all_config
        let recompiler2 = RiscVRecompiler::<Infallible, Function>::new_with_all_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            None,
            0x1000,
            25, // base_func_offset
            false,
            false,
            false,
        );
        assert_eq!(recompiler2.base_func_offset(), 25);
    }
}
