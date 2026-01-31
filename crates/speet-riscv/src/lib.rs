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
//! ```ignore
//! use speet_riscv::RiscVRecompiler;
//! use rv_asm::{Inst, Xlen};
//!
//! // Create a recompiler instance
//!
//!   /// let mut ctx = ();
//!
//! // Decode and translate instructions
//! let instruction_bytes: u32 = 0x00a50533; // add a0, a0, a0
//! let (inst, is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
//! recompiler.translate_instruction(&mut ctx, &inst, 0x1000, is_compressed, &mut |a| Function::new(a.collect::<Vec<_>>()));
//! ```
//!
//! ## RISC-V Specification Compliance
//!
//! This implementation follows the RISC-V Unprivileged Specification:
//! https://docs.riscv.org/reference/isa/unpriv/unpriv-index.html
//!
//! Key specification quotes are included as documentation comments throughout the code.

#![no_std]
pub mod direct;
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

/// Unified context for all callbacks
///
/// This struct provides access to the WebAssembly instruction emitter and other
/// compilation state. It is passed to all callbacks (HINT, ECALL, EBREAK, mapper).
pub struct CallbackContext<'a, Context, E, F: InstructionSink<Context, E>> {
    /// Reference to the reactor for emitting WebAssembly instructions
    pub reactor: &'a mut Reactor<Context, E, F>,
}

impl<'a, Context, E, F: InstructionSink<Context, E>> CallbackContext<'a, Context, E, F> {
    /// Emit a WebAssembly instruction
    pub fn emit(&mut self, ctx: &mut Context, instruction: &Instruction) -> Result<(), E> {
        self.reactor.feed(ctx, instruction)
    }
}

/// Legacy type alias for backwards compatibility
pub type MapperContext<'a, Context, E, F> = CallbackContext<'a, Context, E, F>;
/// Legacy type alias for backwards compatibility
pub type HintContext<'a, Context, E, F> = CallbackContext<'a, Context, E, F>;

/// Trait for address mapping callbacks (paging support)
///
/// This trait defines the interface for callbacks that translate virtual addresses
/// to physical addresses. The callback receives the virtual address on the Wasm stack
/// and should leave the physical address on the stack.
///
/// See PAGING.md for detailed documentation on the paging system.
pub trait MapperCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Translate a virtual address to a physical address
    ///
    /// # Stack State
    /// - Input: Virtual address (i64 or i32 depending on use_memory64/enable_rv64)
    /// - Output: Physical address (same type as input)
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E>;
}

/// Blanket implementation of MapperCallback for FnMut closures
impl<Context, E, F: InstructionSink<Context, E>, T> MapperCallback<Context, E, F> for T
where
    T: FnMut(&mut Context, &mut CallbackContext<Context, E, F>) -> Result<(), E>,
{
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E> {
        self(ctx, callback_ctx)
    }
}

/// Trait for HINT instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when HINT
/// instructions are encountered during translation. Implementations receive
/// both the HINT information and a unified context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait HintCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Process a HINT instruction
    ///
    /// # Arguments
    /// * `hint` - Information about the detected HINT instruction
    /// * `ctx` - User context for passing external state
    /// * `callback_ctx` - Unified context for emitting WebAssembly instructions
    fn call(
        &mut self,
        hint: &HintInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    );
}

/// Blanket implementation of HintCallback for FnMut closures
impl<Context, E, G: InstructionSink<Context, E>, F> HintCallback<Context, E, G> for F
where
    F: FnMut(&HintInfo, &mut Context, &mut CallbackContext<Context, E, G>),
{
    fn call(
        &mut self,
        hint: &HintInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, G>,
    ) {
        self(hint, ctx, callback_ctx)
    }
}

/// Trait for ECALL instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when ECALL
/// instructions are encountered during translation. Implementations receive
/// both the ECALL information and a unified context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait EcallCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Process an ECALL instruction
    ///
    /// # Arguments
    /// * `ecall` - Information about the detected ECALL instruction
    /// * `ctx` - User context for passing external state
    /// * `callback_ctx` - Unified context for emitting WebAssembly instructions
    fn call(
        &mut self,
        ecall: &EcallInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    );
}

/// Blanket implementation of EcallCallback for FnMut closures
impl<Context, E, G: InstructionSink<Context, E>, F> EcallCallback<Context, E, G> for F
where
    F: FnMut(&EcallInfo, &mut Context, &mut CallbackContext<Context, E, G>),
{
    fn call(
        &mut self,
        ecall: &EcallInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, G>,
    ) {
        self(ecall, ctx, callback_ctx)
    }
}

/// Trait for EBREAK instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when EBREAK
/// instructions are encountered during translation. Implementations receive
/// both the EBREAK information and a unified context for generating WebAssembly code.
///
/// The trait is automatically implemented for all `FnMut` closures with the
/// appropriate signature.
pub trait EbreakCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Process an EBREAK instruction
    ///
    /// # Arguments
    /// * `ebreak` - Information about the detected EBREAK instruction
    /// * `ctx` - User context for passing external state
    /// * `callback_ctx` - Unified context for emitting WebAssembly instructions
    fn call(
        &mut self,
        ebreak: &EbreakInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    );
}

/// Blanket implementation of EbreakCallback for FnMut closures
impl<Context, E, G: InstructionSink<Context, E>, F> EbreakCallback<Context, E, G> for F
where
    F: FnMut(&EbreakInfo, &mut Context, &mut CallbackContext<Context, E, G>),
{
    fn call(
        &mut self,
        ebreak: &EbreakInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, G>,
    ) {
        self(ebreak, ctx, callback_ctx)
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
pub struct RiscVRecompiler<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> {
    reactor: Reactor<Context, E, F>,
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
    hint_callback: Option<&'cb mut (dyn HintCallback<Context, E, F> + 'ctx)>,
    /// Optional callback for ECALL instructions
    ecall_callback: Option<&'cb mut (dyn EcallCallback<Context, E, F> + 'ctx)>,
    /// Optional callback for EBREAK instructions
    ebreak_callback: Option<&'cb mut (dyn EbreakCallback<Context, E, F> + 'ctx)>,
    /// Optional callback for address mapping (paging support - see PAGING.md)
    mapper_callback: Option<&'cb mut (dyn MapperCallback<Context, E, F> + 'ctx)>,
    /// Whether to enable RV64 instruction support (disabled by default)
    enable_rv64: bool,
    /// Whether to use memory64 (i64 addresses) instead of memory32 (i32 addresses)
    /// Only relevant when enable_rv64 is true
    use_memory64: bool,
    /// Whether to enable speculative call lowering for ABI-compliant calls
    /// (JAL x1 and JALR x1). See SPECULATIVE_CALLS.md for details.
    /// When enabled, these instructions are lowered to native WASM calls with
    /// exception-based return validation instead of jumps.
    enable_speculative_calls: bool,
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>>
    RiscVRecompiler<'cb, 'ctx, Context, E, F>
{
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
            mapper_callback: None,
            enable_rv64,
            use_memory64,
            enable_speculative_calls: false,
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
            mapper_callback: None,
            enable_rv64,
            use_memory64,
            enable_speculative_calls: false,
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
    /// ```ignore
    /// # use speet_riscv::{RiscVRecompiler, HintInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    ///
    /// let mut ctx = ();
    /// let mut my_callback = |hint: &HintInfo, ctx: &mut HintContext<'_, (), Infallible, Function>| {
    ///     println!("Test case {} at PC 0x{:x}", hint.value, hint.pc);
    ///     // Optionally emit WebAssembly instructions
    ///     callback_ctx.emit(ctx, &Instruction::Nop).ok();
    /// };
    /// recompiler.set_hint_callback(&mut my_callback);
    /// ```
    pub fn set_hint_callback(
        &mut self,
        callback: &'cb mut (dyn HintCallback<Context, E, F> + 'ctx),
    ) {
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
    /// ```ignore
    /// # use speet_riscv::{RiscVRecompiler, EcallInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    ///
    /// let mut ctx = ();
    /// let mut my_callback = |ecall: &EcallInfo, ctx: &mut HintContext<'_, (), Infallible, Function>| {
    ///     println!("ECALL at PC 0x{:x}", ecall.pc);
    ///     // Optionally emit WebAssembly instructions for the ecall
    ///     callback_ctx.emit(ctx, &Instruction::Nop).ok();
    /// };
    /// recompiler.set_ecall_callback(&mut my_callback);
    /// ```
    pub fn set_ecall_callback(
        &mut self,
        callback: &'cb mut (dyn EcallCallback<Context, E, F> + 'ctx),
    ) {
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
    /// ```ignore
    /// # use speet_riscv::{RiscVRecompiler, EbreakInfo, HintContext};
    /// # use wasm_encoder::Instruction;
    ///
    /// let mut recompiler = RiscVRecompiler::new();
    /// let mut ctx = ();
    /// let mut my_callback = |ebreak: &EbreakInfo, ctx: &mut HintContext<'_, (), Infallible, Function>| {
    ///     println!("EBREAK at PC 0x{:x}", ebreak.pc);
    ///     // Optionally emit WebAssembly instructions for the ebreak
    ///     callback_ctx.emit(ctx, &Instruction::Nop).ok();
    /// };
    /// recompiler.set_ebreak_callback(&mut my_callback);
    /// ```
    pub fn set_ebreak_callback(
        &mut self,
        callback: &'cb mut (dyn EbreakCallback<Context, E, F> + 'ctx),
    ) {
        self.ebreak_callback = Some(callback);
    }

    /// Clear the EBREAK callback
    ///
    /// Removes any previously set EBREAK callback.
    pub fn clear_ebreak_callback(&mut self) {
        self.ebreak_callback = None;
    }

    /// Set an address mapping callback for paging support
    ///
    /// When a mapper is set, it will be invoked for every memory load/store operation
    /// to translate virtual addresses to physical addresses. The callback receives the
    /// virtual address on the WebAssembly stack and should leave the physical address
    /// on the stack.
    ///
    /// See PAGING.md for detailed documentation on the paging system.
    ///
    /// # Arguments
    /// * `callback` - A mutable reference to a closure or function that performs address translation
    ///
    /// # Example
    /// ```ignore
    /// # use speet_riscv::{RiscVRecompiler, MapperContext};
    /// # use wasm_encoder::Instruction;
    ///
    /// let mut ctx = ();
    /// let mut my_mapper = |ctx: &mut MapperContext<_, _>| {
    ///     // Example: Simple page table lookup
    ///     // Input: virtual address on stack
    ///     // Output: physical address on stack
    ///     Ok(())
    /// };
    /// recompiler.set_mapper_callback(&mut my_mapper);
    /// ```
    pub fn set_mapper_callback(
        &mut self,
        callback: &'cb mut (dyn MapperCallback<Context, E, F> + 'ctx),
    ) {
        self.mapper_callback = Some(callback);
    }

    /// Clear the address mapping callback
    ///
    /// Removes any previously set mapper callback, returning to identity mapping.
    pub fn clear_mapper_callback(&mut self) {
        self.mapper_callback = None;
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

    /// Enable or disable speculative call lowering for ABI-compliant calls
    ///
    /// When enabled, RISC-V `jal x1, <offset>` and `jalr x1, rs2, <offset>` instructions
    /// (standard ABI-compliant function calls that save the return address to the `ra`
    /// register) are lowered to native WebAssembly `call` instructions instead of jumps.
    ///
    /// The call is wrapped in a try-catch block that validates the return location.
    /// If the callee returns to an unexpected location (e.g., due to stack manipulation),
    /// an exception is thrown and caught by the Reactor's escape handler, which then
    /// resumes execution at the correct guest location.
    ///
    /// This optimization allows the WebAssembly engine to use its native call stack,
    /// improving performance and debuggability. However, it requires an escape tag
    /// to be configured via the constructor.
    ///
    /// See `SPECULATIVE_CALLS.md` in the yecta crate for detailed documentation.
    ///
    /// # Arguments
    /// * `enable` - Whether to enable speculative call lowering
    ///
    /// # Panics
    /// Does not panic, but speculative calls will have no effect if no escape tag
    /// is configured (the recompiler will fall back to jump-based control flow).
    pub fn set_speculative_calls(&mut self, enable: bool) {
        self.enable_speculative_calls = enable;
    }

    /// Check if speculative call lowering is enabled
    pub fn is_speculative_calls_enabled(&self) -> bool {
        self.enable_speculative_calls
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
            (32, int_type),        // x0-x31
            (32, ValType::F64),    // f0-f31 (using F64 for both F and D with NaN-boxing)
            (1, ValType::I32),     // PC
            (num_temps, int_type), // Temporary registers (match integer register type)
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
    fn emit_imm(&mut self, ctx: &mut Context, imm: Imm) -> Result<(), E> {
        if self.enable_rv64 {
            // Sign-extend the 32-bit immediate to 64 bits
            self.reactor
                .feed(ctx, &Instruction::I64Const(imm.as_i32() as i64))
        } else {
            self.reactor.feed(ctx, &Instruction::I32Const(imm.as_i32()))
        }
    }

    /// Emit an integer constant (i32 or i64 depending on RV64 mode)
    fn emit_int_const(&mut self, ctx: &mut Context, value: i32) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Const(value as i64))
        } else {
            self.reactor.feed(ctx, &Instruction::I32Const(value))
        }
    }

    /// Emit an add instruction (I32Add or I64Add depending on RV64 mode)
    fn emit_add(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Add)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Add)
        }
    }

    /// Emit a sub instruction (I32Sub or I64Sub depending on RV64 mode)
    fn emit_sub(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Sub)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Sub)
        }
    }

    /// Emit a multiply instruction (I32Mul or I64Mul depending on RV64 mode)
    fn emit_mul(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Mul)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Mul)
        }
    }

    /// Emit a logical and instruction (I32And or I64And depending on RV64 mode)
    fn emit_and(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64And)
        } else {
            self.reactor.feed(ctx, &Instruction::I32And)
        }
    }

    /// Emit a logical or instruction (I32Or or I64Or depending on RV64 mode)
    fn emit_or(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Or)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Or)
        }
    }

    /// Emit a logical xor instruction (I32Xor or I64Xor depending on RV64 mode)
    fn emit_xor(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Xor)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Xor)
        }
    }

    /// Emit a shift left instruction (I32Shl or I64Shl depending on RV64 mode)
    fn emit_shl(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64Shl)
        } else {
            self.reactor.feed(ctx, &Instruction::I32Shl)
        }
    }

    /// Emit a logical shift right instruction (I32ShrU or I64ShrU depending on RV64 mode)
    fn emit_shr_u(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64ShrU)
        } else {
            self.reactor.feed(ctx, &Instruction::I32ShrU)
        }
    }

    /// Emit an arithmetic shift right instruction (I32ShrS or I64ShrS depending on RV64 mode)
    fn emit_shr_s(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.enable_rv64 {
            self.reactor.feed(ctx, &Instruction::I64ShrS)
        } else {
            self.reactor.feed(ctx, &Instruction::I32ShrS)
        }
    }

    /// Perform a jump to a target PC using yecta's jump API
    fn jump_to_pc(&mut self, ctx: &mut Context, target_pc: u64, params: u32) -> Result<(), E> {
        let target_func = self.pc_to_func_idx(target_pc);
        self.reactor.jmp(ctx, target_func, params)
    }

    /// NaN-box a single-precision float value for storage in a double-precision register
    ///
    /// RISC-V Specification Quote:
    /// "When multiple floating-point precisions are supported, then valid values of
    /// narrower n-bit types, n < FLEN, are represented in the lower n bits of an
    /// FLEN-bit NaN value, with the upper bits all 1s. We call this a NaN-boxed value."
    ///
    /// For F32 values in F64 registers: set upper 32 bits to all 1s
    fn nan_box_f32(&mut self, ctx: &mut Context) -> Result<(), E> {
        // Convert F32 to I32, then to I64, OR with 0xFFFFFFFF00000000, reinterpret as F64
        self.reactor.feed(ctx, &Instruction::I32ReinterpretF32)?;
        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
        self.reactor
            .feed(ctx, &Instruction::I64Const(0xFFFFFFFF00000000_u64 as i64))?;
        self.reactor.feed(ctx, &Instruction::I64Or)?;
        self.reactor.feed(ctx, &Instruction::F64ReinterpretI64)?;
        Ok(())
    }

    /// Unbox a NaN-boxed single-precision value from a double-precision register
    ///
    /// Extract the F32 value from the lower 32 bits of the NaN-boxed F64 value
    fn unbox_f32(&mut self, ctx: &mut Context) -> Result<(), E> {
        // Reinterpret F64 as I64, wrap to I32 (takes lower 32 bits), reinterpret as F32
        self.reactor.feed(ctx, &Instruction::I64ReinterpretF64)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::F32ReinterpretI32)?;
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
    fn emit_mulh_signed(&mut self, ctx: &mut Context, src1: u32, src2: u32) -> Result<(), E> {
        // Load src1 and src2 to locals for reuse
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67; // for accumulating middle terms

        self.reactor.feed(ctx, &Instruction::LocalGet(src1))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::LocalGet(src2))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_b))?;

        // Start with a_hi * b_hi
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // a_hi (sign-extended)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // b_hi (sign-extended)
        self.reactor.feed(ctx, &Instruction::I64Mul)?; // a_hi * b_hi

        // Compute middle term: a_hi * b_lo (full 64-bit result)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // a_hi
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // b_lo
        self.reactor.feed(ctx, &Instruction::I64Mul)?; // a_hi * b_lo (64-bit result)
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?; // save for carry computation

        // Add high 32 bits of (a_hi * b_lo) to result
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // arithmetic shift for signed
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        // Compute other middle term: a_lo * b_hi (full 64-bit result)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // a_lo
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // b_hi
        self.reactor.feed(ctx, &Instruction::I64Mul)?; // a_lo * b_hi (64-bit result)

        // Add it to the middle term accumulator for carry calculation
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Add)?; // sum of middle terms (low parts)
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?;

        // Add high 32 bits of the summed middle terms
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // arithmetic shift
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        Ok(())
    }

    /// Helper to compute high 64 bits of unsigned 64x64 -> 128-bit multiplication
    fn emit_mulh_unsigned(&mut self, ctx: &mut Context, src1: u32, src2: u32) -> Result<(), E> {
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67;

        self.reactor.feed(ctx, &Instruction::LocalGet(src1))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::LocalGet(src2))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_b))?;

        // Start with a_hi * b_hi (all unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // a_hi (unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(ctx, &Instruction::I64Mul)?;

        // Compute middle term: a_hi * b_lo
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // a_hi
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // b_lo
        self.reactor.feed(ctx, &Instruction::I64Mul)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?;

        // Add high 32 bits of (a_hi * b_lo)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        // Compute other middle term: a_lo * b_hi
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // a_lo
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // b_hi
        self.reactor.feed(ctx, &Instruction::I64Mul)?;

        // Add to middle term for carry calculation
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?;

        // Add high 32 bits of summed middle terms
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        Ok(())
    }

    /// Helper to compute high 64 bits of signed-unsigned 64x64 -> 128-bit multiplication
    fn emit_mulh_signed_unsigned(
        &mut self,
        ctx: &mut Context,
        src1: u32,
        src2: u32,
    ) -> Result<(), E> {
        let temp_a = 65;
        let temp_b = 66;
        let temp_mid = 67;

        self.reactor.feed(ctx, &Instruction::LocalGet(src1))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::LocalGet(src2))?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_b))?;

        // src1 is signed, src2 is unsigned

        // Start with a_hi * b_hi (a_hi signed, b_hi unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // a_hi (signed)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(ctx, &Instruction::I64Mul)?;

        // Compute middle term: a_hi * b_lo (a_hi signed, b_lo unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?; // a_hi (signed)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // b_lo
        self.reactor.feed(ctx, &Instruction::I64Mul)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?;

        // Add high 32 bits of (a_hi * b_lo) - use signed shift
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        // Compute other middle term: a_lo * b_hi (a_lo unsigned, b_hi unsigned)
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_a))?;
        self.reactor.feed(ctx, &Instruction::I64Const(0xFFFFFFFF))?;
        self.reactor.feed(ctx, &Instruction::I64And)?; // a_lo
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_b))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?; // b_hi (unsigned)
        self.reactor.feed(ctx, &Instruction::I64Mul)?;

        // Add to middle term for carry calculation
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(temp_mid))?;

        // Add high 32 bits of summed middle terms - use signed shift
        self.reactor.feed(ctx, &Instruction::LocalGet(temp_mid))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrS)?;
        self.reactor.feed(ctx, &Instruction::I64Add)?;

        Ok(())
    }

    /// Finalize the function
    pub fn seal(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.reactor.seal(ctx, &Instruction::Unreachable)
    }

    /// Get the reactor (consumes self)
    pub fn into_reactor(self) -> Reactor<Context, E, F> {
        self.reactor
    }
}

impl<'cb, 'ctx, Context, E, F: InstructionSink<Context, E>> Default
    for RiscVRecompiler<'cb, 'ctx, Context, E, F>
{
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

/// Helper for specifying page table base address location
pub enum PageTableBase {
    /// Static constant address
    Constant(u64),
    /// Runtime value stored in a WebAssembly local variable
    Local(u32),
    /// Runtime value stored in a WebAssembly global variable  
    Global(u32),
}

impl From<u64> for PageTableBase {
    fn from(c: u64) -> Self {
        PageTableBase::Constant(c)
    }
}

impl PageTableBase {
    /// Emit instructions to get the page table base value onto the stack
    fn emit_load<Context, E, F: InstructionSink<Context, E>>(
        &self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
        use_i64: bool,
    ) -> Result<(), E> {
        match self {
            PageTableBase::Constant(addr) => {
                if use_i64 {
                    callback_ctx.emit(ctx, &Instruction::I64Const(*addr as i64))?;
                } else {
                    callback_ctx.emit(ctx, &Instruction::I32Const(*addr as i32))?;
                }
            }
            PageTableBase::Local(idx) => {
                callback_ctx.emit(ctx, &Instruction::LocalGet(*idx))?;
            }
            PageTableBase::Global(idx) => {
                callback_ctx.emit(ctx, &Instruction::GlobalGet(*idx))?;
            }
        }
        Ok(())
    }
}

/// Standard page table mapper for 64KB single-level paging
///
/// This helper generates WebAssembly instructions to translate virtual addresses
/// using a flat page table stored in WebAssembly memory.
///
/// # Page Table Format
/// - Each entry is 8 bytes (i64) containing the physical page base address
/// - Entry address = page_table_base + (page_num * 8)
/// - Page number = vaddr >> 16 (bits 63:16)
/// - Page offset = vaddr & 0xFFFF (bits 15:0)
///
/// # Arguments
/// - `ctx`: Callback context for emitting WebAssembly instructions
/// - `page_table_base`: Page table base (constant u64, local, or global)
/// - `memory_index`: Memory index to use for loads (usually 0)
/// - `use_i64`: Whether to use i64 addressing (true for memory64/RV64)
///
/// # Stack State
/// - Input: Virtual address (i64 or i32) - must be saved to local 66 before calling
/// - Output: Physical address (same type as input)
pub fn standard_page_table_mapper<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    callback_ctx: &mut CallbackContext<Context, E, F>,
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        // 64-bit implementation
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?; // vaddr

        // page_num = vaddr >> 16
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;

        // pte_addr = pt_base + (page_num * 8)
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        pt_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;

        // page_pointer = [pte_addr]
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?;

        // security_index = page_pointer & 0xFFFF
        callback_ctx.emit(ctx, &Instruction::LocalTee(67))?; // Store page_pointer in temp local 67
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;

        // page_base_low48 = page_pointer >> 16
        callback_ctx.emit(ctx, &Instruction::LocalGet(67))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(68))?; // Store page_base_low48 in temp local 68

        // sec_entry_addr = sec_dir_base + (security_index * 4)
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;

        // sec_entry = [sec_entry_addr]
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;

        // page_base_top16 = sec_entry >> 16
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        // phys_page_base = (page_base_top16 << 48) | page_base_low48
        callback_ctx.emit(ctx, &Instruction::I64Const(48))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Or)?;

        // page_offset = vaddr & 0xFFFF
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;

        // phys_addr = phys_page_base + page_offset
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
    } else {
        // 32-bit implementation (falls back to old logic for now)
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;

        callback_ctx.emit(ctx, &Instruction::I32Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I32ShrU)?;

        callback_ctx.emit(ctx, &Instruction::I32Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I32Shl)?;

        pt_base.emit_load(ctx, callback_ctx, false)?;
        callback_ctx.emit(ctx, &Instruction::I32Add)?;

        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;

        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I32Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I32And)?;

        callback_ctx.emit(ctx, &Instruction::I32Add)?;
    }

    Ok(())
}

/// Multi-level page table mapper for 64KB pages
///
/// This helper generates WebAssembly instructions for a 3-level page table structure.
/// Each level uses 16-bit indices, supporting the full 64-bit address space.
///
/// # Page Table Structure
/// - Level 3 (top): Indexed by bits [63:48]
/// - Level 2: Indexed by bits [47:32]
/// - Level 1 (leaf): Indexed by bits [31:16], contains physical page bases
/// - Page offset: bits [15:0]
///
/// # Arguments
/// - `ctx`: Callback context for emitting WebAssembly instructions
/// - `l3_table_base`: Base address of level 3 page table (constant u64, local, or global)
/// - `memory_index`: Memory index to use for loads (usually 0)
/// - `use_i64`: Whether to use i64 addressing
///
/// # Stack State
/// - Input: Virtual address (i64 or i32) - must be saved to local 66 before calling
/// - Output: Physical address (same type as input)
pub fn multilevel_page_table_mapper<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    callback_ctx: &mut CallbackContext<Context, E, F>,
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        // Level 3
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(48))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        l3_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?;

        // Level 2
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(32))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?;

        // Level 1
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?; // page_pointer

        // Security and final address construction
        callback_ctx.emit(ctx, &Instruction::LocalTee(67))?; // page_pointer
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?; // security_index
        callback_ctx.emit(ctx, &Instruction::LocalGet(67))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(68))?; // page_base_low48
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64Const(48))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(48))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Or)?;

        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
    } else {
        // simplified for 32-bit, only single level supported in this path
        multilevel_page_table_mapper_32(
            ctx,
            callback_ctx,
            l3_base,
            sec_dir_base,
            memory_index,
            false,
        )?;
    }

    Ok(())
}

/// Single-level page table mapper with 32-bit physical addresses
///
/// This variant uses 4-byte page table entries for 32-bit physical addresses,
/// supporting up to 4 GiB of physical memory while maintaining 64-bit virtual addresses.
///
/// # Arguments
/// - `ctx`: Callback context for emitting WebAssembly instructions
/// - `page_table_base`: Base address of page table (constant u64, local, or global)
/// - `memory_index`: Memory index to use
/// - `use_i64`: Whether to use i64 addressing (true for memory64/RV64)
///
/// # Stack State
/// - Input: Virtual address (i64 or i32) - must be saved to local 66 before calling
/// - Output: Physical address (same type as input)
pub fn standard_page_table_mapper_32<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    callback_ctx: &mut CallbackContext<Context, E, F>,
    page_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let pt_base = page_table_base.into();
    let sec_dir_base = security_directory_base.into();
    if use_i64 {
        // 64-bit vaddr, 32-bit paddr
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;

        // page_num = vaddr >> 16
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;

        // pte_addr = pt_base + (page_num * 4)
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        pt_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;

        // page_pointer = [pte_addr] (u32)
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::LocalTee(67))?; // temp local 67 for page_pointer
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(68))?; // temp local 68 for page_pointer as u64

        // security_index = page_pointer & 0xFF
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;

        // page_base_low24 = page_pointer >> 8
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(8))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(69))?; // temp local 69 for page_base_low24

        // sec_entry_addr = sec_dir_base + (security_index * 4)
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;

        // sec_entry = [sec_entry_addr] (u32)
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;

        // page_base_top8 = sec_entry >> 24
        callback_ctx.emit(ctx, &Instruction::I64Const(24))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        // phys_page_base = (page_base_top8 << 24) | page_base_low24
        callback_ctx.emit(ctx, &Instruction::I64Const(24))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(69))?;
        callback_ctx.emit(ctx, &Instruction::I64Or)?;

        // page_offset = vaddr & 0xFFFF
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;

        callback_ctx.emit(ctx, &Instruction::I64Add)?;
    } else {
        // 32-bit vaddr, 32-bit paddr
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;

        callback_ctx.emit(ctx, &Instruction::I32Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I32ShrU)?;

        callback_ctx.emit(ctx, &Instruction::I32Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I32Shl)?;
        pt_base.emit_load(ctx, callback_ctx, false)?;
        callback_ctx.emit(ctx, &Instruction::I32Add)?;

        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::LocalTee(67))?; // page_pointer

        // security_index = page_pointer & 0xFF
        callback_ctx.emit(ctx, &Instruction::I32Const(0xFF))?;
        callback_ctx.emit(ctx, &Instruction::I32And)?;

        // page_base_low24 = page_pointer >> 8
        callback_ctx.emit(ctx, &Instruction::LocalGet(67))?;
        callback_ctx.emit(ctx, &Instruction::I32Const(8))?;
        callback_ctx.emit(ctx, &Instruction::I32ShrU)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(68))?; // page_base_low24

        // sec_entry_addr = sec_dir_base + (security_index * 4)
        callback_ctx.emit(ctx, &Instruction::I32Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I32Shl)?;
        sec_dir_base.emit_load(ctx, callback_ctx, false)?;
        callback_ctx.emit(ctx, &Instruction::I32Add)?;

        // sec_entry = [sec_entry_addr]
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;

        // page_base_top8 = sec_entry >> 24
        callback_ctx.emit(ctx, &Instruction::I32Const(24))?;
        callback_ctx.emit(ctx, &Instruction::I32ShrU)?;

        // phys_page_base = (page_base_top8 << 24) | page_base_low24
        callback_ctx.emit(ctx, &Instruction::I32Const(24))?;
        callback_ctx.emit(ctx, &Instruction::I32Shl)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I32Or)?;

        // page_offset = vaddr & 0xFFFF
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I32Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I32And)?;

        callback_ctx.emit(ctx, &Instruction::I32Add)?;
    }

    Ok(())
}

/// Multi-level page table mapper with 32-bit physical addresses
///
/// This variant uses 4-byte page table entries for 32-bit physical addresses,
/// supporting up to 4 GiB of physical memory in a 3-level page table structure.
///
/// # Arguments
/// - `ctx`: Callback context for emitting WebAssembly instructions
/// - `l3_table_base`: Base address of level 3 page table
/// - `memory_index`: Memory index to use
/// - `use_i64`: Whether to use i64 addressing
///
/// # Stack State
/// - Input: Virtual address (i64 or i32) - must be saved to local 66 before calling
/// - Output: Physical address (same type as input)
pub fn multilevel_page_table_mapper_32<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    callback_ctx: &mut CallbackContext<Context, E, F>,
    l3_table_base: impl Into<PageTableBase>,
    security_directory_base: impl Into<PageTableBase>,
    memory_index: u32,
    use_i64: bool,
) -> Result<(), E> {
    let l3_base = l3_table_base.into();
    let sec_dir_base = security_directory_base.into();

    if use_i64 {
        // Level 3
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(48))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        l3_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;

        // Level 2
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(32))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;

        // Level 1
        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(16))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(2))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I32Load(wasm_encoder::MemArg {
                offset: 0,
                align: 2,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::LocalTee(67))?; // page_pointer
        callback_ctx.emit(ctx, &Instruction::I64ExtendI32U)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(68))?;

        // Security and final address construction
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(68))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(8))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::LocalSet(69))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(3))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        sec_dir_base.emit_load(ctx, callback_ctx, true)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
        callback_ctx.emit(
            ctx,
            &Instruction::I64Load(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index,
            }),
        )?;
        callback_ctx.emit(ctx, &Instruction::I64Const(56))?;
        callback_ctx.emit(ctx, &Instruction::I64ShrU)?;
        callback_ctx.emit(ctx, &Instruction::I64Const(24))?;
        callback_ctx.emit(ctx, &Instruction::I64Shl)?;
        callback_ctx.emit(ctx, &Instruction::LocalGet(69))?;
        callback_ctx.emit(ctx, &Instruction::I64Or)?;

        callback_ctx.emit(ctx, &Instruction::LocalGet(66))?;
        callback_ctx.emit(ctx, &Instruction::I64Const(0xFFFF))?;
        callback_ctx.emit(ctx, &Instruction::I64And)?;
        callback_ctx.emit(ctx, &Instruction::I64Add)?;
    } else {
        // 32-bit vaddr, 32-bit paddr
        standard_page_table_mapper_32(
            ctx,
            callback_ctx,
            l3_base,
            sec_dir_base,
            memory_index,
            false,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rv_asm::{Inst, Xlen};
    use yecta::TagIdx;

    #[test]
    fn test_recompiler_creation() {
        let _recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        // Just ensure it can be created without panicking
    }

    #[test]
    fn test_addi_instruction() {
        // Test ADDI instruction: addi x1, x0, 42
        // Use base_pc to offset high addresses
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(42),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_add_instruction() {
        // Test ADD instruction: add x3, x1, x2
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_load_instruction() {
        // Test LW instruction: lw x1, 0(x2)

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::Lw {
            offset: rv_asm::Imm::new_i32(0),
            dest: rv_asm::Reg(1),
            base: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_store_instruction() {
        // Test SW instruction: sw x1, 4(x2)
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
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
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_branch_instruction() {
        // Test BEQ instruction: beq x1, x2, offset

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::Beq {
            offset: rv_asm::Imm::new_i32(8),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_mul_instruction() {
        // Test MUL instruction from M extension

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::Mul {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_fadd_instruction() {
        // Test FADD.S instruction from F extension
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let inst = Inst::FaddS {
            rm: rv_asm::RoundingMode::RoundToNearestTiesToEven,
            dest: rv_asm::FReg(1),
            src1: rv_asm::FReg(2),
            src2: rv_asm::FReg(3),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_decode_and_translate() {
        // Test decoding a real instruction and translating it
        // This is "addi a0, a0, 0" which is a common NOP-like instruction
        let instruction_bytes: u32 = 0x00050513;
        let (inst, is_compressed) = Inst::decode(instruction_bytes, Xlen::Rv32).unwrap();
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, is_compressed, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_multiple_instructions() {
        // Test translating multiple instructions in sequence
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        // addi x1, x0, 5
        let inst1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst1, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
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
                .translate_instruction(&mut ctx, &inst2, 0x1004, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
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
                .translate_instruction(&mut ctx, &inst3, 0x1008, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_translate_from_bytes() {
        // Test translating from raw bytecode

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        // Simple program: addi x1, x0, 5 (0x00500093)
        let bytes = [0x93, 0x00, 0x50, 0x00]; // Little-endian

        let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, Xlen::Rv32, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4); // Should have translated 4 bytes
    }

    #[test]
    fn test_translate_compressed_from_bytes() {
        // Test translating compressed instructions from bytes

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        // c.addi x1, 5 (0x0095) - compressed instruction
        let bytes = [0x95, 0x00];

        let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, Xlen::Rv32, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2); // Should have translated 2 bytes
    }

    #[test]
    fn test_register_mapping() {
        // Test that register indices map correctly
        assert_eq!(
            RiscVRecompiler::<(), Infallible, Function>::reg_to_local(rv_asm::Reg(0)),
            0
        );
        assert_eq!(
            RiscVRecompiler::<(), Infallible, Function>::reg_to_local(rv_asm::Reg(31)),
            31
        );
        assert_eq!(
            RiscVRecompiler::<(), Infallible, Function>::freg_to_local(rv_asm::FReg(0)),
            32
        );
        assert_eq!(
            RiscVRecompiler::<(), Infallible, Function>::freg_to_local(rv_asm::FReg(31)),
            63
        );
        assert_eq!(RiscVRecompiler::<(), Infallible, Function>::pc_local(), 64);
    }

    #[test]
    fn test_hint_tracking_disabled_by_default() {
        // HINT tracking should be disabled by default

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        // Translate a HINT instruction: addi x0, x0, 1
        let hint_inst = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint_inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );

        // Should not have collected any hints since tracking is disabled
        assert_eq!(recompiler.get_hints().len(), 0);
    }

    #[cfg(false)]
    #[test]
    fn test_hint_tracking_enabled() {
        // Test HINT tracking when explicitly enabled

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_all_config(
                Pool { table: (), ty: () },
                escape_tag,
                base_pc,
                base_func_offset,
                track_hints,
                enable_rv64,
                use_memory64,
            );
        let mut ctx = ();
        // Translate a HINT instruction: addi x0, x0, 1
        let hint1 = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint1, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
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
                .translate_instruction(&mut ctx, &hint2, 0x1004, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
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
    #[cfg(false)]
    #[test]
    fn test_hint_vs_regular_addi() {
        // Test that regular ADDI instructions are not tracked as HINTs

        /// let mut ctx = ();
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
                .translate_instruction(&mut ctx, &hint, 0x1004, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );

        // Should only have the HINT
        let hints = recompiler.get_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].pc, 0x1004);
        assert_eq!(hints[0].value, 1);
    }
    #[cfg(false)]
    #[test]
    fn test_hint_clear() {
        // Test clearing collected HINTs

        /// let mut ctx = ();
        // Collect some hints
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
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
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        let hint = Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        };

        // Initially disabled
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 0);

        // Enable tracking
        recompiler.set_hint_tracking(true);
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint, 0x1004, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 1);

        // Disable tracking (should clear existing hints)
        recompiler.set_hint_tracking(false);
        assert_eq!(recompiler.get_hints().len(), 0);
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &hint, 0x1008, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
        assert_eq!(recompiler.get_hints().len(), 0);
    }
    #[cfg(false)]
    #[test]
    fn test_hint_from_rv_corpus_pattern() {
        // Test the actual pattern used in rv-corpus test files

        /// let mut ctx = ();
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
                    .translate_instruction(&mut ctx, &hint, pc, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
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
    #[cfg(false)]
    #[test]
    fn test_hint_callback_basic() {
        // Test basic callback functionality
        use alloc::vec::Vec;

        let mut collected = Vec::new();

        {
            /// let mut ctx = ();
            let mut callback =
                |hint: &HintInfo, _ctx: &mut HintContext<'_, (), Infallible, Function>| {
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
    #[cfg(false)]
    #[test]
    fn test_hint_callback_with_tracking() {
        // Test that callback and tracking work together
        use alloc::vec::Vec;

        let mut callback_hints = Vec::new();
        let tracked_hints_result;

        {
            /// let mut ctx = ();
            let mut callback =
                |hint: &HintInfo, _ctx: &mut HintContext<'_, (), Infallible, Function>| {
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
                    .translate_instruction(&mut ctx, &hint, 0x2000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
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
    #[cfg(false)]
    #[test]
    fn test_hint_callback_clear() {
        // Test clearing the callback
        use alloc::vec::Vec;

        let mut collected = Vec::new();

        {
            /// let mut ctx = ();
            let mut callback =
                |hint: &HintInfo, _ctx: &mut HintContext<'_, (), Infallible, Function>| {
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
                    .translate_instruction(&mut ctx, &hint, 0x1000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            // Tracking is disabled by default
            let mut callback =
                |hint: &HintInfo,
                 _: &mut (),
                 _ctx: &mut HintContext<'_, (), Infallible, Function>| {
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
                    .translate_instruction(&mut ctx, &hint, 0x1000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |_hint: &HintInfo,
                 _: &mut (),
                 _ctx: &mut HintContext<'_, (), Infallible, Function>| {
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
                    .translate_instruction(
                        &mut ctx,
                        &regular_addi,
                        0x1000,
                        IsCompressed::No,
                        &mut |a| { Function::new(a.collect::<Vec<_>>()) }
                    )
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |hint: &HintInfo,
                 _: &mut (),
                 ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    hint_values.push(hint.value);
                    // Generate a NOP instruction for each HINT
                    // callback_ctx.emit(ctx, &Instruction::Nop).ok();
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
                            &mut ctx,
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |ecall: &EcallInfo,
                 _: &mut (),
                 ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ecall_pcs.push(ecall.pc);
                };

            recompiler.set_ecall_callback(&mut callback);

            // Translate an ECALL instruction
            let ecall = Inst::Ecall;
            assert!(
                recompiler
                    .translate_instruction(&mut ctx, &ecall, 0x2000, IsCompressed::No, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |ebreak: &EbreakInfo,
                 _: &mut (),
                 _ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ebreak_pcs.push(ebreak.pc);
                };

            recompiler.set_ebreak_callback(&mut callback);

            // Translate an EBREAK instruction
            let ebreak = Inst::Ebreak;
            assert!(
                recompiler
                    .translate_instruction(&mut ctx, &ebreak, 0x3000, IsCompressed::No, &mut |a| {
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |_ecall: &EcallInfo,
                 _: &mut (),
                 ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ecall_count += 1;
                    // Generate a NOP instruction for each ECALL
                    // callback_ctx.emit(ctx, &Instruction::Nop).ok();
                };

            recompiler.set_ecall_callback(&mut callback);

            // Translate multiple ECALL instructions
            for i in 0..3 {
                let ecall = Inst::Ecall;
                let pc = 0x1000 + (i * 4);
                assert!(
                    recompiler
                        .translate_instruction(&mut ctx, &ecall, pc, IsCompressed::No, &mut |a| {
                            Function::new(a.collect::<Vec<_>>())
                        })
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut callback =
                |_ebreak: &EbreakInfo,
                 _: &mut (),
                 ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ebreak_count += 1;
                    // Generate a NOP instruction for each EBREAK
                    // callback_ctx.emit(ctx, &Instruction::Nop).ok();
                };

            recompiler.set_ebreak_callback(&mut callback);

            // Translate multiple EBREAK instructions
            for i in 0..3 {
                let ebreak = Inst::Ebreak;
                let pc = 0x2000 + (i * 4);
                assert!(
                    recompiler
                        .translate_instruction(&mut ctx, &ebreak, pc, IsCompressed::No, &mut |a| {
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
            let mut recompiler =
                RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
            let mut ctx = ();
            let mut ecall_cb =
                |_ecall: &EcallInfo,
                 _: &mut (),
                 _ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ecall_count += 1;
                };

            let mut ebreak_cb =
                |_ebreak: &EbreakInfo,
                 _: &mut (),
                 _ctx: &mut HintContext<'_, (), Infallible, Function>| {
                    ebreak_count += 1;
                };

            recompiler.set_ecall_callback(&mut ecall_cb);
            recompiler.set_ebreak_callback(&mut ebreak_cb);

            // First ECALL and EBREAK should invoke callbacks
            assert!(
                recompiler
                    .translate_instruction(
                        &mut ctx,
                        &Inst::Ecall,
                        0x1000,
                        IsCompressed::No,
                        &mut |a| { Function::new(a.collect::<Vec<_>>()) }
                    )
                    .is_ok()
            );
            assert!(
                recompiler
                    .translate_instruction(
                        &mut ctx,
                        &Inst::Ebreak,
                        0x1004,
                        IsCompressed::No,
                        &mut |a| { Function::new(a.collect::<Vec<_>>()) }
                    )
                    .is_ok()
            );

            // Clear callbacks
            recompiler.clear_ecall_callback();
            recompiler.clear_ebreak_callback();

            // Second ECALL and EBREAK should not invoke callbacks (will use default behavior)
            assert!(
                recompiler
                    .translate_instruction(
                        &mut ctx,
                        &Inst::Ecall,
                        0x1008,
                        IsCompressed::No,
                        &mut |a| { Function::new(a.collect::<Vec<_>>()) }
                    )
                    .is_ok()
            );
            assert!(
                recompiler
                    .translate_instruction(
                        &mut ctx,
                        &Inst::Ebreak,
                        0x100c,
                        IsCompressed::No,
                        &mut |a| { Function::new(a.collect::<Vec<_>>()) }
                    )
                    .is_ok()
            );

            // Drop recompiler
        }

        // Verify callbacks were invoked only once
        assert_eq!(ecall_count, 1);
        assert_eq!(ebreak_count, 1);
    }
    #[cfg(false)]
    #[test]
    fn test_rv64_instructions() {
        // Test RV64 instructions when RV64 is enabled
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_all_config(
                pool,
                escape_tag,
                base_pc,
                base_func_offset,
                track_hints,
                enable_rv64,
                use_memory64,
            );
        let mut ctx = ();
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

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
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
                .translate_instruction(&mut ctx, &addiw, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }
    #[cfg(false)]
    #[test]
    fn test_rv64_with_memory64() {
        // Test RV64 with memory64 enabled

        /// let mut ctx = ();
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
    #[cfg(false)]
    #[test]
    fn test_rv64_mulh_instructions() {
        // Test RV64 multiply-high instructions (Mulh, Mulhu, Mulhsu)

        /// let mut ctx = ();
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
    #[cfg(false)]
    #[test]
    fn test_rv64_float_conversions() {
        // Test RV64 floating-point conversion instructions

        /// let mut ctx = ();
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
                .translate_instruction(&fcvt_ls_x0, 0x1028, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_base_func_offset() {
        // Test that base_func_offset can be set and retrieved

        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();

        // Default should be 0
        assert_eq!(recompiler.base_func_offset(), 0);

        // Set to a new value
        recompiler.set_base_func_offset(15);
        assert_eq!(recompiler.base_func_offset(), 15);

        // Create with offset using new_with_all_config
        let mut recompiler2 =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_all_config(
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

    #[test]
    fn test_speculative_calls_disabled_by_default() {
        let recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        assert!(!recompiler.is_speculative_calls_enabled());
    }

    #[test]
    fn test_speculative_calls_toggle() {
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        
        // Initially disabled
        assert!(!recompiler.is_speculative_calls_enabled());
        
        // Enable
        recompiler.set_speculative_calls(true);
        assert!(recompiler.is_speculative_calls_enabled());
        
        // Disable again
        recompiler.set_speculative_calls(false);
        assert!(!recompiler.is_speculative_calls_enabled());
    }

    #[test]
    fn test_jal_with_speculative_calls_disabled() {
        // Test JAL instruction without speculative calls (original behavior)
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        let mut ctx = ();
        
        // jal x1, 0x100 (call subroutine at PC+0x100)
        let inst = Inst::Jal {
            offset: rv_asm::Imm::new_i32(0x100),
            dest: rv_asm::Reg(1), // ra = x1
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_jal_with_speculative_calls_enabled() {
        // Test JAL instruction with speculative calls enabled
        let mut recompiler = RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            Some(EscapeTag {
                tag: TagIdx(0),
                ty: TypeIdx(0),
            }),
            0x1000,
        );
        recompiler.set_speculative_calls(true);
        let mut ctx = ();
        
        // jal x1, 0x100 (call subroutine at PC+0x100)
        // This should use speculative call lowering since dest=x1 (ra)
        let inst = Inst::Jal {
            offset: rv_asm::Imm::new_i32(0x100),
            dest: rv_asm::Reg(1), // ra = x1
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_jal_non_abi_call_with_speculative_calls() {
        // Test JAL with dest != x1 (not an ABI-compliant call)
        // Should NOT use speculative calls even when enabled
        let mut recompiler = RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            Some(EscapeTag {
                tag: TagIdx(0),
                ty: TypeIdx(0),
            }),
            0x1000,
        );
        recompiler.set_speculative_calls(true);
        let mut ctx = ();
        
        // jal x5, 0x100 (jump with link to x5, not a standard call)
        let inst = Inst::Jal {
            offset: rv_asm::Imm::new_i32(0x100),
            dest: rv_asm::Reg(5), // t0, not ra
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_jalr_with_speculative_calls_enabled() {
        // Test JALR instruction with speculative calls enabled
        let mut recompiler = RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_config(
            Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            Some(EscapeTag {
                tag: TagIdx(0),
                ty: TypeIdx(0),
            }),
            0x1000,
        );
        recompiler.set_speculative_calls(true);
        let mut ctx = ();
        
        // jalr x1, x2, 0 (indirect call through register x2)
        let inst = Inst::Jalr {
            offset: rv_asm::Imm::new_i32(0),
            base: rv_asm::Reg(2),
            dest: rv_asm::Reg(1), // ra = x1
        };

        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }

    #[test]
    fn test_speculative_calls_requires_escape_tag() {
        // Test that speculative calls are not used when no escape tag is configured
        let mut recompiler =
            RiscVRecompiler::<'_, '_, (), Infallible, Function>::new_with_base_pc(0x1000);
        // Enable speculative calls but don't set escape tag
        recompiler.set_speculative_calls(true);
        let mut ctx = ();
        
        // jal x1, 0x100 - should fall back to non-speculative since no escape tag
        let inst = Inst::Jal {
            offset: rv_asm::Imm::new_i32(0x100),
            dest: rv_asm::Reg(1),
        };

        // Should still work, just uses non-speculative path
        assert!(
            recompiler
                .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
                    Function::new(a.collect::<Vec<_>>())
                })
                .is_ok()
        );
    }
}
