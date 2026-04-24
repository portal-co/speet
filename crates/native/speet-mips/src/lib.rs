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
//! - **Branch/Jump**: BEQ, BNE, BLEZ, BGTZ, BLTZ, BGEZ, J, JAL, JR, JALR
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
pub mod cfg;
use alloc::collections::BTreeMap;
use wax_core::build::InstructionSink;

use rabbitizer::{InstrId, Instruction, registers::GprO32};
use speet_ordering::{emit_fence, emit_load, emit_lr, emit_sc, emit_store};
use wasm_encoder::{Instruction as WasmInstruction, ValType};
use yecta::{
    EscapeTag, Fed, FuncIdx, LocalLayout, LocalPoolBackend, LocalSlot, Mark, Pool, Reactor,
    SlotAssigner, TableIdx, Target, TypeIdx, layout::{CellIdx, CellRegistry},
};
// Re-export the shared memory/mapper and ordering abstractions.
pub use speet_memory::{CallbackContext, MapperCallback};
pub use speet_ordering::{AtomicOpts, MemOrder, RmwOp, RmwWidth};
use speet_traps::{
    InstructionInfo, InstructionTrap, JumpInfo, JumpKind, JumpTrap, TrapAction, TrapConfig,
    insn::{ArchTag, InsnClass},
};

/// Branch operation types for conditional branches
#[derive(Debug, Clone, Copy)]
enum BranchOp {
    Eq,
    Ne,
    LeZ,
    GtZ,
    LtZ,
    GeZ,
}

// Shared snippet to compute table index for indirect jumps
// idx = ((reg_value & ~3) - base_pc) >> 2
struct TableIndexSnippet {
    rs_local: u32,
    base_pc: u32,
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for TableIndexSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &WasmInstruction::LocalGet(self.rs_local))?;
        sink.instruction(ctx, &WasmInstruction::I32Const(0xFFFFFFFC_u32 as i32))?;
        sink.instruction(ctx, &WasmInstruction::I32And)?;
        sink.instruction(ctx, &WasmInstruction::I32Const(self.base_pc as i32))?;
        sink.instruction(ctx, &WasmInstruction::I32Sub)?;
        sink.instruction(ctx, &WasmInstruction::I32Const(2))?;
        sink.instruction(ctx, &WasmInstruction::I32ShrU)?;
        Ok(())
    }
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for TableIndexSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &WasmInstruction::LocalGet(self.rs_local))?;
        sink.instruction(ctx, &WasmInstruction::I32Const(0xFFFFFFFC_u32 as i32))?;
        sink.instruction(ctx, &WasmInstruction::I32And)?;
        sink.instruction(ctx, &WasmInstruction::I32Const(self.base_pc as i32))?;
        sink.instruction(ctx, &WasmInstruction::I32Sub)?;
        sink.instruction(ctx, &WasmInstruction::I32Const(2))?;
        sink.instruction(ctx, &WasmInstruction::I32ShrU)?;
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

/// Trait for SYSCALL instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when SYSCALL
/// instructions are encountered during translation.
pub trait SyscallCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Process a SYSCALL instruction
    ///
    /// # Arguments
    /// * `syscall` - Information about the detected SYSCALL instruction
    /// * `ctx` - User context for passing external state
    /// * `callback_ctx` - Unified context for emitting WebAssembly instructions
    fn call(
        &mut self,
        syscall: &SyscallInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    );
}

/// Blanket implementation of SyscallCallback for FnMut closures
impl<Context, E, G: InstructionSink<Context, E>, F> SyscallCallback<Context, E, G> for F
where
    F: FnMut(&SyscallInfo, &mut Context, &mut CallbackContext<Context, E>),
{
    fn call(
        &mut self,
        syscall: &SyscallInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) {
        self(syscall, ctx, callback_ctx)
    }
}

/// Trait for BREAK instruction callbacks
///
/// This trait defines the interface for callbacks that are invoked when BREAK
/// instructions are encountered during translation.
pub trait BreakCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Process a BREAK instruction
    ///
    /// # Arguments
    /// * `break_info` - Information about the detected BREAK instruction
    /// * `ctx` - User context for passing external state
    /// * `callback_ctx` - Unified context for emitting WebAssembly instructions
    fn call(
        &mut self,
        break_info: &BreakInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    );
}

/// Blanket implementation of BreakCallback for FnMut closures
impl<Context, E, G: InstructionSink<Context, E>, F> BreakCallback<Context, E, G> for F
where
    F: FnMut(&BreakInfo, &mut Context, &mut CallbackContext<Context, E>),
{
    fn call(
        &mut self,
        break_info: &BreakInfo,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) {
        self(break_info, ctx, callback_ctx)
    }
}

/// MIPS to WebAssembly recompiler
///
/// This structure manages the translation of MIPS instructions to WebAssembly,
/// using the yecta reactor for control flow management.
///
/// Each instruction gets its own function, and control flow is managed through
/// jumps between these functions using the yecta reactor.
pub struct MipsRecompiler<
    'cb,
    'ctx,
    Context,
    E,
    F: InstructionSink<Context, E>,
> {
    /// Base PC address - subtracted from PC values to compute function indices
    base_pc: u32,
    /// Optional callback for SYSCALL instructions
    syscall_callback: Option<&'cb mut (dyn SyscallCallback<Context, E, F> + 'ctx)>,
    /// Optional callback for BREAK instructions
    break_callback: Option<&'cb mut (dyn BreakCallback<Context, E, F> + 'ctx)>,
    /// Optional callback for address mapping (paging support)
    mapper_callback: Option<&'cb mut (dyn MapperCallback<Context, E> + 'ctx)>,
    /// Whether to enable MIPS64 instruction support (disabled by default)
    enable_mips64: bool,
    /// Memory ordering mode for load/store emission.
    ///
    /// `MemOrder::Strong` (default) emits all stores eagerly.
    /// `MemOrder::Relaxed` emits stores via `feed_lazy`, letting the yecta
    /// reactor sink them to the latest control-flow boundary.  Only enable
    /// for MIPS binaries that conform to the weak memory model.
    mem_order: MemOrder,
    /// Optional atomic instruction substitution.
    ///
    /// When `use_atomic_insns` is true, integer load/store instructions are
    /// replaced with their wasm atomic equivalents.
    atomic_opts: AtomicOpts,
    /// Pluggable instruction-level and jump-level trap hooks.
    /// Total wasm function parameter count (recompiler params + trap params).
    /// Unified layout: arch params + trap params, then per-function locals.
    /// Mark placed after all param slots; used to rewind before each function.
    /// Registry mapping unique (function-type params, locals) combinations
    /// to [`CellIdx`] handles.  Populated on each `init_function` call.
    /// The [`CellIdx`] allocated for the most-recently initialised function.
    /// Slot for per-function GPR-type temp locals (num_temps of them).
    temps_slot: LocalSlot,
    /// Slot for the single load-address scratch local.
    addr_scratch_slot: LocalSlot,
    /// Slot for i32 pool locals.
    pool_i32_slot: LocalSlot,
    /// Slot for i64 pool locals.
    pool_i64_slot: LocalSlot,
    /// Optional slot assigner: controls which guest PCs receive function slots.
    slot_assigner: Option<alloc::boxed::Box<dyn SlotAssigner + Send + Sync>>,
}

impl<'cb, 'ctx, Context, E, F> MipsRecompiler<'cb, 'ctx, Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    /// Create a new MIPS recompiler instance with full configuration
    ///
    /// # Arguments
    /// * `pool` - Pool configuration for indirect calls
    /// * `escape_tag` - Optional exception tag for non-local control flow
    /// * `base_pc` - Base PC address to offset function indices
    /// * `enable_mips64` - Whether to enable MIPS64 instruction support
    pub fn new_with_full_config(
        base_pc: u32,
        enable_mips64: bool,
    ) -> Self
    {
        Self {
            base_pc,
            syscall_callback: None,
            break_callback: None,
            mapper_callback: None,
            enable_mips64,
            mem_order: MemOrder::Strong,
            atomic_opts: AtomicOpts::NONE,
            temps_slot: LocalSlot::default(),
            addr_scratch_slot: LocalSlot::default(),
            pool_i32_slot: LocalSlot::default(),
            pool_i64_slot: LocalSlot::default(),
            slot_assigner: None,
        }
    }

    pub fn new_with_all_config(
        base_pc: u32,
        _base_func_offset: u32,
        enable_mips64: bool,
    ) -> Self
    {
        Self::new_with_full_config(base_pc, enable_mips64)
    }

    /// Get the current base function offset.
    pub fn base_func_offset(&self, rctx: &dyn ReactorContext<Context, E, FnType = F>) -> u32 {
        rctx.base_func_offset()
    }

    /// Set the base function offset.
    pub fn set_base_func_offset(&mut self, rctx: &mut dyn ReactorContext<Context, E, FnType = F>, offset: u32) {
        rctx.set_base_func_offset(offset);
    }

    /// Create a new MIPS recompiler instance
    pub fn new_with_config(base_pc: u32) -> Self {
        Self::new_with_full_config(base_pc, false)
    }

    /// Create a new MIPS recompiler with default configuration (base_pc = 0)
    pub fn new() -> Self {
        Self::new_with_full_config(0, false)
    }

    /// Create a new MIPS recompiler with a specified base PC
    ///
    /// # Arguments
    /// * `base_pc` - Base PC address - this is subtracted from instruction PCs to compute function indices
    pub fn new_with_base_pc(base_pc: u32) -> Self {
        Self::new_with_config(base_pc)
    }

    /// Set a callback for SYSCALL instructions
    ///
    /// When a callback is set, it will be invoked immediately when a SYSCALL instruction
    /// is encountered during translation.
    pub fn set_syscall_callback(
        &mut self,
        callback: &'cb mut (dyn SyscallCallback<Context, E, F> + 'ctx),
    ) {
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
    pub fn set_break_callback(
        &mut self,
        callback: &'cb mut (dyn BreakCallback<Context, E, F> + 'ctx),
    ) {
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
    pub fn set_mapper_callback(
        &mut self,
        callback: &'cb mut (dyn MapperCallback<Context, E> + 'ctx),
    ) {
        self.mapper_callback = Some(callback);
    }

    /// Clear the address mapping callback
    pub fn clear_mapper_callback(&mut self) {
        self.mapper_callback = None;
    }

    /// Set the memory ordering mode for load/store emission.
    ///
    /// * [`MemOrder::Strong`] (default) — all stores are emitted eagerly.
    ///   `SYNC` instructions flush the lazy buffer and are otherwise no-ops.
    ///
    /// * [`MemOrder::Relaxed`] — stores are emitted via `feed_lazy`, letting
    ///   yecta sink them to the latest control-flow boundary.  Only use this
    ///   for MIPS binaries that conform to the weak memory model.
    pub fn set_mem_order(&mut self, order: MemOrder) {
        self.mem_order = order;
    }

    /// Return the current memory ordering mode.
    pub fn mem_order(&self) -> MemOrder {
        self.mem_order
    }

    /// Set the atomic instruction options.
    ///
    /// When `atomic.use_atomic_insns` is `true`, integer load/store
    /// instructions will be replaced with their wasm atomic equivalents.
    /// This is independent of [`MemOrder`].
    pub fn set_atomic_opts(&mut self, atomic: AtomicOpts) {
        self.atomic_opts = atomic;
    }

    /// Return the current atomic instruction options.
    pub fn atomic_opts(&self) -> AtomicOpts {
        self.atomic_opts
    }

    // ── Trap hooks ───────────────────────────────────────────────────────

    /// MIPS base parameter count: 32 GPRs + HI + LO + PC = 35.
    pub const BASE_PARAMS: u32 = 35;

    /// Install an instruction trap.
    ///
    /// Call [`setup_traps`](Self::setup_traps) after installing traps and
    /// before the first `translate_instruction` call.
    pub fn set_instruction_trap(
        &mut self,
        _trap: &'cb mut (dyn InstructionTrap<Context, E> + 'ctx),
    ) {
        unimplemented!("set trap on the Linker/ReactorContext provider directly")
    }

    pub fn clear_instruction_trap(&mut self) {}

    pub fn set_jump_trap(
        &mut self,
        _trap: &'cb mut (dyn JumpTrap<Context, E> + 'ctx),
    ) {
        unimplemented!("set trap on the Linker/ReactorContext provider directly")
    }

    pub fn clear_jump_trap(&mut self) {}

    /// **Phase 1** — register trap parameters and compute `total_params`.
    pub fn setup_traps<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        rctx: &mut RC,
    ) -> u32 {
        let gpr_type = if self.enable_mips64 { ValType::I64 } else { ValType::I32 };
        rctx.layout_mut().append(32, gpr_type);
        rctx.layout_mut().append(2, gpr_type);
        rctx.layout_mut().append(1, ValType::I32);
        rctx.declare_trap_params();
        let mark = rctx.layout().mark();
        rctx.set_locals_mark(mark);
        mark.total_locals
    }

    /// The current total wasm function parameter count.
    pub fn total_params<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self,
        rctx: &RC,
    ) -> u32 {
        rctx.locals_mark().total_locals
    }

    /// Classify a decoded MIPS instruction into [`InsnClass`] flags.
    fn classify_insn(id: rabbitizer::InstrId) -> InsnClass {
        use rabbitizer::InstrId;
        match id {
            // Loads
            InstrId::cpu_lw | InstrId::cpu_lh | InstrId::cpu_lb
            | InstrId::cpu_lwu | InstrId::cpu_lhu | InstrId::cpu_lbu
            | InstrId::cpu_ld | InstrId::cpu_ldc1 | InstrId::cpu_lwc1
            // Stores
            | InstrId::cpu_sw | InstrId::cpu_sh | InstrId::cpu_sb
            | InstrId::cpu_sd | InstrId::cpu_sdc1 | InstrId::cpu_swc1
                => InsnClass::MEMORY,
            // Conditional branches
            InstrId::cpu_beq | InstrId::cpu_bne | InstrId::cpu_blez | InstrId::cpu_bgtz
            | InstrId::cpu_bltz | InstrId::cpu_bgez | InstrId::cpu_beql | InstrId::cpu_bnel
            | InstrId::cpu_bltzl | InstrId::cpu_bgezl | InstrId::cpu_blezl | InstrId::cpu_bgtzl
                => InsnClass::BRANCH,
            // Direct unconditional jump
            InstrId::cpu_j => InsnClass::BRANCH,
            // Direct call
            InstrId::cpu_jal => InsnClass::CALL,
            // Indirect jump
            InstrId::cpu_jr => InsnClass::BRANCH | InsnClass::INDIRECT,
            // Indirect call
            InstrId::cpu_jalr => InsnClass::CALL | InsnClass::INDIRECT,
            // System
            InstrId::cpu_syscall => InsnClass::PRIVILEGED,
            InstrId::cpu_break => InsnClass::PRIVILEGED,
            // FP loads/stores
            InstrId::cpu_lwc2 | InstrId::cpu_swc2
                => InsnClass::MEMORY | InsnClass::FLOAT,
            _ => InsnClass::OTHER,
        }
    }
    /// When enabled, MIPS64-specific instructions will be translated instead of
    /// emitting unreachable.
    pub fn set_mips64_support(&mut self, enable: bool) {
        self.enable_mips64 = enable;
    }

    /// Check if MIPS64 support is enabled
    pub fn is_mips64_enabled(&self) -> bool {
        self.enable_mips64
    }

    /// Install a slot assigner to control which guest PCs receive WASM function slots.
    ///
    /// When set, `pc_to_func_idx` uses `SlotAssigner::slot_for_pc` instead of the
    /// legacy `(pc - base_pc) / 4` formula.  Must be called before `translate_bytes`.
    pub fn set_slot_assigner(&mut self, gate: impl SlotAssigner + Send + Sync + 'static) {
        self.slot_assigner = Some(alloc::boxed::Box::new(gate));
    }

    /// Return the total WASM function slots declared by the installed slot assigner.
    ///
    /// Panics if no slot assigner has been installed via `set_slot_assigner`.
    pub fn count_fns(&self) -> u32 {
        self.slot_assigner
            .as_ref()
            .expect("set_slot_assigner must be called before count_fns")
            .total_slots()
    }

    /// Convert a PC value to its 0-based WASM function slot index.
    ///
    /// When a slot assigner is installed, uses `SlotAssigner::slot_for_pc` (correct
    /// for all instruction widths).  Falls back to `(pc - base_pc) / 4` otherwise.
    ///
    /// Returns `None` when the PC is omitted; callers emit `unreachable` at the jump site.
    fn pc_to_func_idx(&self, pc: u32) -> Option<FuncIdx> {
        if let Some(gate) = &self.slot_assigner {
            gate.slot_for_pc(pc as u64).map(FuncIdx)
        } else {
            let offset_pc = pc.wrapping_sub(self.base_pc);
            Some(FuncIdx(offset_pc / 4))
        }
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
    fn init_function<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        _pc: u32,
        num_temps: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<usize, E> {
        let gpr_type = if self.enable_mips64 { ValType::I64 } else { ValType::I32 };
        let mark = rctx.locals_mark();
        rctx.layout_mut().rewind(&mark);
        self.temps_slot = rctx.layout_mut().append(num_temps, gpr_type);
        self.addr_scratch_slot = rctx.layout_mut().append(1, ValType::I32);
        self.pool_i32_slot = rctx.layout_mut().append(Self::N_POOL_I32, ValType::I32);
        self.pool_i64_slot = rctx.layout_mut().append(Self::N_POOL_I64, ValType::I64);
        rctx.declare_trap_locals();
        let _cell = rctx.alloc_cell();
        let pool_i32_start = rctx.layout().base(self.pool_i32_slot);
        let pool_i64_start = rctx.layout().base(self.pool_i64_slot);
        rctx.with_local_pool(&mut |p| p.seed_i32(pool_i32_start, Self::N_POOL_I32));
        rctx.with_local_pool(&mut |p| p.seed_i64(pool_i64_start, Self::N_POOL_I64));
        let fn_type = f(&mut rctx.layout().iter_since(&mark).collect::<alloc::vec::Vec<_>>().into_iter());
        rctx.next_with(ctx, fn_type, 1)
    }

    /// Number of i32 locals reserved in the local pool for lazy-store operand saving.
    const N_POOL_I32: u32 = 8;
    /// Number of i64 locals reserved in the local pool for lazy-store operand saving.
    const N_POOL_I64: u32 = 4;

    /// Scratch local used to save the effective load address for alias checks.
    ///
    /// Follows immediately after the `num_temps` GPR-type temp locals.
    /// `translate_instruction` always passes `num_temps = 8`, so this is
    /// local 35 + 8 = 43.  Always `i32`.
    fn load_addr_scratch_local(&self, layout: &yecta::LocalLayout) -> u32 {
        layout.base(self.addr_scratch_slot)
    }

    /// The wasm [`ValType`] of an effective (post-mapper) memory address.
    ///
    /// MIPS linear memory addresses are always 32-bit (`i32`) — there is no
    /// memory64 mode for MIPS yet.
    fn addr_val_type(&self) -> ValType {
        ValType::I32
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
    fn emit_int_const<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize, value: i32,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(value as i64))
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(value))
        }
    }

    /// Emit an unsigned integer constant (i32 or i64 depending on MIPS64 mode)
    fn emit_uint_const<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize, value: u32,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(value as i64))
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(value as i32))
        }
    }

    /// Emit an add instruction (I32Add or I64Add depending on MIPS64 mode)
    fn emit_add<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Add)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Add)
        }
    }

    /// Emit a sub instruction (I32Sub or I64Sub depending on MIPS64 mode)
    fn emit_sub<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Sub)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Sub)
        }
    }

    /// Emit a multiply instruction (I32Mul or I64Mul depending on MIPS64 mode)
    fn emit_mul<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Mul)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Mul)
        }
    }

    /// Emit a logical and instruction (I32And or I64And depending on MIPS64 mode)
    fn emit_and<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64And)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32And)
        }
    }

    /// Emit a logical or instruction (I32Or or I64Or depending on MIPS64 mode)
    fn emit_or<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Or)
        }
    }

    /// Emit a logical xor instruction (I32Xor or I64Xor depending on MIPS64 mode)
    fn emit_xor<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Xor)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Xor)
        }
    }

    /// Emit a shift left instruction (I32Shl or I64Shl depending on MIPS64 mode)
    fn emit_shl<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Shl)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Shl)
        }
    }

    /// Emit a logical shift right instruction (I32ShrU or I64ShrU depending on MIPS64 mode)
    fn emit_shr_u<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrU)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrU)
        }
    }

    /// Emit an arithmetic shift right instruction (I32ShrS or I64ShrS depending on MIPS64 mode)
    fn emit_shr_s<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,  ) -> Result<(), E> {
        if self.enable_mips64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrS)
        } else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrS)
        }
    }

    /// Perform a jump to a target PC using yecta's jump API.
    ///
    /// If `target_pc` is omitted from the slot assigner, emits `unreachable` instead.
    fn jump_to_pc<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&mut self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize, target_pc: u32, params: u32,  ) -> Result<(), E> {
        match self.pc_to_func_idx(target_pc) {
            Some(target_func) => rctx.jmp(ctx, tail_idx, target_func, params),
            None => rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable),
        }
    }

    /// Helper to translate branch instructions using yecta's ji API with custom condition
    fn translate_branch<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        rs: GprO32,
        rt: Option<GprO32>,
        offset: i32,
        pc: u32,
        op: BranchOp,
    ) -> Result<(), E> {
        // Calculate target PC: PC + 4 + (offset << 2)
        let target_pc = (pc as i32 + 4 + (offset << 2)) as u32;

        // Create a custom Snippet for the branch condition
        struct BranchCondition {
            rs_local: u32,
            rt_local: Option<u32>,
            op: BranchOp,
        }

        impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for BranchCondition {
            fn emit(
                &self,
                ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
            ) -> Result<(), E> {
                // Emit comparison instructions
                sink.instruction(ctx, &WasmInstruction::LocalGet(self.rs_local))?;
                match self.op {
                    BranchOp::Eq | BranchOp::Ne => {
                        sink.instruction(ctx, &WasmInstruction::LocalGet(self.rt_local.unwrap()))?;
                        sink.instruction(
                            ctx,
                            &match self.op {
                                BranchOp::Eq => WasmInstruction::I32Eq,
                                BranchOp::Ne => WasmInstruction::I32Ne,
                                _ => unreachable!(),
                            },
                        )?;
                    }
                    BranchOp::LeZ | BranchOp::GtZ | BranchOp::LtZ | BranchOp::GeZ => {
                        sink.instruction(ctx, &WasmInstruction::I32Const(0))?;
                        sink.instruction(
                            ctx,
                            &match self.op {
                                BranchOp::LeZ => WasmInstruction::I32LeS,
                                BranchOp::GtZ => WasmInstruction::I32GtS,
                                BranchOp::LtZ => WasmInstruction::I32LtS,
                                BranchOp::GeZ => WasmInstruction::I32GeS,
                                _ => unreachable!(),
                            },
                        )?;
                    }
                }
                Ok(())
            }
        }

        impl<Context, E> wax_core::build::InstructionSource<Context, E> for BranchCondition {
            fn emit_instruction(
                &self,
                ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
            ) -> Result<(), E> {
                // Emit the same instructions as emit_instruction
                sink.instruction(ctx, &WasmInstruction::LocalGet(self.rs_local))?;
                match self.op {
                    BranchOp::Eq | BranchOp::Ne => {
                        sink.instruction(ctx, &WasmInstruction::LocalGet(self.rt_local.unwrap()))?;
                        sink.instruction(
                            ctx,
                            &match self.op {
                                BranchOp::Eq => WasmInstruction::I32Eq,
                                BranchOp::Ne => WasmInstruction::I32Ne,
                                _ => unreachable!(),
                            },
                        )?;
                    }
                    BranchOp::LeZ | BranchOp::GtZ | BranchOp::LtZ | BranchOp::GeZ => {
                        sink.instruction(ctx, &WasmInstruction::I32Const(0))?;
                        sink.instruction(
                            ctx,
                            &match self.op {
                                BranchOp::LeZ => WasmInstruction::I32LeS,
                                BranchOp::GtZ => WasmInstruction::I32GtS,
                                BranchOp::LtZ => WasmInstruction::I32LtS,
                                BranchOp::GeZ => WasmInstruction::I32GeS,
                                _ => unreachable!(),
                            },
                        )?;
                    }
                }
                Ok(())
            }
        }

        let condition = BranchCondition {
            rs_local: Self::gpr_to_local(rs),
            rt_local: rt.map(|r| Self::gpr_to_local(r)),
            op,
        };

        // Jump trap: conditional branch.
        let branch_info =
            JumpInfo::direct(pc as u64, target_pc as u64, JumpKind::ConditionalBranch);
        if rctx.on_jump(&branch_info, ctx)?
            == TrapAction::Skip
        {
            return Ok(());
        }

        // Use ji with condition for branch taken path
        let Some(target_func) = self.pc_to_func_idx(target_pc) else {
            rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
            return Ok(());
        };
        let target = Target::Static { func: target_func };

        rctx.ji(
            ctx,
            tail_idx,
            rctx.locals_mark().total_locals, // params: pass all registers (including trap params)
            &BTreeMap::new(),  // fixups: none needed
            target,            // target: branch target
            None,              // call: not an escape call
            rctx.pool(),  // pool: for indirect calls
            Some(&condition),  // condition: branch condition
        )?;

        Ok(())
    }

    /// Translate a single MIPS instruction to WebAssembly
    ///
    /// This creates a separate function for instruction at given PC and
    /// handles jumps to other instructions using the yecta reactor's jump APIs.
    ///
    /// # Arguments
    /// * `instruction` - The decoded MIPS instruction
    /// * `f` - Function to create the instruction sink
    pub fn translate_instruction<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        instruction: &Instruction,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        let pc = instruction.vram;

        if let Some(gate) = &self.slot_assigner {
            if gate.slot_for_pc(pc as u64).is_none() {
                return Ok(());
            }
        }

        let tail_idx = self.init_function(ctx, rctx, pc, 8, f)?;

        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(pc as i32))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::pc_local()))?;

        let insn_info = InstructionInfo {
            pc: pc as u64,
            len: 4,
            arch: ArchTag::Mips,
            class: Self::classify_insn(instruction.unique_id),
        };
        if rctx.on_instruction(&insn_info, ctx)? == TrapAction::Skip {
            return Ok(());
        }

        let opcode = instruction.unique_id;

        match opcode {
            // Arithmetic instructions
            InstrId::cpu_add => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_addu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_addi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_addiu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_sub => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_sub(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_subu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_sub(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            // Logical operations
            InstrId::cpu_and => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_or => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_xor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_nor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    self.emit_int_const(ctx, rctx, tail_idx, -1)?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_andi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_ori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            InstrId::cpu_xori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Shift operations
            InstrId::cpu_sll => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shl(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_srl => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_u(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            InstrId::cpu_sra => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_s(ctx, rctx, tail_idx)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }
            }

            // Load upper immediate
            InstrId::cpu_lui => {
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const((imm << 16) as i32))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Branch instructions
            InstrId::cpu_beq => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, Some(rt), offset, pc, BranchOp::Eq)?;
            }

            InstrId::cpu_bne => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, Some(rt), offset, pc, BranchOp::Ne)?;
            }

            InstrId::cpu_blez => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::LeZ)?;
            }

            InstrId::cpu_bgtz => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::GtZ)?;
            }

            InstrId::cpu_bltz => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::LtZ)?;
            }

            InstrId::cpu_bgez => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::GeZ)?;
            }

            // Load Byte (LB) - signed
            InstrId::cpu_lb => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // load byte (signed) -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I32Load8S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    // extend to i64 if needed
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Load Byte Unsigned (LBU)
            InstrId::cpu_lbu => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // load byte unsigned -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I32Load8U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Load Halfword (LH) - signed
            InstrId::cpu_lh => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // load halfword signed -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I32Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Load Halfword Unsigned (LHU)
            InstrId::cpu_lhu => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // load halfword unsigned -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I32Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Store Byte (SB)
            InstrId::cpu_sb => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;

                if let Some(mapper) = self.mapper_callback.as_mut() {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    mapper.call(ctx, &mut callback_ctx)?;
                }

                // value to store: wrap to i32 then store 8 bits
                if self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }

            // Store Halfword (SH)
            InstrId::cpu_sh => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;

                if let Some(mapper) = self.mapper_callback.as_mut() {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    mapper.call(ctx, &mut callback_ctx)?;
                }

                // value to store: wrap to i32 then store 16 bits
                if self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }

            // Load Word (LW)
            InstrId::cpu_lw => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    // compute effective address: base + imm
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    // invoke mapper callback if present (virtual -> physical)
                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // perform memory load: always load 32-bit, sign-extend to 64 if MIPS64
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Store Word (SW)
            InstrId::cpu_sw => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                // compute effective address: base + imm
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;

                // invoke mapper callback if present (virtual -> physical)
                if let Some(mapper) = self.mapper_callback.as_mut() {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    mapper.call(ctx, &mut callback_ctx)?;
                }

                // value to store: if MIPS64 wrap to i32 then store 32-bit
                if self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }

            // Load Doubleword (LD) - 64-bit load (MIPS64)
            InstrId::cpu_ld => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if !self.enable_mips64 {
                    // LD is only valid when MIPS64 support is enabled
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else if rt != GprO32::zero {
                    // compute effective address: base + imm
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    // invoke mapper callback if present (virtual -> physical)
                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // perform 64-bit memory load
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        ValType::I32,
                        self.atomic_opts,
                        WasmInstruction::I64Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // Store Doubleword (SD) - 64-bit store (MIPS64)
            InstrId::cpu_sd => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if !self.enable_mips64 {
                    // SD is only valid when MIPS64 support is enabled
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else {
                    // compute effective address: base + imm
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    // invoke mapper callback if present (virtual -> physical)
                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    // store 64-bit value directly
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        ValType::I32,
                        WasmInstruction::I64Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                }
            }

            // Jump instructions
            InstrId::cpu_j => {
                // J uses a 26-bit instruction index field (instr_index)
                let target = (instruction.get_instr_index() as u32) << 2;
                let target_pc = (pc & 0xF0000000) | target;

                let j_info = JumpInfo::direct(pc as u64, target_pc as u64, JumpKind::DirectJump);
                if rctx.on_jump(&j_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }
                self.jump_to_pc(ctx, rctx, tail_idx, target_pc, rctx.locals_mark().total_locals)?;
                return Ok(());
            }

            InstrId::cpu_jal => {
                let target = (instruction.get_instr_index() as u32) << 2;
                let target_pc = (pc & 0xF0000000) | target;

                // Save return address in $ra ($31)
                let return_addr = pc + 8; // JAL has a delay slot
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(GprO32::ra)),
                )?;

                let jal_info = JumpInfo::direct(pc as u64, target_pc as u64, JumpKind::Call);
                if rctx.on_jump(&jal_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }
                self.jump_to_pc(ctx, rctx, tail_idx, target_pc, rctx.locals_mark().total_locals)?;
                return Ok(());
            }

            InstrId::cpu_jr => {
                let rs: GprO32 = instruction.get_rs_o32();

                // Tee rs into load_addr_scratch_local for the jump trap.
                let scratch = self.load_addr_scratch_local(rctx.layout());
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(scratch))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::Drop)?;

                // JR $ra = return; other JR = indirect jump
                let jr_kind = if rs == GprO32::ra {
                    JumpKind::Return
                } else {
                    JumpKind::IndirectJump
                };
                let jr_info = JumpInfo::indirect(pc as u64, scratch, jr_kind);
                if rctx.on_jump(&jr_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }

                let snippet = TableIndexSnippet {
                    rs_local: Self::gpr_to_local(rs),
                    base_pc: self.base_pc,
                };
                let params =
                    yecta::JumpCallParams::indirect_jump(&snippet, rctx.locals_mark().total_locals, rctx.pool());
                rctx.ji_with_params(ctx, tail_idx, params)?
;                return Ok(());
            }

            InstrId::cpu_jalr => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                // Save return address
                let return_addr = pc + 8; // JALR has a delay slot
                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rd)))?;
                }

                // Tee rs into load_addr_scratch_local for the jump trap.
                let scratch = self.load_addr_scratch_local(rctx.layout());
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rs)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(scratch))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::Drop)?;

                let jalr_info = JumpInfo::indirect(pc as u64, scratch, JumpKind::IndirectCall);
                if rctx.on_jump(&jalr_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }

                let snippet = TableIndexSnippet {
                    rs_local: Self::gpr_to_local(rs),
                    base_pc: self.base_pc,
                };
                let params =
                    yecta::JumpCallParams::indirect_jump(&snippet, rctx.locals_mark().total_locals, rctx.pool());
                rctx.ji_with_params(ctx, tail_idx, params)?;
                return Ok(());
            }

            // System instructions
            InstrId::cpu_syscall => {
                let syscall_info = SyscallInfo {
                    pc,
                    syscall_number: 0, // Would need to extract from $v0
                };

                // Invoke callback if set
                if let Some(ref mut callback) = self.syscall_callback {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    callback.call(&syscall_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: system call - implementation specific
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                }
            }

            InstrId::cpu_break => {
                let code = instruction.get_code() as u32;
                let break_info = BreakInfo { pc, code };

                // Invoke callback if set
                if let Some(ref mut callback) = self.break_callback {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    callback.call(&break_info, ctx, &mut callback_ctx);
                } else {
                    // Default behavior: breakpoint - implementation specific
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                }
            }

            // SYNC: MIPS memory barrier
            //
            // MIPS Specification: "The SYNC instruction affects the order in
            // which memory access operations are seen by other processors or
            // devices sharing the same memory."
            //
            // Under MemOrder::Relaxed, emit_fence flushes all stores that were
            // deferred by feed_lazy, committing them in program order before the
            // next instruction.  Under MemOrder::Strong the lazy buffer is always
            // empty, so this is a guaranteed no-op.
            InstrId::cpu_sync => {
                emit_fence(ctx, rctx, self.mem_order, tail_idx)?;
            }

            // ── Atomic load-linked / store-conditional ────────────────────────────
            //
            // MIPS LL/SC implement optimistic concurrency.  On wasm (single-threaded
            // model) the SC always succeeds; on shared-memory wasm the wasm atomic
            // load/store give the necessary ordering.  The reservation register is
            // not tracked — SC always writes 1 (success) into rt.
            //
            // LL rt, offset(base) — load-linked word (32-bit)
            InstrId::cpu_ll => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_lr(
                        ctx,
                        rctx,
                        RmwWidth::W32,
                        self.atomic_opts,
                        load_addr,
                        ValType::I32,
                        speet_ordering::MemOrder::Strong,
                        tail_idx,
                    )?;

                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // SC rt, offset(base) — store-conditional word (32-bit); always succeeds
            InstrId::cpu_sc => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                // Compute effective address
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;

                if let Some(mapper) = self.mapper_callback.as_mut() {
                    let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                    mapper.call(ctx, &mut callback_ctx)?;
                }

                // Value to store: rt (truncated to i32 if MIPS64)
                if self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                } else {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;
                }

                emit_sc(
                    ctx,
                    rctx,
                    RmwWidth::W32,
                    self.atomic_opts,
                    self.mem_order,
                    tail_idx,
                )?;

                // SC always succeeds: write 1 into rt
                if rt != GprO32::zero {
                    let one: WasmInstruction<'static> = if self.enable_mips64 {
                        WasmInstruction::I64Const(1)
                    } else {
                        WasmInstruction::I32Const(1)
                    };
                    rctx.feed(ctx, tail_idx, &one)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // LLD rt, offset(base) — load-linked doubleword (MIPS64 only)
            InstrId::cpu_lld => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if !self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_lr(
                        ctx,
                        rctx,
                        RmwWidth::W64,
                        self.atomic_opts,
                        load_addr,
                        ValType::I32,
                        speet_ordering::MemOrder::Strong,
                        tail_idx,
                    )?;

                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                }
            }

            // SCD rt, offset(base) — store-conditional doubleword (MIPS64 only); always succeeds
            InstrId::cpu_scd => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i32;

                if !self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(base)))?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(mapper) = self.mapper_callback.as_mut() {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut callback_ctx = CallbackContext::new(&mut fed);
                        mapper.call(ctx, &mut callback_ctx)?;
                    }

                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(Self::gpr_to_local(rt)))?;

                    emit_sc(
                        ctx,
                        rctx,
                        RmwWidth::W64,
                        self.atomic_opts,
                        self.mem_order,
                        tail_idx,
                    )?;

                    // SCD always succeeds: write 1 into rt
                    if rt != GprO32::zero {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(1))?;
                        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(Self::gpr_to_local(rt)))?;
                    }
                }
            }

            // Unsupported or unimplemented instructions
            _ => {
                // Emit unreachable for unsupported instructions
                rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
            }
        }

        Ok(())
    }
}

// ── Recompile impl ────────────────────────────────────────────────────────────

use alloc::{string::String, vec::Vec};
use speet_link_core::{
    context::{FedContext, ReactorContext},
    recompiler::Recompile,
    unit::{BinaryUnit, FuncType},
};

impl<'cb, 'ctx, Context, E, F> Recompile<Context, E, F>
    for MipsRecompiler<'cb, 'ctx, Context, E, F>
where
    F: InstructionSink<Context, E>,
{
    /// New `base_pc` for the next MIPS binary.
    type BinaryArgs = u32;

    fn reset_for_next_binary(
        &mut self,
        _ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        new_base_pc: u32,
    ) {
        self.base_pc = new_base_pc;
    }

    fn drain_unit(
        &mut self,
        rctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(alloc::string::String, u32)>,
    ) -> BinaryUnit<F> {
        use wasm_encoder::ValType;
        // All MIPS params are i32 (32-bit register file + PC).
        let total_params = rctx.locals_mark().total_locals;
        let param_types: alloc::vec::Vec<ValType> =
            (0..total_params).map(|_| ValType::I32).collect();
        let func_type = FuncType::from_val_types(&param_types, &[]);

        let base = rctx.base_func_offset();
        let fns = rctx.drain_fns();
        let count = fns.len();
        BinaryUnit {
            fns,
            base_func_offset: base,
            entry_points,
            func_types: alloc::vec![func_type; count],
            data_segments: alloc::vec![],
            data_init_fn: None,
        }
    }
}
