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
//! - Local 35: Expected return address for speculative JAL/JALR/JR
//!   (`CallEscape::{Flag,Exception}`)
//! - Locals 36+: Temporary variables for complex operations
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
pub use speet_memory::{
    AddressMapper, AddressWidth, CallbackContext, DirectMemory, IntWidth, MapperCallback,
    MemoryAccess,
    mem::{LoadKind, StoreKind},
};
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

// Shared snippet to compute a WASM table index for indirect jumps:
//   table_idx = ((reg_value & ~3) - base_pc) >> 2 + base_func_offset
//
// `base_func_offset` is required, not optional: the WASM table's `elem`
// segment populates indices `[n_imports, n_imports + n_fns)`, not
// `[0, n_fns)` (see `speet-recompile`'s `finish_module`). Omitting it here
// makes every `JR`/`JALR` target either an unpopulated (imports-reserved)
// slot or the wrong guest function, off by exactly `base_func_offset` — the
// "speet emission gap" (see `docs/guides/thin-runtime-genericity.md`
// principle 1) that AArch64's `A64IndirectTarget` and x86-64's
// `ReturnAddressSnippet` fix the same way.
struct TableIndexSnippet {
    rs_local: u32,
    text_base: speet_link_core::TextBaseSnippet,
    base_func_offset: u32,
}

impl TableIndexSnippet {
    fn from_constant_base(rs_local: u32, base_pc: u32, base_func_offset: u32) -> Self {
        Self {
            rs_local,
            text_base: speet_link_core::TextBaseSnippet::new(
                speet_link_core::TextBaseSource::Constant(base_pc as u64),
            ),
            base_func_offset,
        }
    }
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
        sink.instruction(ctx, &WasmInstruction::I64ExtendI32U)?;
        wax_core::build::InstructionSource::emit_instruction(&self.text_base, ctx, sink)?;
        sink.instruction(ctx, &WasmInstruction::I64Sub)?;
        sink.instruction(ctx, &WasmInstruction::I64Const(2))?;
        sink.instruction(ctx, &WasmInstruction::I64ShrU)?;
        sink.instruction(ctx, &WasmInstruction::I64Const(self.base_func_offset as i64))?;
        sink.instruction(ctx, &WasmInstruction::I64Add)?;
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
        sink.instruction(ctx, &WasmInstruction::I64ExtendI32U)?;
        wax_core::build::InstructionSource::emit_instruction(&self.text_base, ctx, sink)?;
        sink.instruction(ctx, &WasmInstruction::I64Sub)?;
        sink.instruction(ctx, &WasmInstruction::I64Const(2))?;
        sink.instruction(ctx, &WasmInstruction::I64ShrU)?;
        sink.instruction(ctx, &WasmInstruction::I64Const(self.base_func_offset as i64))?;
        sink.instruction(ctx, &WasmInstruction::I64Add)?;
        Ok(())
    }
}

/// Computes the expected return address for a speculative JAL/JALR call —
/// a fixup snippet written into `expected_ra_slot` at the call site (see
/// `docs/guides/yecta.md`'s speculative-call section and `speet-riscv`'s
/// `direct.rs` for the reference pattern).
///
/// Always emits an `i32` constant: MIPS PCs are 32-bit regardless of
/// `enable_mips64` (mirrors `pc_slot`, which stays `i32` in both modes).
struct ExpectedRaSnippet {
    return_addr: u32,
}

impl<Context, E> wax_core::build::InstructionSource<Context, E> for ExpectedRaSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &WasmInstruction::I32Const(self.return_addr as i32))
    }
}

impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ExpectedRaSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &WasmInstruction::I32Const(self.return_addr as i32))
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
    /// Optional memory access implementation (address mapping + load/store emission)
    memory_access: Option<alloc::boxed::Box<dyn MemoryAccess<Context, E>>>,
    /// Whether to enable MIPS64 instruction support (disabled by default)
    enable_mips64: bool,
    /// Whether memory operations use i64 addresses (wasm `memory64`) instead
    /// of i32 addresses. Disabled by default.
    use_memory64: bool,
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
    /// Slot for the 32 GPRs ($0–$31).
    gpr_slot: LocalSlot,
    /// Slot for HI (index 0) and LO (index 1).
    hi_lo_slot: LocalSlot,
    /// Slot for the PC register.
    pc_slot: LocalSlot,
    /// Hidden expected-RA for speculative JAL/JALR/JR (`CallEscape::{Flag,Exception}`).
    /// Always `i32` (MIPS PCs are 32-bit regardless of `enable_mips64`).
    expected_ra_slot: LocalSlot,
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
    /// Optional fetcher for the instruction at an arbitrary guest PC, used to
    /// translate branch/jump delay slots inline (MIPS executes the
    /// instruction at `pc+4` before every control transfer). Without a
    /// fetcher, branches are emitted WITHOUT their delay slot and an
    /// unsupported note is recorded (coverage signal; the recompiled
    /// semantics are then knowingly wrong for delay-slot-bearing code).
    delay_slot_fetcher:
        Option<alloc::boxed::Box<dyn Fn(u32) -> Option<Instruction> + 'cb>>,
    /// Coverage-signal notes for delay-slot handling (missing fetcher,
    /// out-of-range slot, control transfer in a slot).
    delay_slot_notes: alloc::vec::Vec<alloc::string::String>,
    /// PCs already executed inline as delay slots — the sequential
    /// translation loop must skip these (executing them again as their own
    /// slot would double-execute the slot on the not-taken path).
    absorbed_delay_pcs: alloc::vec::Vec<u32>,
    /// Dedicated i32 scratch for big-endian byte swaps (per-function).
    swap_tmp_slot: LocalSlot,
    /// Dedicated i64 scratch for big-endian byte swaps (per-function).
    swap_tmp_i64_slot: LocalSlot,
    /// Whether guest DATA memory is big-endian (default: true, matching the
    /// big-endian instruction fetch every caller uses for MIPS32). WASM
    /// linear memory is little-endian, so raw-path loads/stores of 16/32/64
    /// bits byte-swap when this is set. The `memory_access` mapper path is
    /// expected to handle byte order itself.
    big_endian_memory: bool,
    /// WASM import index for `env.__speet_stub_for_pc` (fn-ptr arg rewrite).
    stub_for_pc_import_idx: Option<u32>,
    /// When true, ABI JAL/JALR (link to `$ra`) and `JR $ra` use native-stack
    /// speculative lowering (`CallEscape::{Flag,Exception}`).
    enable_speculative_calls: bool,
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
            memory_access: None,
            enable_mips64,
            use_memory64: false,
            mem_order: MemOrder::Strong,
            atomic_opts: AtomicOpts::NONE,
            gpr_slot: LocalSlot::default(),
            hi_lo_slot: LocalSlot::default(),
            pc_slot: LocalSlot::default(),
            expected_ra_slot: LocalSlot::default(),
            temps_slot: LocalSlot::default(),
            addr_scratch_slot: LocalSlot::default(),
            pool_i32_slot: LocalSlot::default(),
            pool_i64_slot: LocalSlot::default(),
            slot_assigner: None,
            delay_slot_fetcher: None,
            delay_slot_notes: alloc::vec::Vec::new(),
            absorbed_delay_pcs: alloc::vec::Vec::new(),
            swap_tmp_slot: LocalSlot::default(),
            swap_tmp_i64_slot: LocalSlot::default(),
            big_endian_memory: false,
            stub_for_pc_import_idx: None,
            enable_speculative_calls: false,
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

    /// Enable or disable memory64 mode
    ///
    /// When enabled, memory operations use i64 addresses instead of i32
    /// addresses, matching a wasm `memory64` linear memory.
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

    /// Set a memory access implementation for address mapping and load/store emission.
    ///
    /// When set, `DirectMemory` (or any `MemoryAccess` impl) will handle virtual-to-physical
    /// address translation and the actual WASM load/store instructions.
    pub fn set_memory_access(
        &mut self,
        ma: alloc::boxed::Box<dyn MemoryAccess<Context, E>>,
    ) {
        self.memory_access = Some(ma);
    }

    /// Clear the memory access implementation.
    pub fn clear_memory_access(&mut self) {
        self.memory_access = None;
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

    // ── Speculative calls ────────────────────────────────────────────────

    /// Enable ABI JAL/JALR/JR speculative native-stack lowering
    /// (`CallEscape::{Flag,Exception}`).
    pub fn set_speculative_calls(&mut self, enable: bool) {
        self.enable_speculative_calls = enable;
    }

    /// Whether speculative call optimization is enabled.
    pub fn is_speculative_calls_enabled(&self) -> bool {
        self.enable_speculative_calls
    }

    /// Set the speculative-call escape policy on the reactor.
    pub fn set_escape<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        rctx: &mut RC,
        escape: yecta::CallEscape,
    ) {
        rctx.set_escape(escape);
    }

    /// Set exception-tag escape (convenience for [`yecta::CallEscape::Exception`]).
    pub fn set_escape_tag<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        rctx: &mut RC,
        tag: Option<EscapeTag>,
    ) {
        rctx.set_escape_tag(tag);
    }

    /// Current escape tag, if any.
    pub fn get_escape_tag<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self,
        rctx: &RC,
    ) -> Option<EscapeTag> {
        rctx.escape_tag()
    }

    /// WASM import index for `env.__speet_stub_for_pc` (fn-ptr arg rewrite).
    pub fn set_stub_for_pc_import_idx(&mut self, idx: u32) {
        self.stub_for_pc_import_idx = Some(idx);
    }

    /// Bind layout-param slots into the installed memory mapper (HostOffset path).
    pub fn bind_memory_layout<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        rctx: &RC,
    ) {
        if let (Some(ma), Some(params)) = (
            self.memory_access.as_deref_mut(),
            rctx.runtime_layout_params(),
        ) {
            ma.bind_layout_slots(rctx.layout(), &params.slots);
        }
    }

    // ── Trap hooks ───────────────────────────────────────────────────────

    /// MIPS base parameter count: 32 GPRs + HI + LO + PC + expected_RA = 36.
    pub const BASE_PARAMS: u32 = 36;

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
        _ctx: &mut Context,
    ) -> u32 {
        let gpr_type = if self.enable_mips64 { ValType::I64 } else { ValType::I32 };
        self.gpr_slot   = rctx.layout_mut().append(32, gpr_type); // $0–$31
        self.hi_lo_slot = rctx.layout_mut().append(2,  gpr_type); // HI, LO
        self.pc_slot    = rctx.layout_mut().append(1,  ValType::I32); // PC
        self.expected_ra_slot = rctx.layout_mut().append(1, ValType::I32); // speculative expected RA
        rctx.declare_trap_params(&mut ());
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

    /// Install the delay-slot fetcher. The closure returns the decoded
    /// instruction at a guest PC (normally `pc+4` for a branch/jump at
    /// `pc`), or `None` when the address is out of range. Callers with the
    /// code bytes in hand decode the word with rabbitizer and return it.
    pub fn set_delay_slot_fetcher(
        &mut self,
        fetcher: alloc::boxed::Box<dyn Fn(u32) -> Option<Instruction> + 'cb>,
    ) {
        self.delay_slot_fetcher = Some(fetcher);
    }

    /// Coverage-signal notes recorded while translating branch/jump delay
    /// slots (missing fetcher, out-of-range slot, UNPREDICTABLE slot).
    pub fn delay_slot_notes(&self) -> &[alloc::string::String] {
        &self.delay_slot_notes
    }

    /// Select guest data-memory byte order (default: big-endian). Call this
    /// only for little-endian MIPS guests; WASM memory itself is always LE.
    pub fn set_little_endian_memory(&mut self) {
        self.big_endian_memory = false;
    }

    /// Byte-swap the low 16 bits of the i32 on the stack (stack: [v] -> [v']).
    /// Uses the dedicated swap temp local — never the lazy-store pool.
    fn emit_swap16_i32<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        // t = ((v & 0xFF) << 8) | ((v >> 8) & 0xFF)
        let t = rctx.layout().base(self.swap_tmp_slot);
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(t))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Or)?;
        Ok(())
    }

    /// Byte-swap the i32 on the stack (stack: [v] -> [v']).
    fn emit_swap32_i32<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        // r = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF)
        let t0 = rctx.layout().base(self.swap_tmp_slot);
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(24))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF00))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF00))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(24))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Or)?;
        Ok(())
    }

    /// Byte-swap the i64 on the stack (stack: [v] -> [v']).
    fn emit_swap64_i64<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let t0 = rctx.layout().base(self.swap_tmp_i64_slot);
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(56))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF00))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(40))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF0000))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(24))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF000000))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Shl)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(8))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF000000))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(24))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF0000))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(40))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF00))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(t0))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(56))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ShrU)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0xFF))?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64And)?;
        rctx.feed(ctx, tail_idx, &WasmInstruction::I64Or)?;
        Ok(())
    }


    /// Execute a branch/jump's delay slot inline: set the PC local to
    /// `pc + 4` and emit the delay instruction's body into the current
    /// slot before the control transfer.
    ///
    /// Returns `Ok(true)` when the caller should proceed with the jump
    /// emission, `Ok(false)` when the delay slot made the transfer
    /// UNPREDICTABLE (a control-transfer instruction in a delay slot) and
    /// the jump must be dropped.
    fn emit_delay_slot<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        pc: u32,
    ) -> Result<bool, E> {
        let Some(fetcher) = &self.delay_slot_fetcher else {
            self.delay_slot_notes.push(alloc::format!("delay-slot: no fetcher (branch at {pc:#x})"));
            return Ok(true);
        };
        let Some(delay) = fetcher(pc.wrapping_add(4)) else {
            self.delay_slot_notes.push(alloc::format!("delay-slot: out of range at {pc:#x}"));
            return Ok(true);
        };
        let delay_class = Self::classify_insn(delay.unique_id);
        if delay_class.contains(InsnClass::BRANCH) || delay_class.contains(InsnClass::CALL) {
            // Control transfer in a delay slot is UNPREDICTABLE per the ISA.
            self.delay_slot_notes.push(alloc::format!("delay-slot: control transfer at {pc:#x}"));
            return Ok(false);
        }
        // Architectural PC during the delay slot is pc+4.
        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(pc.wrapping_add(4) as i32))?;
        self.emit_pc_set(ctx, rctx, tail_idx)?;
        self.emit_insn(ctx, rctx, tail_idx, &delay)?;
        self.absorbed_delay_pcs.push(pc.wrapping_add(4));
        Ok(true)
    }

    /// Whether the sequential translation loop should skip translating a
    /// standalone slot at `pc` (it was already absorbed as a delay slot).
    pub fn is_absorbed_delay_pc(&self, pc: u32) -> bool {
        self.absorbed_delay_pcs.contains(&pc)
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
        self.addr_scratch_slot = rctx.layout_mut().append(1, self.addr_val_type());
        self.pool_i32_slot = rctx.layout_mut().append(Self::N_POOL_I32, ValType::I32);
        self.pool_i64_slot = rctx.layout_mut().append(Self::N_POOL_I64, ValType::I64);
        // Dedicated byte-swap scratch (never touched by the lazy-store pool).
        self.swap_tmp_slot = rctx.layout_mut().append(1, ValType::I32);
        self.swap_tmp_i64_slot = rctx.layout_mut().append(1, ValType::I64);
        let mut unit = ();
        let extra: &mut dyn yecta::LocalDeclarator = match self.memory_access.as_deref_mut() {
            Some(m) => m as &mut dyn yecta::LocalDeclarator,
            None => &mut unit,
        };
        rctx.declare_trap_locals(extra);
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
    /// local 36 + 8 = 44.  Always `i32`.
    fn load_addr_scratch_local(&self, layout: &yecta::LocalLayout) -> u32 {
        layout.base(self.addr_scratch_slot)
    }

    /// The wasm [`ValType`] of an effective (post-mapper) memory address.
    ///
    /// MIPS computes effective addresses in 32-bit arithmetic regardless of
    /// `use_memory64`; when memory64 mode is enabled the i32 result is
    /// widened to i64 before it reaches the actual load/store instruction
    /// (see callers of this function).
    fn addr_val_type(&self) -> ValType {
        if self.use_memory64 { ValType::I64 } else { ValType::I32 }
    }

    /// Get the local index for a general-purpose register.
    fn gpr_to_local(&self, reg: GprO32, layout: &yecta::LocalLayout) -> u32 {
        layout.local(self.gpr_slot, reg as u32)
    }

    /// Get the local index for the HI register.
    fn hi_local(&self, layout: &yecta::LocalLayout) -> u32 {
        layout.local(self.hi_lo_slot, 0)
    }

    /// Get the local index for the LO register.
    fn lo_local(&self, layout: &yecta::LocalLayout) -> u32 {
        layout.local(self.hi_lo_slot, 1)
    }

    /// Get the local index for the program counter.
    fn pc_local(&self, layout: &yecta::LocalLayout) -> u32 {
        layout.local(self.pc_slot, 0)
    }

    // ── Layout-aware register emit helpers ────────────────────────────────────

    fn feed_instrs<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
        instrs: alloc::vec::Vec<WasmInstruction<'static>>,
    ) -> Result<(), E> {
        for instr in &instrs { rctx.feed(ctx, tail_idx, instr)?; }
        Ok(())
    }

    pub fn emit_gpr_get<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize, reg: GprO32,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_get(self.gpr_slot, reg as u32);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_gpr_set<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize, reg: GprO32,
    ) -> Result<(), E> {
        let (need_meta, instrs) = {
            let layout = rctx.layout();
            let need_meta = !matches!(layout.slot_kind(self.gpr_slot), yecta::SlotKind::Plain);
            (need_meta, layout.emit_set(self.gpr_slot, reg as u32))
        };
        if need_meta {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64Const(0))?;
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0))?;
            rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0))?;
        }
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_hi_get<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_get(self.hi_lo_slot, 0);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_hi_set<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_set(self.hi_lo_slot, 0);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_lo_get<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_get(self.hi_lo_slot, 1);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_lo_set<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_set(self.hi_lo_slot, 1);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    pub fn emit_pc_set<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_set(self.pc_slot, 0);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
    }

    /// Push the expected-RA slot onto the WASM stack.
    pub(crate) fn emit_expected_ra_get<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize,
    ) -> Result<(), E> {
        let instrs = rctx.layout().emit_get(self.expected_ra_slot, 0);
        Self::feed_instrs(ctx, rctx, tail_idx, instrs)
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

    /// Reconcile a freshly-computed effective address (top of stack, in
    /// `emit_add`'s width: i64 under MIPS64 GPRs, i32 otherwise) with
    /// `addr_val_type()` (i64 under memory64, i32 otherwise). Mirrors
    /// speet-riscv's RV32/RV64 x memory64 address width handling.
    fn emit_addr_widen<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(&self, ctx: &mut Context, rctx: &mut RC, tail_idx: usize) -> Result<(), E> {
        if self.enable_mips64 {
            if !self.use_memory64 {
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
            }
        } else if self.use_memory64 {
            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
        }
        Ok(())
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
            None => rctx.oob_jump(ctx, tail_idx, target_pc as u64, params),
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

        // Evaluate the branch predicate NOW, before the delay slot executes:
        // architecturally the condition uses register values as of the
        // branch, not values written by the delay-slot instruction.
        let cond_local = rctx.layout().base(self.temps_slot); // temp 0
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(self.gpr_to_local(rs, rctx.layout())))?;
        match op {
            BranchOp::Eq | BranchOp::Ne => {
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(self.gpr_to_local(rt.expect("eq/ne need rt"), rctx.layout())))?;
                rctx.feed(ctx, tail_idx, &match op {
                    BranchOp::Eq => WasmInstruction::I32Eq,
                    _ => WasmInstruction::I32Ne,
                })?;
            }
            BranchOp::LeZ | BranchOp::GtZ | BranchOp::LtZ | BranchOp::GeZ => {
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(0))?;
                rctx.feed(ctx, tail_idx, &match op {
                    BranchOp::LeZ => WasmInstruction::I32LeS,
                    BranchOp::GtZ => WasmInstruction::I32GtS,
                    BranchOp::LtZ => WasmInstruction::I32LtS,
                    BranchOp::GeZ => WasmInstruction::I32GeS,
                    _ => unreachable!(),
                })?;
            }
        }
        rctx.feed(ctx, tail_idx, &WasmInstruction::LocalSet(cond_local))?;

        struct PrecomputedCondition { local: u32 }
        impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for PrecomputedCondition {
            fn emit(
                &self,
                _ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
            ) -> Result<(), E> {
                sink.instruction(_ctx, &WasmInstruction::LocalGet(self.local))?;
                Ok(())
            }
        }
        impl<Context, E> wax_core::build::InstructionSource<Context, E> for PrecomputedCondition {
            fn emit_instruction(
                &self,
                _ctx: &mut Context,
                sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
            ) -> Result<(), E> {
                sink.instruction(_ctx, &WasmInstruction::LocalGet(self.local))?;
                Ok(())
            }
        }

        // Jump trap: conditional branch.
        let branch_info =
            JumpInfo::direct(pc as u64, target_pc as u64, JumpKind::ConditionalBranch);
        if rctx.on_jump(&branch_info, ctx)?
            == TrapAction::Skip
        {
            return Ok(());
        }

        // MIPS delay slot: execute the instruction at pc+4 before the
        // transfer takes effect (both taken and not-taken paths).
        if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
            return Ok(());
        }

        // Use ji with condition for branch taken path
        let Some(target_func) = self.pc_to_func_idx(target_pc) else {
            rctx.oob_jump(ctx, tail_idx, target_pc as u64, rctx.locals_mark().total_locals)?;
            return Ok(());
        };
        let target = Target::Static { func: target_func };

        rctx.ji(
            ctx,
            tail_idx,
            rctx.locals_mark().total_locals, // params: pass all registers (including trap params)
            &BTreeMap::new(),  // fixups: none needed
            target,            // target: branch target
            yecta::CallEscape::Jump, // call: not an escape call
            rctx.pool(),  // pool: for indirect calls
            Some(&PrecomputedCondition { local: cond_local }),  // condition
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
        self.emit_pc_set(ctx, rctx, tail_idx)?;

        let insn_info = InstructionInfo {
            pc: pc as u64,
            len: 4,
            arch: ArchTag::Mips,
            class: Self::classify_insn(instruction.unique_id),
        };
        if rctx.on_instruction(&insn_info, ctx)? == TrapAction::Skip {
            return Ok(());
        }

        self.emit_insn(ctx, rctx, tail_idx, instruction)?;

        Ok(())
    }

    /// Emit the WASM body for a single decoded MIPS instruction into the
    /// current function slot (`tail_idx`). This is everything after the
    /// per-instruction hook — shared by `translate_instruction` and the
    /// delay-slot path of branch/jump translation.
    fn emit_insn<RC: ReactorContext<Context, E, FnType = F> + ?Sized>(
        &mut self,
        ctx: &mut Context,
        rctx: &mut RC,
        tail_idx: usize,
        instruction: &Instruction,
    ) -> Result<(), E> {
        let pc = instruction.vram;
        let opcode = instruction.unique_id;

        match opcode {
            // Arithmetic instructions
            InstrId::cpu_add => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_addu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_addi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            InstrId::cpu_addiu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            InstrId::cpu_sub => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_sub(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_subu => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_sub(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            // Logical operations
            InstrId::cpu_and => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_or => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_xor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_nor => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    self.emit_int_const(ctx, rctx, tail_idx, -1)?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_andi => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_and(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            InstrId::cpu_ori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_or(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            InstrId::cpu_xori => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm as i32))?;
                    self.emit_xor(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Shift operations
            InstrId::cpu_sll => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shl(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_srl => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_u(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            InstrId::cpu_sra => {
                let rt: GprO32 = instruction.get_rt_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let sa = instruction.get_sa();

                if rd != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(sa as i32))?;
                    self.emit_shr_s(ctx, rctx, tail_idx)?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }
            }

            // Load upper immediate
            InstrId::cpu_lui => {
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as u32;

                if rt != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const((imm << 16) as i32))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Branch instructions
            InstrId::cpu_beq => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, Some(rt), offset, pc, BranchOp::Eq)?;
            }

            InstrId::cpu_bne => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, Some(rt), offset, pc, BranchOp::Ne)?;
            }

            InstrId::cpu_blez => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::LeZ)?;
            }

            InstrId::cpu_bgtz => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::GtZ)?;
            }

            InstrId::cpu_bltz => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::LtZ)?;
            }

            InstrId::cpu_bgez => {
                let rs: GprO32 = instruction.get_rs_o32();
                let offset = instruction.get_immediate() as i16 as i32;

                self.translate_branch(ctx, rctx, tail_idx, rs, None, offset, pc, BranchOp::GeZ)?;
            }

            // Load Byte (LB) - signed
            InstrId::cpu_lb => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_addr_widen(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I8S)?;
                        if self.enable_mips64 {
                            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                        }
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
                    }

                    // load byte (signed) -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        self.addr_val_type(),
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
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Load Byte Unsigned (LBU)
            InstrId::cpu_lbu => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_addr_widen(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I8U)?;
                        if self.enable_mips64 {
                            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
                        }
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
                    }

                    // load byte unsigned -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        self.addr_val_type(),
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
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Load Halfword (LH) - signed
            InstrId::cpu_lh => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_addr_widen(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I16S)?;
                        if self.enable_mips64 {
                            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                        }
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
                    }

                    // load halfword signed -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        self.addr_val_type(),
                        self.atomic_opts,
                        WasmInstruction::I32Load16S(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.big_endian_memory {
                        // swap the low 16 bytes then re-sign-extend from bit 15
                        self.emit_swap16_i32(ctx, rctx, tail_idx)?;
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(16))?;
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Shl)?;
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(16))?;
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32ShrS)?;
                    }
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Load Halfword Unsigned (LHU)
            InstrId::cpu_lhu => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_addr_widen(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I16U)?;
                        if self.enable_mips64 {
                            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
                        }
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
                    }

                    // load halfword unsigned -> i32
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        self.addr_val_type(),
                        self.atomic_opts,
                        WasmInstruction::I32Load16U(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.big_endian_memory {
                        self.emit_swap16_i32(ctx, rctx, tail_idx)?;
                    }
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32U)?;
                    }
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Store Byte (SB)
            InstrId::cpu_sb => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;
                self.emit_addr_widen(ctx, rctx, tail_idx)?;

                let rt_instrs_sb = rctx.layout().emit_get(self.gpr_slot, rt as u32);
                if let Some(ma) = self.memory_access.as_deref_mut() {
                    use speet_ordering::EagerMemorySink;
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_addr(ctx, &mut sink)?;
                    }
                    for instr in &rt_instrs_sb { rctx.feed(ctx, tail_idx, instr)?; }
                    if ma.needs_wrap_for_narrow_store(StoreKind::I8) {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    }
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_insn(ctx, &mut sink, StoreKind::I8)?;
                    }
                    return Ok(());
                }

                // value to store: wrap to i32 then store 8 bits
                if self.enable_mips64 {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
                        WasmInstruction::I32Store8(wasm_encoder::MemArg {
                            offset: 0,
                            align: 0,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
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
                let imm = instruction.get_immediate() as i16 as i32;

                self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;
                self.emit_addr_widen(ctx, rctx, tail_idx)?;

                let rt_instrs_sh = rctx.layout().emit_get(self.gpr_slot, rt as u32);
                if let Some(ma) = self.memory_access.as_deref_mut() {
                    use speet_ordering::EagerMemorySink;
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_addr(ctx, &mut sink)?;
                    }
                    for instr in &rt_instrs_sh { rctx.feed(ctx, tail_idx, instr)?; }
                    if ma.needs_wrap_for_narrow_store(StoreKind::I16) {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    }
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_insn(ctx, &mut sink, StoreKind::I16)?;
                    }
                    return Ok(());
                }

                // value to store: wrap to i32 then store 16 bits
                if self.enable_mips64 {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    if self.big_endian_memory {
                        self.emit_swap16_i32(ctx, rctx, tail_idx)?;
                    }
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
                        WasmInstruction::I32Store16(wasm_encoder::MemArg {
                            offset: 0,
                            align: 1,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    if self.big_endian_memory {
                        self.emit_swap16_i32(ctx, rctx, tail_idx)?;
                    }
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
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
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    // compute effective address: base + imm
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;
                    self.emit_addr_widen(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I32S)?;
                        if self.enable_mips64 {
                            rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                        }
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
                    }

                    // perform memory load: always load 32-bit, sign-extend to 64 if MIPS64
                    let load_addr = self.load_addr_scratch_local(rctx.layout());
                    rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(load_addr))?;
                    emit_load(
                        ctx,
                        rctx,
                        load_addr,
                        self.addr_val_type(),
                        self.atomic_opts,
                        WasmInstruction::I32Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                    if self.big_endian_memory {
                        self.emit_swap32_i32(ctx, rctx, tail_idx)?;
                    }
                    if self.enable_mips64 {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I64ExtendI32S)?;
                    }
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Store Word (SW)
            InstrId::cpu_sw => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                // compute effective address: base + imm
                self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;
                self.emit_addr_widen(ctx, rctx, tail_idx)?;

                // invoke mapper callback if present (virtual -> physical)
                let rt_instrs_sw = rctx.layout().emit_get(self.gpr_slot, rt as u32);
                if let Some(ma) = self.memory_access.as_deref_mut() {
                    use speet_ordering::EagerMemorySink;
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_addr(ctx, &mut sink)?;
                    }
                    for instr in &rt_instrs_sw { rctx.feed(ctx, tail_idx, instr)?; }
                    if ma.needs_wrap_for_narrow_store(StoreKind::I32) {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    }
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_insn(ctx, &mut sink, StoreKind::I32)?;
                    }
                    return Ok(());
                }

                // value to store: if MIPS64 wrap to i32 then store 32-bit
                if self.enable_mips64 {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    if self.big_endian_memory {
                        self.emit_swap32_i32(ctx, rctx, tail_idx)?;
                    }
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
                        WasmInstruction::I32Store(wasm_encoder::MemArg {
                            offset: 0,
                            align: 2,
                            memory_index: 0,
                        }),
                        tail_idx,
                    )?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    if self.big_endian_memory {
                        self.emit_swap32_i32(ctx, rctx, tail_idx)?;
                    }
                    emit_store(
                        ctx,
                        rctx,
                        self.mem_order,
                        self.atomic_opts,
                        self.addr_val_type(),
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
                let imm = instruction.get_immediate() as i16 as i32;

                if !self.enable_mips64 {
                    // LD is only valid when MIPS64 support is enabled
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else if rt != GprO32::zero {
                    // compute effective address: base + imm
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    // invoke mapper callback if present (virtual -> physical)
                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_load(ctx, &mut sink, LoadKind::I64)?;
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
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
                    if self.big_endian_memory {
                        self.emit_swap64_i64(ctx, rctx, tail_idx)?;
                    }
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // Store Doubleword (SD) - 64-bit store (MIPS64)
            InstrId::cpu_sd => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if !self.enable_mips64 {
                    // SD is only valid when MIPS64 support is enabled
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else {
                    // compute effective address: base + imm
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    // invoke mapper callback if present (virtual -> physical)
                    let rt_instrs_sd = rctx.layout().emit_get(self.gpr_slot, rt as u32);
                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        {
                            let mut fed = FedContext::new(rctx, tail_idx);
                            let mut sink = EagerMemorySink::new(&mut fed);
                            ma.emit_store_addr(ctx, &mut sink)?;
                        }
                        for instr in &rt_instrs_sd { rctx.feed(ctx, tail_idx, instr)?; }
                        // StoreKind::I64 never needs wrapping
                        {
                            let mut fed = FedContext::new(rctx, tail_idx);
                            let mut sink = EagerMemorySink::new(&mut fed);
                            ma.emit_store_insn(ctx, &mut sink, StoreKind::I64)?;
                        }
                        return Ok(());
                    }

                    // store 64-bit value directly
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    if self.big_endian_memory {
                        self.emit_swap64_i64(ctx, rctx, tail_idx)?;
                    }
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
                if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
                    return Ok(());
                }
                self.jump_to_pc(ctx, rctx, tail_idx, target_pc, rctx.locals_mark().total_locals)?;
                return Ok(());
            }

            InstrId::cpu_jal => {
                let target = (instruction.get_instr_index() as u32) << 2;
                let target_pc = (pc & 0xF0000000) | target;
                let return_addr = pc + 8; // JAL has a delay slot

                // JAL always links to $ra (no destination-register field).
                let use_speculative =
                    self.enable_speculative_calls && rctx.escape().is_native_stack();

                if use_speculative {
                    // Speculative call lowering: emit a native WASM call.
                    // See SPECULATIVE_CALLS.md and set_speculative_calls() docs.
                    let escape = rctx.escape();
                    let Some(target_func) = self.pc_to_func_idx(target_pc) else {
                        rctx.oob_jump(ctx, tail_idx, target_pc as u64, rctx.locals_mark().total_locals)?;
                        return Ok(());
                    };

                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, GprO32::ra)?;

                    // Delay slot executes before the call takes effect.
                    if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
                        return Ok(());
                    }

                    let expected_ra_snippet = ExpectedRaSnippet { return_addr };

                    let params = match escape {
                        yecta::CallEscape::Exception(tag) => yecta::JumpCallParams::call(
                            target_func,
                            rctx.locals_mark().total_locals,
                            tag,
                            rctx.pool(),
                        ),
                        yecta::CallEscape::Flag => yecta::JumpCallParams::call_flag(
                            target_func,
                            rctx.locals_mark().total_locals,
                            rctx.pool(),
                        ),
                        yecta::CallEscape::Jump => unreachable!(),
                    }
                    .with_fixup(rctx.layout().local(self.expected_ra_slot, 0), &expected_ra_snippet);

                    rctx.ji_with_params(ctx, tail_idx, params)?;
                    return Ok(());
                }

                // Non-speculative path: original jump-based implementation.
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                self.emit_gpr_set(ctx, rctx, tail_idx, GprO32::ra)?;

                let jal_info = JumpInfo::direct(pc as u64, target_pc as u64, JumpKind::Call);
                if rctx.on_jump(&jal_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }
                if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
                    return Ok(());
                }
                self.jump_to_pc(ctx, rctx, tail_idx, target_pc, rctx.locals_mark().total_locals)?;
                return Ok(());
            }

            InstrId::cpu_jr => {
                let rs: GprO32 = instruction.get_rs_o32();

                // Tee rs into load_addr_scratch_local for the jump trap.
                let scratch = self.load_addr_scratch_local(rctx.layout());
                self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                self.emit_addr_widen(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(scratch))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::Drop)?;

                // JR $ra = return; other JR = indirect jump. Compute the kind
                // once and fire the trap exactly once, regardless of which
                // path below actually handles the jump.
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
                if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
                    return Ok(());
                }

                // ABI-compliant return: check ra vs expected_ra when
                // speculative calls are enabled with a native-stack escape.
                let is_abi_return = self.enable_speculative_calls
                    && rs == GprO32::ra
                    && rctx.escape().is_native_stack();

                if is_abi_return {
                    let escape = rctx.escape();

                    // Load ra (current return address) and compare against
                    // expected_ra. Both are i32 — see ExpectedRaSnippet.
                    self.emit_gpr_get(ctx, rctx, tail_idx, GprO32::ra)?;
                    self.emit_expected_ra_get(ctx, rctx, tail_idx)?;
                    if self.enable_mips64 {
                        // gpr_slot is i64 under MIPS64; wrap to i32 to
                        // compare against the (always-i32) expected_ra.
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Eq)?;

                    rctx.feed(ctx, tail_idx, &WasmInstruction::If(wasm_encoder::BlockType::Empty))?;
                    match escape {
                        yecta::CallEscape::Flag => {
                            rctx.ret_flag(ctx, tail_idx, rctx.locals_mark().total_locals, false)?;
                        }
                        yecta::CallEscape::Exception(_) | yecta::CallEscape::Jump => {
                            // (register_file) -> (register_file)
                            for p in 0..rctx.locals_mark().total_locals {
                                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalGet(p))?;
                            }
                            rctx.feed(ctx, tail_idx, &WasmInstruction::Return)?;
                        }
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Else)?;
                    match escape {
                        yecta::CallEscape::Exception(tag) => {
                            rctx.ret(ctx, tail_idx, rctx.locals_mark().total_locals, tag)?;
                        }
                        yecta::CallEscape::Flag => {
                            rctx.ret_flag(ctx, tail_idx, rctx.locals_mark().total_locals, true)?;
                        }
                        yecta::CallEscape::Jump => unreachable!(),
                    }
                    rctx.feed(ctx, tail_idx, &WasmInstruction::End)?;
                    return Ok(());
                }

                let snippet = TableIndexSnippet::from_constant_base(
                    self.gpr_to_local(rs, rctx.layout()),
                    self.base_pc,
                    rctx.base_func_offset(),
                );
                let params =
                    yecta::JumpCallParams::indirect_jump(&snippet, rctx.locals_mark().total_locals, rctx.pool());
                rctx.ji_with_params(ctx, tail_idx, params)?;
                return Ok(());
            }

            InstrId::cpu_jalr => {
                let rs: GprO32 = instruction.get_rs_o32();
                let rd: GprO32 = instruction.get_rd_o32();
                let return_addr = pc + 8; // JALR has a delay slot

                // ABI-compliant call: link register is $ra, and speculative
                // calls with a native-stack escape are enabled.
                let use_speculative = self.enable_speculative_calls
                    && rd == GprO32::ra
                    && rctx.escape().is_native_stack();

                if use_speculative {
                    // Speculative call lowering for indirect calls: since
                    // JALR is indirect, dispatch through the pool table (see
                    // SPECULATIVE_CALLS.md and the JAL handler above).
                    let escape = rctx.escape();

                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;

                    let expected_ra_snippet = ExpectedRaSnippet { return_addr };
                    let target_snippet = TableIndexSnippet::from_constant_base(
                        self.gpr_to_local(rs, rctx.layout()),
                        self.base_pc,
                        rctx.base_func_offset(),
                    );

                    let mut fixups = alloc::collections::BTreeMap::new();
                    fixups.insert(
                        rctx.layout().local(self.expected_ra_slot, 0),
                        &expected_ra_snippet as &(dyn yecta::Snippet<Context, E> + '_),
                    );

                    let params = yecta::JumpCallParams {
                        params: rctx.locals_mark().total_locals,
                        fixups,
                        target: yecta::Target::Dynamic { idx: &target_snippet },
                        call: escape,
                        pool: rctx.pool(),
                        condition: None,
                        condition_hook: None,
                    };

                    rctx.ji_with_params(ctx, tail_idx, params)?;
                    return Ok(());
                }

                // Non-speculative path: original implementation.
                if rd != GprO32::zero {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(return_addr as i32))?;
                    self.emit_gpr_set(ctx, rctx, tail_idx, rd)?;
                }

                // Tee rs into load_addr_scratch_local for the jump trap.
                let scratch = self.load_addr_scratch_local(rctx.layout());
                self.emit_gpr_get(ctx, rctx, tail_idx, rs)?;
                self.emit_addr_widen(ctx, rctx, tail_idx)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::LocalTee(scratch))?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::Drop)?;

                let jalr_info = JumpInfo::indirect(pc as u64, scratch, JumpKind::IndirectCall);
                if rctx.on_jump(&jalr_info, ctx)?
                    == TrapAction::Skip
                {
                    return Ok(());
                }
                if !self.emit_delay_slot(ctx, rctx, tail_idx, pc)? {
                    return Ok(());
                }

                let snippet = TableIndexSnippet::from_constant_base(
                    self.gpr_to_local(rs, rctx.layout()),
                    self.base_pc,
                    rctx.base_func_offset(),
                );
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
                let imm = instruction.get_immediate() as i16 as i32;

                if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        {
                            let mut fed = FedContext::new(rctx, tail_idx);
                            let mut sink = EagerMemorySink::new(&mut fed);
                            ma.emit_store_addr(ctx, &mut sink)?;
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
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
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
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // SC rt, offset(base) — store-conditional word (32-bit); always succeeds
            InstrId::cpu_sc => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                // Compute effective address
                self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                self.emit_add(ctx, rctx, tail_idx)?;

                let rt_instrs_swl = rctx.layout().emit_get(self.gpr_slot, rt as u32);
                if let Some(ma) = self.memory_access.as_deref_mut() {
                    use speet_ordering::EagerMemorySink;
                    {
                        let mut fed = FedContext::new(rctx, tail_idx);
                        let mut sink = EagerMemorySink::new(&mut fed);
                        ma.emit_store_addr(ctx, &mut sink)?;
                    }
                    for instr in &rt_instrs_swl { rctx.feed(ctx, tail_idx, instr)?; }
                    if ma.needs_wrap_for_narrow_store(StoreKind::I32) {
                        rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
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
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                    }
                    return Ok(());
                }

                // Value to store: rt (truncated to i32 if MIPS64)
                if self.enable_mips64 {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32WrapI64)?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
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
                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // LLD rt, offset(base) — load-linked doubleword (MIPS64 only)
            InstrId::cpu_lld => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if !self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else if rt != GprO32::zero {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        {
                            let mut fed = FedContext::new(rctx, tail_idx);
                            let mut sink = EagerMemorySink::new(&mut fed);
                            ma.emit_store_addr(ctx, &mut sink)?;
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
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        return Ok(());
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

                    self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                }
            }

            // SCD rt, offset(base) — store-conditional doubleword (MIPS64 only); always succeeds
            InstrId::cpu_scd => {
                let base: GprO32 = instruction.get_rs_o32();
                let rt: GprO32 = instruction.get_rt_o32();
                let imm = instruction.get_immediate() as i16 as i32;

                if !self.enable_mips64 {
                    rctx.feed(ctx, tail_idx, &WasmInstruction::Unreachable)?;
                } else {
                    self.emit_gpr_get(ctx, rctx, tail_idx, base)?;
                    rctx.feed(ctx, tail_idx, &WasmInstruction::I32Const(imm))?;
                    self.emit_add(ctx, rctx, tail_idx)?;

                    if let Some(ma) = self.memory_access.as_deref_mut() {
                        use speet_ordering::EagerMemorySink;
                        {
                            let mut fed = FedContext::new(rctx, tail_idx);
                            let mut sink = EagerMemorySink::new(&mut fed);
                            ma.emit_store_addr(ctx, &mut sink)?;
                        }
                        self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;
                        // StoreKind::I64 never needs wrapping
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
                            self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
                        }
                        return Ok(());
                    }

                    self.emit_gpr_get(ctx, rctx, tail_idx, rt)?;

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
                        self.emit_gpr_set(ctx, rctx, tail_idx, rt)?;
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
        // Read the exact per-param types back from the layout rather than
        // assuming a uniform width: GPRs/HI/LO are i32 or i64 depending on
        // `enable_mips64`, but PC and expected_RA are always i32 (see the
        // module doc's local-variable layout and `setup_traps`).
        let mark = rctx.locals_mark();
        let param_types: alloc::vec::Vec<ValType> = rctx
            .layout()
            .iter_before(&mark)
            .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
            .collect();
        // Register-file ABI: (register_file) -> (register_file) — results
        // mirror params. Required by speculative calls' Block/TryTable/Catch
        // and the ABI-compliant `Return`/`ret_flag` paths (see
        // `docs/guides/yecta.md`). Harmless when speculative calls are
        // disabled: `return_call` is a diverging control transfer, so the
        // validator never checks the declared results against it.
        let func_type = FuncType::from_val_types(&param_types, &param_types);

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

/// Calling convention for a compile-time PLT redirect of `symbol` on MIPS
/// o32/n64.
///
/// Built manually rather than through `speet_abi_stubs::plt_calling_convention`
/// — that function's `import_to_calling_convention` matches exhaustively on
/// `binary_io::BinArch`, which has no MIPS variant; adding one would cascade
/// through every such `match arch` (none of which have a MIPS arm), and the
/// checked-in stub tables it also consults only cover x86_64/aarch64 anyway,
/// so this loses nothing relative to that fallback. Replicates
/// `hook_calling_convention_from_manifest`'s manifest-driven fallback
/// directly against [`speet_host_api::ImportManifest`].
///
/// `$a0`–`$a3` are GPR `$4`–`$7` in both the o32 and n64 ABIs, and GPRs map
/// 1:1 onto WASM locals 0–31 (see `setup_traps`), so `a0` is always WASM
/// local 4 regardless of `enable_mips64`.
pub fn plt_calling_convention(symbol: &str) -> speet_plugin_api::external_target::CallingConvention {
    use speet_host_api::{ImportManifest, WasmValType};
    type CallingConvention = speet_plugin_api::external_target::CallingConvention;

    let manifest = ImportManifest::integrated_native();
    let Some((_, import_name)) = manifest.resolve_intercept(symbol) else {
        return CallingConvention::default();
    };
    let Some(imp) = manifest.func_imports.iter().find(|i| i.name == import_name) else {
        return CallingConvention::default();
    };

    let arg_locals = (4..4 + imp.params.len() as u32).collect();
    let arg_wrap_i32 = imp
        .params
        .iter()
        .map(|t| matches!(t, WasmValType::I32))
        .collect();
    let (result_local, result_extend_i32) = match imp.results.first() {
        None => (None, false),
        Some(WasmValType::I32) => (Some(0), true),
        Some(_) => (Some(0), false),
    };
    CallingConvention {
        arg_locals,
        arg_wrap_i32,
        result_local,
        result_extend_i32,
    }
}
