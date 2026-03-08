//! # DEX to WebAssembly Recompiler
//!
//! This crate provides a DEX (Dalvik Executable) to WebAssembly static recompiler
//! that translates DEX bytecode to WebAssembly using the yecta control flow library.
//!
//! ## Architecture
//!
//! Every DEX instruction in every method is assigned a flat index and becomes a
//! separate wasm function.  The index of instruction at code-unit offset `o` inside
//! a method whose [`FlatMethod::base`] is `b` is `b + o`; the emitted wasm function
//! index is `base_func_offset + b + o`.
//!
//! All `max_registers` DEX register slots are wasm *parameters* (so they survive
//! `return_call` across function boundaries).  Any extra scratch locals are appended
//! per-function via the [`LocalLayout`] slot system.
//!
//! ## Trap hooks
//!
//! Install traps **before** calling [`DexRecompiler::setup_traps`]:
//!
//! ```ignore
//! recompiler.set_jump_trap(&mut my_jump_trap);
//! recompiler.setup_traps();
//! ```
//!
//! `setup_traps` appends trap parameter groups to the shared layout and fixes
//! `total_params`.  Per-function trap locals are declared in `translate_instruction`
//! via `declare_locals`.
//!
//! ## Usage
//!
//! ```ignore
//! let dex_data = std::fs::read("classes.dex")?;
//! let mut recompiler = DexRecompiler::new(&dex_data)?;
//! recompiler.setup_traps();
//! for method in recompiler.methods() {
//!     recompiler.translate_method(&mut ctx, method, &mut |locals| Function::new(locals.collect()))?;
//! }
//! ```

#![no_std]
extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use wax_core::build::{InstructionSink, InstructionSource, InstructionOperatorSource};

use dex::{DexReader, code::ExceptionType, jtype::Type, string::DexString};
use wasm_encoder::{Instruction, ValType};
use yecta::{
    EscapeTag, FuncIdx, LocalLayout, LocalPoolBackend, LocalSlot, Mark, Pool, Reactor,
    TableIdx, Target, TypeIdx,
};

pub use speet_memory::{CallbackContext, MapperCallback};
pub use speet_ordering::{AtomicOpts, MemOrder, RmwOp, RmwWidth};
use speet_traps::{
    InstructionInfo, InstructionTrap, JumpInfo, JumpKind, JumpTrap, TrapAction, TrapConfig,
    insn::{ArchTag, InsnClass},
};

use dex_bytecode::Instruction as DexInsn;

/// Classify a decoded DEX instruction into [`InsnClass`] flags for the trap system.
fn insn_class(insn: &DexInsn) -> InsnClass {
    match insn {
        // Array access
        DexInsn::Aget { .. } | DexInsn::AgetWide { .. } | DexInsn::AgetObject { .. }
        | DexInsn::AgetBoolean { .. } | DexInsn::AgetByte { .. } | DexInsn::AgetChar { .. }
        | DexInsn::AgetShort { .. } | DexInsn::Aput { .. } | DexInsn::AputWide { .. }
        | DexInsn::AputObject { .. } | DexInsn::AputBoolean { .. } | DexInsn::AputByte { .. }
        | DexInsn::AputChar { .. } | DexInsn::AputShort { .. }
        // Instance and static field access
        | DexInsn::Iget { .. } | DexInsn::IgetWide { .. } | DexInsn::IgetObject { .. }
        | DexInsn::IgetBoolean { .. } | DexInsn::IgetByte { .. } | DexInsn::IgetChar { .. }
        | DexInsn::IgetShort { .. } | DexInsn::Iput { .. } | DexInsn::IputWide { .. }
        | DexInsn::IputObject { .. } | DexInsn::IputBoolean { .. } | DexInsn::IputByte { .. }
        | DexInsn::IputChar { .. } | DexInsn::IputShort { .. }
        | DexInsn::Sget { .. } | DexInsn::SgetWide { .. } | DexInsn::SgetObject { .. }
        | DexInsn::SgetBoolean { .. } | DexInsn::SgetByte { .. } | DexInsn::SgetChar { .. }
        | DexInsn::SgetShort { .. } | DexInsn::Sput { .. } | DexInsn::SputWide { .. }
        | DexInsn::SputObject { .. } | DexInsn::SputBoolean { .. } | DexInsn::SputByte { .. }
        | DexInsn::SputChar { .. } | DexInsn::SputShort { .. }
        => InsnClass::MEMORY,

        DexInsn::ReturnVoid | DexInsn::Return { .. }
        | DexInsn::ReturnWide { .. } | DexInsn::ReturnObject { .. }
        => InsnClass::RETURN,

        DexInsn::Throw { .. } | DexInsn::MonitorEnter { .. } | DexInsn::MonitorExit { .. }
        => InsnClass::PRIVILEGED,

        DexInsn::Goto { .. } | DexInsn::Goto16 { .. } | DexInsn::Goto32 { .. }
        => InsnClass::BRANCH,

        DexInsn::PackedSwitch { .. } | DexInsn::SparseSwitch { .. }
        => InsnClass::BRANCH | InsnClass::INDIRECT,

        DexInsn::IfEq { .. } | DexInsn::IfNe { .. } | DexInsn::IfLt { .. }
        | DexInsn::IfGe { .. } | DexInsn::IfGt { .. } | DexInsn::IfLe { .. }
        | DexInsn::IfEqz { .. } | DexInsn::IfNez { .. } | DexInsn::IfLtz { .. }
        | DexInsn::IfGez { .. } | DexInsn::IfGtz { .. } | DexInsn::IfLez { .. }
        => InsnClass::BRANCH,

        DexInsn::InvokeDirect { .. } | DexInsn::InvokeStatic { .. }
        | DexInsn::InvokeDirectRange { .. } | DexInsn::InvokeStaticRange { .. }
        | DexInsn::InvokeCustom { .. } | DexInsn::InvokeCustomRange { .. }
        => InsnClass::CALL,

        DexInsn::InvokeVirtual { .. } | DexInsn::InvokeSuper { .. }
        | DexInsn::InvokeInterface { .. } | DexInsn::InvokeVirtualRange { .. }
        | DexInsn::InvokeSuperRange { .. } | DexInsn::InvokeInterfaceRange { .. }
        | DexInsn::InvokePolymorphic { .. } | DexInsn::InvokePolymorphicRange { .. }
        => InsnClass::CALL | InsnClass::INDIRECT,

        DexInsn::CmplFloat { .. } | DexInsn::CmpgFloat { .. }
        | DexInsn::CmplDouble { .. } | DexInsn::CmpgDouble { .. }
        | DexInsn::AddFloat { .. } | DexInsn::SubFloat { .. } | DexInsn::MulFloat { .. }
        | DexInsn::DivFloat { .. } | DexInsn::RemFloat { .. }
        | DexInsn::AddDouble { .. } | DexInsn::SubDouble { .. } | DexInsn::MulDouble { .. }
        | DexInsn::DivDouble { .. } | DexInsn::RemDouble { .. }
        | DexInsn::AddFloat2addr { .. } | DexInsn::SubFloat2addr { .. }
        | DexInsn::MulFloat2addr { .. } | DexInsn::DivFloat2addr { .. }
        | DexInsn::RemFloat2addr { .. }
        | DexInsn::AddDouble2addr { .. } | DexInsn::SubDouble2addr { .. }
        | DexInsn::MulDouble2addr { .. } | DexInsn::DivDouble2addr { .. }
        | DexInsn::RemDouble2addr { .. }
        | DexInsn::IntToFloat { .. } | DexInsn::IntToDouble { .. }
        | DexInsn::LongToFloat { .. } | DexInsn::LongToDouble { .. }
        | DexInsn::FloatToInt { .. } | DexInsn::FloatToLong { .. }
        | DexInsn::FloatToDouble { .. } | DexInsn::DoubleToInt { .. }
        | DexInsn::DoubleToLong { .. } | DexInsn::DoubleToFloat { .. }
        | DexInsn::NegFloat { .. } | DexInsn::NegDouble { .. }
        => InsnClass::FLOAT,

        _ => InsnClass::OTHER,
    }
}

/// Comparison operation for a DEX conditional branch condition snippet.
#[derive(Clone, Copy)]
enum DexCondOp {
    I32Eq,
    I32Ne,
    I32LtS,
    I32GeS,
    I32GtS,
    I32LeS,
    I32Eqz,
    I32Nez,
    I32LtzS,
    I32GezS,
    I32GtzS,
    I32LezS,
}

/// Branch condition snippet for DEX if-* instructions.
/// Emits an i32 condition value onto the wasm stack.
struct DexCondSnippet {
    a: u32,
    b: u32,
    op: DexCondOp,
}

impl<Context, E> InstructionSource<Context, E> for DexCondSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        use DexCondOp::*;
        use Instruction as W;
        sink.instruction(ctx, &W::LocalGet(self.a))?;
        match self.op {
            I32Eq  => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32Eq)?; }
            I32Ne  => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32Ne)?; }
            I32LtS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32LtS)?; }
            I32GeS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32GeS)?; }
            I32GtS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32GtS)?; }
            I32LeS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32LeS)?; }
            I32Eqz  => { sink.instruction(ctx, &W::I32Eqz)?; }
            I32Nez  => { /* just the register value */ }
            I32LtzS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32LtS)?; }
            I32GezS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32GeS)?; }
            I32GtzS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32GtS)?; }
            I32LezS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32LeS)?; }
        }
        Ok(())
    }
}

impl<Context, E> InstructionOperatorSource<Context, E> for DexCondSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        use DexCondOp::*;
        use Instruction as W;
        sink.instruction(ctx, &W::LocalGet(self.a))?;
        match self.op {
            I32Eq  => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32Eq)?; }
            I32Ne  => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32Ne)?; }
            I32LtS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32LtS)?; }
            I32GeS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32GeS)?; }
            I32GtS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32GtS)?; }
            I32LeS => { sink.instruction(ctx, &W::LocalGet(self.b))?; sink.instruction(ctx, &W::I32LeS)?; }
            I32Eqz  => { sink.instruction(ctx, &W::I32Eqz)?; }
            I32Nez  => { /* just the register value */ }
            I32LtzS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32LtS)?; }
            I32GezS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32GeS)?; }
            I32GtzS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32GtS)?; }
            I32LezS => { sink.instruction(ctx, &W::I32Const(0))?; sink.instruction(ctx, &W::I32LeS)?; }
        }
        Ok(())
    }
}

// ── Flat DEX method representation ───────────────────────────────────────────

/// Flat representation of one DEX method.
#[derive(Debug, Clone)]
pub struct FlatMethod {
    /// Method index in the DEX file.
    pub idx: u64,
    /// Class definition index.
    pub class_idx: u32,
    /// Method name.
    pub name: DexString,
    /// Parameter types.
    pub params: Vec<Type>,
    /// Return type.
    pub returns: Type,
    /// Total number of registers (locals + params) for this method.
    pub registers_size: u16,
    /// Number of incoming parameter registers.
    pub ins_size: u16,
    /// Number of outgoing argument registers (for call overhead).
    pub outs_size: u16,
    /// Number of 16-bit code units in this method's bytecode.
    pub code_units: u64,
    /// Code-unit offset of the first instruction in the flat code array.
    pub base: u64,
}

/// Try/catch block information.
#[derive(Debug, Clone)]
pub struct TryBlock {
    /// Flat code-unit offset of the first protected instruction.
    pub start_addr: u64,
    /// Flat code-unit offset one past the last protected instruction.
    pub end_addr: u64,
    /// Ordered list of exception handlers for this block.
    pub handlers: Vec<Handler>,
}

/// One exception handler within a [`TryBlock`].
#[derive(Debug, Clone)]
pub struct Handler {
    /// Flat code-unit offset of the handler entry point.
    pub addr: u64,
    /// Exception type (or catch-all).
    pub type_idx: ExceptionType,
}

/// All DEX methods flattened into a single code-unit array.
#[derive(Default)]
pub struct FlatMethods {
    methods: Vec<FlatMethod>,
    tries: Vec<TryBlock>,
    /// Concatenated code units from all methods, in parsing order.
    pub code: Vec<u16>,
    /// Running total of code units appended so far.
    total_code_units: u64,
}

impl FlatMethods {
    /// Maximum `registers_size` across all methods.
    ///
    /// Used as the wasm parameter count so that every translated function has
    /// the same signature and `return_call` can forward all register state
    /// across function boundaries.
    pub fn max_registers(&self) -> u32 {
        self.methods
            .iter()
            .map(|m| m.registers_size as u32)
            .max()
            .unwrap_or(0)
    }

    /// All parsed [`FlatMethod`]s.
    pub fn methods(&self) -> &[FlatMethod] {
        &self.methods
    }

    /// The method that contains flat code-unit index `idx`, if any.
    pub fn method_of(&self, idx: u64) -> Option<&FlatMethod> {
        self.methods
            .iter()
            .find(|m| m.base <= idx && idx < m.base + m.code_units)
    }

    /// Parse DEX file bytes, appending all concrete methods to the flat representation.
    pub fn parse_dex(&mut self, data: &[u8]) -> Result<(), dex::Error> {
        let dex_file = DexReader::from_vec(data)?;
        for class_def_result in dex_file.classes() {
            let class_def: dex::class::Class = class_def_result?;
            for method in class_def.methods() {
                let code: &dex::code::CodeItem = match method.code() {
                    Some(code) => code,
                    None => continue, // abstract or native
                };
                let base = self.total_code_units;
                let flat_method = FlatMethod {
                    idx: method.id(),
                    class_idx: class_def.id(),
                    name: method.name().clone(),
                    params: method.params().clone(),
                    returns: method.return_type().clone(),
                    registers_size: code.registers_size(),
                    ins_size: code.ins_size(),
                    outs_size: code.outs_size(),
                    code_units: code.insns().len() as u64,
                    base,
                };
                for try_item in code.tries().iter() {
                    self.tries.push(TryBlock {
                        start_addr: try_item.start_addr() as u64 + base,
                        end_addr: try_item.insn_count() as u64
                            + try_item.start_addr() as u64
                            + base,
                        handlers: try_item
                            .catch_handlers()
                            .iter()
                            .map(|h| Handler {
                                addr: h.addr() as u64 + base,
                                type_idx: h.exception().clone(),
                            })
                            .collect(),
                    });
                }
                self.total_code_units += flat_method.code_units;
                self.code.extend_from_slice(code.insns());
                self.methods.push(flat_method);
            }
        }
        Ok(())
    }
}

// ── DexRecompiler ─────────────────────────────────────────────────────────────

/// DEX-to-WebAssembly recompiler.
///
/// Every DEX instruction becomes one wasm function.  All DEX registers are wasm
/// parameters so they survive `return_call` chains.  Trap hooks fire at each
/// instruction (via [`InstructionTrap`]) and at each control-flow instruction
/// (via [`JumpTrap`]).
pub struct DexRecompiler<
    'cb,
    'ctx,
    Context,
    E,
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend = yecta::LocalPool,
> {
    reactor: Reactor<Context, E, F, P>,
    pool: Pool,
    escape_tag: Option<EscapeTag>,

    /// Flattened method and code-unit data.
    flat: FlatMethods,

    /// Memory ordering mode for load/store emission.
    mem_order: MemOrder,
    /// Atomic instruction substitution options.
    atomic_opts: AtomicOpts,

    /// Pluggable trap hooks.
    traps: TrapConfig<'cb, 'ctx, Context, E, Reactor<Context, E, F, P>>,

    /// Total wasm function parameter count (registers + trap params).
    ///
    /// Set by [`setup_traps`](Self::setup_traps).  Pass as the `params`
    /// argument to every `jmp` / `ji` / `ji_with_params` call.
    total_params: u32,

    /// Unified layout: DEX register params, then trap params, then per-function locals.
    layout: LocalLayout,
    /// Mark placed after all param slots; used to rewind before each function.
    locals_mark: Mark,
    /// Slot for a single per-function scratch `i32` local.
    scratch_slot: LocalSlot,
    /// Slot for a single per-function scratch `i64` local (used for wide operations).
    scratch_i64_slot: LocalSlot,
}

impl<'cb, 'ctx, Context, E, F, P> DexRecompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + Default,
{
    /// Create a new DEX recompiler from raw DEX file bytes.
    ///
    /// Call [`setup_traps`](Self::setup_traps) (even with no traps installed)
    /// before any translation calls.
    pub fn new(input: &[u8]) -> Result<Self, dex::Error> {
        let mut flat = FlatMethods::default();
        flat.parse_dex(input)?;
        let max_regs = flat.max_registers();
        let mut recomp = Self {
            reactor: Reactor::default(),
            pool: Pool { table: TableIdx(0), ty: TypeIdx(0) },
            escape_tag: None,
            flat,
            mem_order: MemOrder::Strong,
            atomic_opts: AtomicOpts::NONE,
            traps: TrapConfig::new(),
            total_params: max_regs,
            layout: LocalLayout::empty(),
            locals_mark: Mark { slot_count: 0, total_locals: 0 },
            scratch_slot: LocalSlot(0),
            scratch_i64_slot: LocalSlot(0),
        };
        recomp.setup_traps();
        Ok(recomp)
    }

    /// Create with a base function-index offset.
    ///
    /// `base_func_offset` is added to every emitted wasm function index,
    /// useful when translated functions are preceded by imports or helper
    /// functions in the same module.
    pub fn new_with_func_offset(input: &[u8], base_func_offset: u32) -> Result<Self, dex::Error> {
        let mut recomp = Self::new(input)?;
        recomp.reactor.set_base_func_offset(base_func_offset);
        Ok(recomp)
    }
}

impl<'cb, 'ctx, Context, E, F, P> DexRecompiler<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    // ── Trap installation ─────────────────────────────────────────────────

    /// Install an instruction trap.
    ///
    /// Must be called before [`setup_traps`](Self::setup_traps).
    pub fn set_instruction_trap(
        &mut self,
        trap: &'cb mut (dyn InstructionTrap<Context, E, Reactor<Context, E, F, P>> + 'ctx),
    ) {
        self.traps.set_instruction_trap(trap);
    }

    /// Remove the instruction trap.
    pub fn clear_instruction_trap(&mut self) {
        self.traps.clear_instruction_trap();
    }

    /// Install a jump trap.
    ///
    /// Must be called before [`setup_traps`](Self::setup_traps).
    pub fn set_jump_trap(
        &mut self,
        trap: &'cb mut (dyn JumpTrap<Context, E, Reactor<Context, E, F, P>> + 'ctx),
    ) {
        self.traps.set_jump_trap(trap);
    }

    /// Remove the jump trap.
    pub fn clear_jump_trap(&mut self) {
        self.traps.clear_jump_trap();
    }

    // ── Phase 1: setup ────────────────────────────────────────────────────

    /// **Phase 1** — build the shared [`LocalLayout`] and compute `total_params`.
    ///
    /// Call once after installing all traps and before any translation.  Safe to
    /// call with no traps installed (equivalent to `total_params = max_registers`).
    ///
    /// Returns the total wasm function parameter count.
    pub fn setup_traps(&mut self) -> u32 {
        let max_regs = self.flat.max_registers();
        self.layout = LocalLayout::empty();
        // All DEX registers are params 0..max_regs-1 (i32 each).
        // DEX registers are 32-bit slots; wide types occupy adjacent pairs.
        if max_regs > 0 {
            self.layout.append(max_regs, ValType::I32);
        }
        self.traps.declare_params(&mut self.layout);
        self.locals_mark = self.layout.mark();
        self.total_params = self.locals_mark.total_locals;
        self.total_params
    }

    /// The current total wasm function parameter count.
    pub fn total_params(&self) -> u32 {
        self.total_params
    }

    // ── Reactor delegation ────────────────────────────────────────────────

    /// Returns the underlying reactor's function-index base offset.
    pub fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }

    /// Sets the function-index base offset.
    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.reactor.set_base_func_offset(offset);
    }

    // ── Memory / atomic mode ──────────────────────────────────────────────

    /// Set the memory ordering mode for load/store emission.
    pub fn set_mem_order(&mut self, order: MemOrder) {
        self.mem_order = order;
    }

    /// Set atomic instruction substitution options.
    pub fn set_atomic_opts(&mut self, opts: AtomicOpts) {
        self.atomic_opts = opts;
    }

    // ── Method access ─────────────────────────────────────────────────────

    /// All parsed flat methods.
    pub fn methods(&self) -> &[FlatMethod] {
        self.flat.methods()
    }

    /// All parsed try/catch blocks (flat-index addressed).
    pub fn try_blocks(&self) -> &[TryBlock] {
        &self.flat.tries
    }

    // ── Phase 2: per-function init ────────────────────────────────────────

    /// **Phase 2** — initialise a new wasm function for the instruction at
    /// `flat_idx`.
    ///
    /// Rewinds the layout to the params mark, appends scratch locals, lets
    /// traps declare their per-function locals, then calls `reactor.next_with`
    /// with the non-param local declarations.
    ///
    /// `depth` is the yecta control-flow depth hint forwarded to `next_with`
    /// (typically `1` for sequential instructions).
    fn init_function(
        &mut self,
        ctx: &mut Context,
        _flat_idx: u64,
        depth: u32,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        self.layout.rewind(&self.locals_mark);
        // Per-function scratch locals for intermediate values.
        self.scratch_slot = self.layout.append(1, ValType::I32);
        self.scratch_i64_slot = self.layout.append(1, ValType::I64);
        self.traps.declare_locals(&mut self.layout);
        let mut locals_iter = self.layout.iter_since(&self.locals_mark);
        self.reactor.next_with(ctx, f(&mut locals_iter), depth)?;
        Ok(())
    }

    // ── Phase 3: instruction translation ──────────────────────────────────

    /// Translate the DEX instruction at code-unit `offset` within `method`.
    ///
    /// `f` is the wasm function factory (same contract as in the native arches).
    ///
    /// Uses `dex_bytecode::decode` to parse the instruction, fires
    /// [`InstructionTrap::on_instruction`] and, for control-flow instructions,
    /// [`JumpTrap::on_jump`], then emits the wasm body via `translate_body`.
    pub fn translate_instruction(
        &mut self,
        ctx: &mut Context,
        method: &FlatMethod,
        offset: usize,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        let flat_idx = method.base + offset as u64;

        // Each instruction becomes one wasm function.
        self.init_function(ctx, flat_idx, 1, f)?;

        let code_base = method.base as usize;
        let code_end = code_base + method.code_units as usize;
        let units_at = &self.flat.code[code_base + offset..code_end];

        let (insn, insn_len) = match dex_bytecode::decode(units_at) {
            Some(pair) => pair,
            None => {
                // Reserved opcode or pseudo-instruction marker — emit unreachable.
                self.reactor.feed(ctx, &Instruction::Unreachable)?;
                return Ok(());
            }
        };

        let class = insn_class(&insn);
        let insn_info = InstructionInfo {
            pc: flat_idx,
            len: insn_len as u32,
            arch: ArchTag::Dex,
            class,
        };

        // SAFETY: `traps` and `layout` are separate struct fields; the raw
        // pointer borrow does not alias the reactor borrow used by on_instruction.
        let layout_ptr = &self.layout as *const LocalLayout;
        if self
            .traps
            .on_instruction(&insn_info, ctx, &mut self.reactor, unsafe { &*layout_ptr })?
            == TrapAction::Skip
        {
            return Ok(());
        }

        if insn.is_control_flow() {
            let scratch = self.layout.base(self.scratch_slot);
            let jump_info = Self::build_jump_info(flat_idx, &insn, scratch);
            if self
                .traps
                .on_jump(&jump_info, ctx, &mut self.reactor, unsafe { &*layout_ptr })?
                == TrapAction::Skip
            {
                return Ok(());
            }
        }

        self.translate_body(ctx, flat_idx, &insn)?;
        Ok(())
    }

    /// Build a [`JumpInfo`] for a decoded control-flow instruction.
    fn build_jump_info(flat_idx: u64, insn: &DexInsn, scratch: u32) -> JumpInfo {
        let branch_target = |off: i64| (flat_idx as i64).wrapping_add(off) as u64;
        match insn {
            DexInsn::ReturnVoid | DexInsn::Return { .. }
            | DexInsn::ReturnWide { .. } | DexInsn::ReturnObject { .. }
            => JumpInfo::direct(flat_idx, 0, JumpKind::Return),

            DexInsn::Throw { .. }
            => JumpInfo::direct(flat_idx, 0, JumpKind::IndirectJump),

            DexInsn::Goto { offset } => JumpInfo::direct(
                flat_idx, branch_target(*offset as i64), JumpKind::DirectJump),
            DexInsn::Goto16 { offset } => JumpInfo::direct(
                flat_idx, branch_target(*offset as i64), JumpKind::DirectJump),
            DexInsn::Goto32 { offset } => JumpInfo::direct(
                flat_idx, branch_target(*offset as i64), JumpKind::DirectJump),

            DexInsn::PackedSwitch { .. } | DexInsn::SparseSwitch { .. }
            => JumpInfo::direct(flat_idx, 0, JumpKind::IndirectJump),

            DexInsn::IfEq { offset, .. } | DexInsn::IfNe { offset, .. }
            | DexInsn::IfLt { offset, .. } | DexInsn::IfGe { offset, .. }
            | DexInsn::IfGt { offset, .. } | DexInsn::IfLe { offset, .. }
            => JumpInfo::direct(flat_idx, branch_target(*offset as i64), JumpKind::ConditionalBranch),

            DexInsn::IfEqz { offset, .. } | DexInsn::IfNez { offset, .. }
            | DexInsn::IfLtz { offset, .. } | DexInsn::IfGez { offset, .. }
            | DexInsn::IfGtz { offset, .. } | DexInsn::IfLez { offset, .. }
            => JumpInfo::direct(flat_idx, branch_target(*offset as i64), JumpKind::ConditionalBranch),

            DexInsn::InvokeDirect { .. } | DexInsn::InvokeStatic { .. }
            | DexInsn::InvokeDirectRange { .. } | DexInsn::InvokeStaticRange { .. }
            | DexInsn::InvokeCustom { .. } | DexInsn::InvokeCustomRange { .. }
            => JumpInfo::direct(flat_idx, 0, JumpKind::Call),

            DexInsn::InvokeVirtual { .. } | DexInsn::InvokeSuper { .. }
            | DexInsn::InvokeInterface { .. } | DexInsn::InvokeVirtualRange { .. }
            | DexInsn::InvokeSuperRange { .. } | DexInsn::InvokeInterfaceRange { .. }
            | DexInsn::InvokePolymorphic { .. } | DexInsn::InvokePolymorphicRange { .. }
            => JumpInfo::indirect(flat_idx, scratch, JumpKind::IndirectCall),

            _ => JumpInfo::direct(flat_idx, 0, JumpKind::DirectJump),
        }
    }

    /// Emit the wasm body for a decoded DEX instruction.
    ///
    /// Non-object, non-array, non-invoke operations are fully translated.
    /// Unimplemented operations (object operations, array access, field access,
    /// invocations, switch) emit `unreachable` as a stub.
    fn translate_body(
        &mut self,
        ctx: &mut Context,
        flat_idx: u64,
        insn: &DexInsn,
    ) -> Result<(), E> {
        // Precompute scratch indices before any reactor borrows.
        let scratch = self.layout.base(self.scratch_slot);
        let scratch_i64 = self.layout.base(self.scratch_i64_slot);
        let total_params = self.total_params;
        let pool = self.pool;
        let bfo = self.reactor.base_func_offset();
        let escape_tag = self.escape_tag;

        // Compute target FuncIdx for a flat-index + signed branch offset.
        let target = |off: i64| -> FuncIdx {
            FuncIdx(bfo + (flat_idx as i64).wrapping_add(off) as u32)
        };

        macro_rules! feed {
            ($insn:expr) => { self.reactor.feed(ctx, &$insn)? };
        }

        match insn {
            // ── Nop ─────────────────────────────────────────────────────────
            DexInsn::Nop => { /* fall-through */ }

            // ── Move (i32) ──────────────────────────────────────────────────
            DexInsn::Move { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MoveFrom16 { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::Move16 { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Move (wide — two adjacent i32 regs) ─────────────────────────
            DexInsn::MoveWide { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32 + 1));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }
            DexInsn::MoveWideFrom16 { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32 + 1));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }
            DexInsn::MoveWide16 { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32 + 1));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }

            // MoveResult*, MoveObject*, MoveException — object/result-register
            // conventions require runtime support; stub.
            DexInsn::MoveResult { .. } | DexInsn::MoveResultWide { .. }
            | DexInsn::MoveResultObject { .. } | DexInsn::MoveException { .. }
            | DexInsn::MoveObject { .. } | DexInsn::MoveObjectFrom16 { .. }
            | DexInsn::MoveObject16 { .. }
            => { feed!(Instruction::Unreachable); }

            // ── Return ──────────────────────────────────────────────────────
            DexInsn::ReturnVoid | DexInsn::Return { .. }
            | DexInsn::ReturnWide { .. } | DexInsn::ReturnObject { .. } => {
                if let Some(tag) = escape_tag {
                    self.reactor.ret(ctx, total_params, tag)?;
                } else {
                    feed!(Instruction::Unreachable);
                }
            }

            // ── Constants (i32) ─────────────────────────────────────────────
            DexInsn::Const4 { dst, value } => {
                feed!(Instruction::I32Const(*value as i32));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::Const16 { dst, value } => {
                feed!(Instruction::I32Const(*value as i32));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::Const { dst, value } => {
                feed!(Instruction::I32Const(*value));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ConstHigh16 { dst, value } => {
                feed!(Instruction::I32Const((*value as i32) << 16));
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Constants (wide / i64 split to two i32 regs) ────────────────
            DexInsn::ConstWide16 { dst, value } => {
                let v = *value as i64;
                feed!(Instruction::I32Const(v as i32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::I32Const((v >> 32) as i32));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }
            DexInsn::ConstWide32 { dst, value } => {
                let v = *value as i64;
                feed!(Instruction::I32Const(v as i32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::I32Const((v >> 32) as i32));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }
            DexInsn::ConstWide { dst, value } => {
                feed!(Instruction::I32Const(*value as i32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::I32Const((*value >> 32) as i32));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }
            DexInsn::ConstWideHigh16 { dst, value } => {
                // Value = (value as i64) << 48
                let v = (*value as i64) << 48;
                feed!(Instruction::I32Const(v as i32));
                feed!(Instruction::LocalSet(*dst as u32));
                feed!(Instruction::I32Const((v >> 32) as i32));
                feed!(Instruction::LocalSet(*dst as u32 + 1));
            }

            // const-string / const-class — object operations; stub.
            DexInsn::ConstString { .. } | DexInsn::ConstStringJumbo { .. }
            | DexInsn::ConstClass { .. } | DexInsn::ConstMethodHandle { .. }
            | DexInsn::ConstMethodType { .. }
            => { feed!(Instruction::Unreachable); }

            // ── Unary int ops ────────────────────────────────────────────────
            DexInsn::NegInt { dst, src } => {
                feed!(Instruction::I32Const(0));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::NotInt { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(-1));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Unary long ops (wide pair) ───────────────────────────────────
            DexInsn::NegLong { dst, src } => {
                // Combine src pair into i64, negate, split to dst pair.
                feed!(Instruction::I64Const(0));
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Sub);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::NotLong { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Const(-1));
                feed!(Instruction::I64Xor);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }

            // ── Unary float/double ────────────────────────────────────────────
            DexInsn::NegFloat { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Neg);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::NegDouble { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Neg);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }

            // ── Type conversions ─────────────────────────────────────────────
            DexInsn::IntToLong { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I64ExtendI32S);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::IntToFloat { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ConvertI32S);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::IntToDouble { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F64ConvertI32S);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::LongToInt { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::LongToFloat { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F32ConvertI64S);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::LongToDouble { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ConvertI64S);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::FloatToInt { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::I32TruncSatF32S);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::FloatToLong { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::I64TruncSatF32S);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::FloatToDouble { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F64PromoteF32);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DoubleToInt { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::I32TruncSatF64S);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DoubleToLong { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::I64TruncSatF64S);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DoubleToFloat { dst, src } => {
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F32DemoteF64);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::IntToByte { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(24));
                feed!(Instruction::I32Shl);
                feed!(Instruction::I32Const(24));
                feed!(Instruction::I32ShrS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::IntToChar { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(0xffff));
                feed!(Instruction::I32And);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::IntToShort { dst, src } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(16));
                feed!(Instruction::I32Shl);
                feed!(Instruction::I32Const(16));
                feed!(Instruction::I32ShrS);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Binary int ops (23x: dst = a op b) ──────────────────────────
            DexInsn::AddInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Add);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::SubInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Mul);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32DivS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32RemS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::AndInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32And);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::OrInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Or);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::XorInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShlInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32Shl);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShrInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32ShrS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::UshrInt { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I32ShrU);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Binary int 2addr (12x: dst op= src) ─────────────────────────
            DexInsn::AddInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Add);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::SubInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Mul);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32DivS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32RemS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::AndInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32And);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::OrInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Or);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::XorInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShlInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Shl);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShrInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32ShrS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::UshrInt2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32ShrU);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Int lit16 (22s: dst = src op lit) ───────────────────────────
            DexInsn::AddIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Add);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RsubInt { dst, src, lit } => {
                // reverse-subtract: dst = lit - src
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Mul);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32DivS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32RemS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::AndIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32And);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::OrIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Or);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::XorIntLit16 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Int lit8 (22b: dst = src op lit) ────────────────────────────
            DexInsn::AddIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Add);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RsubIntLit8 { dst, src, lit } => {
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Mul);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32DivS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32RemS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::AndIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32And);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::OrIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Or);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::XorIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShlIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32Shl);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::ShrIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32ShrS);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::UshrIntLit8 { dst, src, lit } => {
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I32Const(*lit as i32));
                feed!(Instruction::I32ShrU);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Binary long ops (23x — wide pairs) ──────────────────────────
            DexInsn::AddLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64Add);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::SubLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64Sub);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::MulLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64Mul);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DivLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64DivS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::RemLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64RemS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::AndLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64And);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::OrLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64Or);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::XorLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::I64Xor);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            // Long shifts: value is a wide pair, shift amount is a single i32.
            DexInsn::ShlLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64Shl);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::ShrLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64ShrS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::UshrLong { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64ShrU);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }

            // ── Binary long 2addr ────────────────────────────────────────────
            DexInsn::AddLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Add);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::SubLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Sub);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::MulLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Mul);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DivLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64DivS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::RemLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64RemS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::AndLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64And);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::OrLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Or);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::XorLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::I64Xor);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::ShlLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64Shl);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::ShrLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64ShrS);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::UshrLong2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::I64ExtendI32U);
                feed!(Instruction::I64ShrU);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }

            // ── Binary float ops (i32 as f32 bit pattern) ───────────────────
            DexInsn::AddFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Add);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::SubFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Sub);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Mul);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Div);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemFloat { dst, a, b } => {
                // wasm has no f32.rem; emit unreachable (runtime call needed).
                let _ = (dst, a, b);
                feed!(Instruction::Unreachable);
            }

            // ── Float 2addr ──────────────────────────────────────────────────
            DexInsn::AddFloat2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Add);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::SubFloat2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Sub);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::MulFloat2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Mul);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::DivFloat2addr { dst, src } => {
                feed!(Instruction::LocalGet(*dst as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*src as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Div);
                feed!(Instruction::I32ReinterpretF32);
                feed!(Instruction::LocalSet(*dst as u32));
            }
            DexInsn::RemFloat2addr { dst, src } => {
                let _ = (dst, src);
                feed!(Instruction::Unreachable);
            }

            // ── Binary double ops (wide pair as f64 bit pattern) ─────────────
            DexInsn::AddDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Add);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::SubDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Sub);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::MulDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Mul);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DivDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Div);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::RemDouble { dst, a, b } => {
                let _ = (dst, a, b);
                feed!(Instruction::Unreachable);
            }

            // ── Double 2addr ─────────────────────────────────────────────────
            DexInsn::AddDouble2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Add);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::SubDouble2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Sub);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::MulDouble2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Mul);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::DivDouble2addr { dst, src } => {
                self.emit_wide_to_stack(ctx, *dst)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *src)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Div);
                feed!(Instruction::I64ReinterpretF64);
                self.emit_split_wide(ctx, *dst, scratch_i64)?;
            }
            DexInsn::RemDouble2addr { dst, src } => {
                let _ = (dst, src);
                feed!(Instruction::Unreachable);
            }

            // ── Comparisons ──────────────────────────────────────────────────

            // cmpl-float: NaN → -1
            // result = (a > b) - !(a >= b)
            DexInsn::CmplFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Gt);
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Ge);
                feed!(Instruction::I32Const(1));
                feed!(Instruction::I32Xor);
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // cmpg-float: NaN → +1
            // result = !(a <= b) - (a < b)
            DexInsn::CmpgFloat { dst, a, b } => {
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Le);
                feed!(Instruction::I32Const(1));
                feed!(Instruction::I32Xor);
                feed!(Instruction::LocalGet(*a as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::LocalGet(*b as u32));
                feed!(Instruction::F32ReinterpretI32);
                feed!(Instruction::F32Lt);
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // cmpl-double: NaN → -1
            DexInsn::CmplDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Gt);
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Ge);
                feed!(Instruction::I32Const(1));
                feed!(Instruction::I32Xor);
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // cmpg-double: NaN → +1
            DexInsn::CmpgDouble { dst, a, b } => {
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Le);
                feed!(Instruction::I32Const(1));
                feed!(Instruction::I32Xor);
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::F64ReinterpretI64);
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::F64ReinterpretI64);
                feed!(Instruction::F64Lt);
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // cmp-long: result = (a > b) - (a < b), -1/0/1
            // Store b pair in scratch_i64, compute (a > scratch) - (a < scratch).
            DexInsn::CmpLong { dst, a, b } => {
                // Store b as i64 in scratch.
                self.emit_wide_to_stack(ctx, *b)?;
                feed!(Instruction::LocalSet(scratch_i64));
                // (a > b): combine a, get scratch, compare.
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::LocalGet(scratch_i64));
                feed!(Instruction::I64GtS);
                // (a < b): combine a again, get scratch.
                self.emit_wide_to_stack(ctx, *a)?;
                feed!(Instruction::LocalGet(scratch_i64));
                feed!(Instruction::I64LtS);
                feed!(Instruction::I32Sub);
                feed!(Instruction::LocalSet(*dst as u32));
            }

            // ── Unconditional branches ───────────────────────────────────────
            DexInsn::Goto { offset } => {
                let tgt = target(*offset as i64);
                self.reactor.jmp(ctx, tgt, total_params)?;
            }
            DexInsn::Goto16 { offset } => {
                let tgt = target(*offset as i64);
                self.reactor.jmp(ctx, tgt, total_params)?;
            }
            DexInsn::Goto32 { offset } => {
                let tgt = target(*offset as i64);
                self.reactor.jmp(ctx, tgt, total_params)?;
            }

            // ── Conditional branches (two-register) ─────────────────────────
            DexInsn::IfEq { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32Eq };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfNe { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32Ne };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfLt { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32LtS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfGe { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32GeS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfGt { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32GtS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfLe { a, b, offset } => {
                let cond = DexCondSnippet { a: *a as u32, b: *b as u32, op: DexCondOp::I32LeS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }

            // ── Conditional branches (register vs zero) ──────────────────────
            DexInsn::IfEqz { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32Eqz };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfNez { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32Nez };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfLtz { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32LtzS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfGez { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32GezS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfGtz { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32GtzS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }
            DexInsn::IfLez { reg, offset } => {
                let cond = DexCondSnippet { a: *reg as u32, b: 0, op: DexCondOp::I32LezS };
                let tgt = target(*offset as i64);
                self.reactor.ji(ctx, total_params, &BTreeMap::new(), Target::Static { func: tgt }, None, pool, Some(&cond))?;
            }

            // ── Not yet implemented: object/array/field/invoke/switch ────────
            _ => {
                feed!(Instruction::Unreachable);
            }
        }
        Ok(())
    }

    /// Emit: combine adjacent i32 register pair `(v, v+1)` into an i64 on the
    /// wasm stack (leaves the i64 value on the stack).
    ///
    /// The low register `v` contributes the low 32 bits; `v+1` the high 32 bits.
    fn emit_wide_to_stack(&mut self, ctx: &mut Context, v: u8) -> Result<(), E> {
        // low
        self.reactor.feed(ctx, &Instruction::LocalGet(v as u32))?;
        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
        // high << 32
        self.reactor.feed(ctx, &Instruction::LocalGet(v as u32 + 1))?;
        self.reactor.feed(ctx, &Instruction::I64ExtendI32U)?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64Shl)?;
        // low | (high << 32)
        self.reactor.feed(ctx, &Instruction::I64Or)?;
        Ok(())
    }

    /// Emit: split the i64 in `scratch_i64` into two adjacent i32 registers
    /// `(dst, dst+1)`.  `scratch_i64` is overwritten with the high half.
    fn emit_split_wide(&mut self, ctx: &mut Context, dst: u8, scratch_i64: u32) -> Result<(), E> {
        // tee scratch_i64 so we can use it twice.
        self.reactor.feed(ctx, &Instruction::LocalTee(scratch_i64))?;
        // low 32 bits → dst
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(dst as u32))?;
        // high 32 bits → dst+1
        self.reactor.feed(ctx, &Instruction::LocalGet(scratch_i64))?;
        self.reactor.feed(ctx, &Instruction::I64Const(32))?;
        self.reactor.feed(ctx, &Instruction::I64ShrU)?;
        self.reactor.feed(ctx, &Instruction::I32WrapI64)?;
        self.reactor.feed(ctx, &Instruction::LocalSet(dst as u32 + 1))?;
        Ok(())
    }

    /// Translate all instructions in `method`, invoking `f` once per instruction
    /// to construct the wasm function body.
    ///
    /// The instruction walk uses `dex_bytecode::decode` for standard instructions
    /// and `dex_bytecode::pseudo_len` for variable-length data tables
    /// (`packed-switch-data`, `sparse-switch-data`, `fill-array-data`),
    /// skipping data tables silently.
    pub fn translate_method(
        &mut self,
        ctx: &mut Context,
        method_idx: usize,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        // Clone just the metadata — code is borrowed from self.flat.code below.
        let method = self.flat.methods[method_idx].clone();
        let base = method.base as usize;

        let mut offset = 0usize;
        while offset < method.code_units as usize {
            let abs = base + offset;
            if abs >= self.flat.code.len() {
                break;
            }
            let units_at = &self.flat.code[abs..];

            // Pseudo-instruction data tables start with 0x00 + non-zero high byte.
            if units_at[0] & 0xff == 0x00 && (units_at[0] >> 8) != 0 {
                let len = dex_bytecode::pseudo_len(units_at).max(1);
                offset += len;
                continue;
            }

            let insn_len = dex_bytecode::decode(units_at)
                .map(|(_, len)| len)
                .unwrap_or(1);
            self.translate_instruction(ctx, &method, offset, f)?;
            offset += insn_len;
        }
        Ok(())
    }

    /// Translate every method in the DEX file in parsing order.
    pub fn translate_all(
        &mut self,
        ctx: &mut Context,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        for idx in 0..self.flat.methods.len() {
            self.translate_method(ctx, idx, f)?;
        }
        Ok(())
    }
}
