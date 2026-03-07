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

use alloc::vec::Vec;
use wax_core::build::InstructionSink;

use dex::{DexReader, code::ExceptionType, jtype::Type, string::DexString};
use wasm_encoder::{Instruction, ValType};
use yecta::{
    EscapeTag, LocalLayout, LocalPoolBackend, LocalSlot, Mark, Pool, Reactor,
    TableIdx, TypeIdx,
};

pub use speet_memory::{CallbackContext, MapperCallback};
pub use speet_ordering::{AtomicOpts, MemOrder, RmwOp, RmwWidth};
use speet_traps::{
    InstructionInfo, InstructionTrap, JumpInfo, JumpKind, JumpTrap, TrapAction, TrapConfig,
    insn::{ArchTag, InsnClass},
};

// ── DEX opcode table ─────────────────────────────────────────────────────────

/// A raw DEX opcode (low byte of the first 16-bit code unit of an instruction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DexOpcode(pub u8);

impl DexOpcode {
    /// Number of 16-bit code units this instruction occupies.
    ///
    /// Returns `1` for unknown/unused opcodes so the instruction walker can
    /// always advance.  Returns `None` for variable-length pseudo-instructions
    /// (`packed-switch-data`, `sparse-switch-data`, `fill-array-data`), which
    /// must be sized at the call site.
    pub fn code_units(self) -> Option<usize> {
        // Lengths from the Dalvik Executable Format specification.
        // https://source.android.com/docs/core/runtime/dalvik-bytecode
        Some(match self.0 {
            // ── 1-unit instructions ────────────────────────────────────────
            0x00 => 1, // nop
            0x01 => 1, // move vA, vB
            0x04 => 1, // move-wide vA, vB
            0x07 => 1, // move-object vA, vB
            0x0a => 1, // move-result vAA
            0x0b => 1, // move-result-wide vAA
            0x0c => 1, // move-result-object vAA
            0x0d => 1, // move-exception vAA
            0x0e => 1, // return-void
            0x0f => 1, // return vAA
            0x10 => 1, // return-wide vAA
            0x11 => 1, // return-object vAA
            0x12 => 1, // const/4 vA, #+B
            0x1d => 1, // monitor-enter vAA
            0x1e => 1, // monitor-exit vAA
            0x21 => 1, // array-length vA, vB
            0x27 => 1, // throw vAA
            0x28 => 1, // goto +AA
            // 2addr unary and binary ops (12x format)
            0x7b..=0x8f => 1, // neg-int .. int-to-short (unary)
            0xb0..=0xcf => 1, // add-int/2addr .. rem-double/2addr
            // unused / reserved (treat as 1-unit nops)
            0x3e..=0x43 => 1,
            0x73 => 1,
            0x79 => 1,
            0x7a => 1,
            0xe3..=0xf9 => 1,

            // ── 2-unit instructions ────────────────────────────────────────
            0x02 => 2, // move/from16 vAA, vBBBB
            0x05 => 2, // move-wide/from16 vAA, vBBBB
            0x08 => 2, // move-object/from16 vAA, vBBBB
            0x13 => 2, // const/16 vAA, #+BBBB
            0x15 => 2, // const/high16 vAA, #+BBBB0000
            0x16 => 2, // const-wide/16 vAA, #+BBBB
            0x19 => 2, // const-wide/high16 vAA, #+BBBB000000000000
            0x1a => 2, // const-string vAA, string@BBBB
            0x1c => 2, // const-class vAA, type@BBBB
            0x1f => 2, // check-cast vAA, type@BBBB
            0x20 => 2, // instance-of vA, vB, type@CCCC
            0x22 => 2, // new-instance vAA, type@BBBB
            0x23 => 2, // new-array vA, vB, type@CCCC
            0x29 => 2, // goto/16 +AAAA
            // cmp* (23x)
            0x2d..=0x31 => 2,
            // if-* (22t), if-*z (21t)
            0x32..=0x3d => 2,
            // array ops (23x)
            0x44..=0x51 => 2,
            // iget/iput (22c)
            0x52..=0x5f => 2,
            // sget/sput (21c)
            0x60..=0x6d => 2,
            // binary ALU (23x): add-int .. rem-double
            0x90..=0xaf => 2,
            // binary lit16 (22s)
            0xd0..=0xd7 => 2,
            // binary lit8 (22b)
            0xd8..=0xe2 => 2,
            // const-method-handle / const-method-type (21c)
            0xfe => 2,
            0xff => 2,

            // ── 3-unit instructions ────────────────────────────────────────
            0x03 => 3, // move/16 vAAAA, vBBBB
            0x06 => 3, // move-wide/16 vAAAA, vBBBB
            0x09 => 3, // move-object/16 vAAAA, vBBBB
            0x14 => 3, // const vAA, #+BBBBBBBB
            0x17 => 3, // const-wide/32 vAA, #+BBBBBBBB
            0x1b => 3, // const-string/jumbo vAA, string@BBBBBBBB
            0x24 => 3, // filled-new-array {vC..vG}, type@BBBB (35c)
            0x25 => 3, // filled-new-array/range {vCCCC..vNNNN}, type@BBBB (3rc)
            0x26 => 3, // fill-array-data vAA, +BBBBBBBB (31t)
            0x2a => 3, // goto/32 +AAAAAAAA
            0x2b => 3, // packed-switch vAA, +BBBBBBBB (31t)
            0x2c => 3, // sparse-switch vAA, +BBBBBBBB (31t)
            // invoke-* (35c, 3rc)
            0x6e..=0x72 => 3,
            0x74..=0x78 => 3,
            // invoke-custom (35c, 3rc)
            0xfc => 3,
            0xfd => 3,

            // ── 4-unit instructions ────────────────────────────────────────
            0xfa => 4, // invoke-polymorphic (45cc)
            0xfb => 4, // invoke-polymorphic/range (4rcc)

            // ── 5-unit instructions ────────────────────────────────────────
            0x18 => 5, // const-wide vAA, #+BBBBBBBBBBBBBBBB

            // Variable-length pseudo-instructions — caller must handle
            // 0x00 with high-byte 0x01/0x02/0x03: packed-switch-data,
            //   sparse-switch-data, fill-array-data
            // We expose these through `pseudo_len` instead.
            _ => return None,
        })
    }

    /// Returns `true` if this opcode is a control-flow transfer.
    pub fn is_jump(self) -> bool {
        matches!(
            self.0,
            0x0e // return-void
            | 0x0f // return
            | 0x10 // return-wide
            | 0x11 // return-object
            | 0x27 // throw
            | 0x28 // goto
            | 0x29 // goto/16
            | 0x2a // goto/32
            | 0x2b // packed-switch
            | 0x2c // sparse-switch
            | 0x32..=0x3d // if-* / if-*z
            | 0x6e..=0x72 // invoke-*
            | 0x74..=0x78 // invoke-*/range
            | 0xfa | 0xfb // invoke-polymorphic
            | 0xfc | 0xfd // invoke-custom
        )
    }

    /// Classify into [`InsnClass`] flags for the trap system.
    pub fn classify(self) -> InsnClass {
        match self.0 {
            // Memory access
            0x44..=0x51 // aget / aput
            | 0x52..=0x5f // iget / iput
            | 0x60..=0x6d // sget / sput
            => InsnClass::MEMORY,
            // Returns
            0x0e..=0x11 => InsnClass::RETURN,
            // Throw (privilege / exceptional control flow)
            0x27 => InsnClass::PRIVILEGED,
            // Goto (unconditional direct jump)
            0x28..=0x2a => InsnClass::BRANCH,
            // Switch (indirect branch)
            0x2b | 0x2c => InsnClass::BRANCH | InsnClass::INDIRECT,
            // Conditional branches
            0x32..=0x3d => InsnClass::BRANCH,
            // Direct calls
            0x6e | 0x70 | 0x74 | 0x76 | 0xfc | 0xfd => InsnClass::CALL,
            // Indirect / virtual calls
            0x6f | 0x71 | 0x72 | 0x75 | 0x77 | 0x78
            | 0xfa | 0xfb => InsnClass::CALL | InsnClass::INDIRECT,
            // Floating-point ALU
            0x2d | 0x2e // cmpl/cmpg-float
            | 0x96 | 0x97 // add-float / sub-float
            | 0x98 | 0x99 // mul-float / div-float
            | 0x9a         // rem-float
            | 0xa6..=0xaa  // add-double .. rem-double
            | 0xb6..=0xbb  // add-float/2addr .. rem-float/2addr
            | 0xbc..=0xc1  // add-double/2addr .. rem-double/2addr
            | 0x85..=0x8f  // int-to-float .. long-to-float (casts)
            => InsnClass::FLOAT,
            _ => InsnClass::OTHER,
        }
    }
}

/// Length of a variable-length DEX pseudo-instruction starting at `units[0]`.
///
/// Called when the opcode byte is `0x00` and the high byte of the first
/// code unit distinguishes the pseudo-instruction kind.
pub fn pseudo_len(units: &[u16]) -> usize {
    if units.is_empty() {
        return 1;
    }
    match units[0] >> 8 {
        0x01 => {
            // packed-switch-data: size = 4 + 2*N
            let n = units.get(1).copied().unwrap_or(0) as usize;
            4 + 2 * n
        }
        0x02 => {
            // sparse-switch-data: size = 2 + 4*N
            let n = units.get(1).copied().unwrap_or(0) as usize;
            2 + 4 * n
        }
        0x03 => {
            // fill-array-data: size = 4 + ceil(N*element_width / 2)
            let element_width = units.get(1).copied().unwrap_or(1) as usize;
            let n = if units.len() >= 4 {
                let lo = units[2] as u32;
                let hi = units[3] as u32;
                ((hi << 16) | lo) as usize
            } else {
                0
            };
            4 + (n * element_width + 1) / 2
        }
        _ => 1, // unknown — skip 1 unit
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
        // Per-function scratch local for intermediate values.
        self.scratch_slot = self.layout.append(1, ValType::I32);
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
    /// Fires [`InstructionTrap::on_instruction`] and, for control-flow
    /// instructions, [`JumpTrap::on_jump`].  The instruction body is currently
    /// a stub (`unreachable`) — a full DEX instruction translator will be added
    /// when the DEX instruction parser is integrated.
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

        let code_slice = &self.flat.code
            [method.base as usize..][..method.code_units as usize];

        let first_unit = code_slice.get(offset).copied().unwrap_or(0);
        let opcode = DexOpcode((first_unit & 0xff) as u8);
        let class = opcode.classify();

        // Fire instruction trap.
        let insn_len = opcode.code_units().unwrap_or_else(|| {
            pseudo_len(&code_slice[offset..])
        });
        let insn_info = InstructionInfo {
            pc: flat_idx,
            len: insn_len as u32,
            arch: ArchTag::Dex,
            class,
        };
        let layout_ptr = &self.layout as *const LocalLayout;
        if self
            .traps
            .on_instruction(&insn_info, ctx, &mut self.reactor, unsafe { &*layout_ptr })?
            == TrapAction::Skip
        {
            return Ok(());
        }

        // Fire jump trap for control-flow instructions.
        if opcode.is_jump() {
            let jump_info = self.build_jump_info(opcode, flat_idx, code_slice, offset, insn_len);
            if self
                .traps
                .on_jump(&jump_info, ctx, &mut self.reactor, unsafe { &*layout_ptr })?
                == TrapAction::Skip
            {
                return Ok(());
            }
        }

        // TODO: Translate opcode body.  Emit unreachable until the DEX
        // instruction parser is integrated.
        self.reactor.feed(ctx, &Instruction::Unreachable)?;
        Ok(())
    }

    /// Build a [`JumpInfo`] for a known control-flow instruction.
    fn build_jump_info(
        &self,
        opcode: DexOpcode,
        flat_idx: u64,
        code: &[u16],
        offset: usize,
        _len: usize,
    ) -> JumpInfo {
        match opcode.0 {
            // return-void / return / return-wide / return-object
            0x0e..=0x11 => JumpInfo::direct(flat_idx, 0, JumpKind::Return),

            // throw
            0x27 => JumpInfo::direct(flat_idx, 0, JumpKind::IndirectJump),

            // goto +AA (sign-extended 8-bit offset in bits 8-15 of first unit)
            0x28 => {
                let off = ((code[offset] >> 8) as i8) as i64;
                let target = (flat_idx as i64).wrapping_add(off) as u64;
                JumpInfo::direct(flat_idx, target, JumpKind::DirectJump)
            }

            // goto/16 +AAAA (16-bit signed offset in second unit)
            0x29 => {
                let off = code.get(offset + 1).copied().unwrap_or(0) as i16 as i64;
                let target = (flat_idx as i64).wrapping_add(off) as u64;
                JumpInfo::direct(flat_idx, target, JumpKind::DirectJump)
            }

            // goto/32 +AAAAAAAA (32-bit signed offset in units 1-2)
            0x2a => {
                let lo = code.get(offset + 1).copied().unwrap_or(0) as u32;
                let hi = code.get(offset + 2).copied().unwrap_or(0) as u32;
                let off = ((hi << 16) | lo) as i32 as i64;
                let target = (flat_idx as i64).wrapping_add(off) as u64;
                JumpInfo::direct(flat_idx, target, JumpKind::DirectJump)
            }

            // packed-switch / sparse-switch (indirect via table)
            0x2b | 0x2c => JumpInfo::direct(flat_idx, 0, JumpKind::IndirectJump),

            // if-* (22t): offset in bits 0-15 of second unit
            0x32..=0x37 => {
                let off = code.get(offset + 1).copied().unwrap_or(0) as i16 as i64;
                let target = (flat_idx as i64).wrapping_add(off) as u64;
                JumpInfo::direct(flat_idx, target, JumpKind::ConditionalBranch)
            }

            // if-*z (21t): offset in bits 0-15 of second unit
            0x38..=0x3d => {
                let off = code.get(offset + 1).copied().unwrap_or(0) as i16 as i64;
                let target = (flat_idx as i64).wrapping_add(off) as u64;
                JumpInfo::direct(flat_idx, target, JumpKind::ConditionalBranch)
            }

            // invoke-* / invoke-*/range: static/direct → Call; virtual/interface → indirect
            0x6e | 0x70 | 0x74 | 0x76 | 0xfc | 0xfd => {
                JumpInfo::direct(flat_idx, 0, JumpKind::Call)
            }
            0x6f | 0x71 | 0x72 | 0x75 | 0x77 | 0x78 | 0xfa | 0xfb => {
                // Use the scratch local as a placeholder target local.
                let scratch = self.layout.base(self.scratch_slot);
                JumpInfo::indirect(flat_idx, scratch, JumpKind::IndirectCall)
            }

            _ => JumpInfo::direct(flat_idx, 0, JumpKind::DirectJump),
        }
    }

    /// Translate all instructions in `method`, invoking `f` once per instruction
    /// to construct the wasm function body.
    ///
    /// The instruction walk handles both standard and variable-length
    /// pseudo-instructions (`packed-switch-data`, `sparse-switch-data`,
    /// `fill-array-data`), skipping pseudo-instructions without calling traps
    /// (they are data, not executable code).
    pub fn translate_method(
        &mut self,
        ctx: &mut Context,
        method_idx: usize,
        f: &mut (dyn FnMut(&mut (dyn Iterator<Item = (u32, ValType)> + '_)) -> F + '_),
    ) -> Result<(), E> {
        // Clone just the metadata — code is borrowed from self.flat.code below.
        let method = self.flat.methods[method_idx].clone();
        let base = method.base as usize;
        let end = base + method.code_units as usize;

        let mut offset = 0usize;
        while offset < method.code_units as usize {
            let abs = base + offset;
            if abs >= self.flat.code.len() {
                break;
            }
            let first_unit = self.flat.code[abs];
            let opcode = DexOpcode((first_unit & 0xff) as u8);

            // Variable-length pseudo-instructions (data, not code — skip silently).
            if opcode.0 == 0x00 && (first_unit >> 8) != 0 {
                let len = pseudo_len(&self.flat.code[abs..]);
                offset += len.max(1);
                continue;
            }

            let len = opcode.code_units().unwrap_or(1);
            self.translate_instruction(ctx, &method, offset, f)?;
            offset += len;
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
