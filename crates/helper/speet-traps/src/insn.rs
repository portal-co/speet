//! [`InstructionTrap`] — per-instruction trap hook.
//!
//! An `InstructionTrap` fires once for every translated guest instruction,
//! immediately after the yecta function is opened and the PC local is written
//! but **before** the instruction body is emitted.  This ordering means the
//! trap can emit a preamble (e.g. increment a counter, perform a context-switch
//! check) and then either allow normal translation to proceed or replace it
//! entirely.
//!
//! ## Firing order within a single instruction
//!
//! ```text
//! init_function(…)                  ← opens new yecta function, declares locals
//! reactor.feed(I32Const(pc))        ← writes PC local
//! reactor.feed(LocalSet(pc_local))  ←
//! TrapConfig::on_instruction(…)     ← InstructionTrap fires HERE
//! <instruction body>                ← only if TrapAction::Continue
//! ```
//!
//! ## `TrapAction`
//!
//! Returning [`TrapAction::Skip`] causes the recompiler to suppress the
//! instruction body and instead emit the trap's
//! [`InstructionTrap::skip_snippet`] (which defaults to `unreachable` if the
//! trap does not override it).  The recompiler must still emit a valid wasm
//! function terminator; `unreachable` satisfies that requirement.
//!
//! ## Local / parameter declaration
//!
//! Traps that need wasm locals or parameters override
//! [`declare_locals`](InstructionTrap::declare_locals) and/or
//! [`declare_params`](InstructionTrap::declare_params).  Both methods receive
//! a `&mut LocalLayout` to which the trap appends its groups via
//! [`LocalLayout::append`].  The returned [`LocalSlot`] handles should be
//! stored in the trap struct and used later (inside `on_instruction` /
//! `skip_snippet`) via `trap_ctx.locals().local(slot, n)` and
//! `trap_ctx.params().local(slot, n)`.
//!
//! `declare_params` is called **once** at trap-installation time (during
//! [`TrapConfig::setup`]).  `declare_locals` is also called once at
//! installation time; the base offset of the locals layout is updated per
//! function via [`TrapConfig::set_local_base`].
//!
//! ## Blanket impl for closures
//!
//! Any `FnMut(&InstructionInfo, &mut Context, &mut TrapContext<…>) -> Result<TrapAction, E>`
//! closure automatically implements `InstructionTrap`, so one-off traps can
//! be written inline without declaring a struct.

use alloc::{boxed::Box, vec::Vec};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::{LocalLayout, LocalSlot};

use crate::context::TrapContext;

// ── Supporting types ──────────────────────────────────────────────────────────

/// Which architecture produced the instruction being trapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchTag {
    RiscV,
    Mips,
    X86_64,
    PowerPC,
    Dex,
    Other,
}

/// Bit-mask of instruction semantic classes.
///
/// A single instruction may belong to multiple classes simultaneously
/// (e.g. an atomic memory operation is both `MEMORY` and `ATOMIC`).
/// Use bitwise-OR to combine flags and [`InsnClass::contains`] to test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct InsnClass(pub u32);

impl InsnClass {
    /// No class bits set — catches only instructions with `class == OTHER`.
    pub const OTHER: InsnClass = InsnClass(0);
    /// Instruction accesses linear memory (load or store).
    pub const MEMORY: InsnClass = InsnClass(1 << 0);
    /// Control-flow transfer: branch or conditional jump.
    pub const BRANCH: InsnClass = InsnClass(1 << 1);
    /// Function call (direct or indirect).
    pub const CALL: InsnClass = InsnClass(1 << 2);
    /// Function return.
    pub const RETURN: InsnClass = InsnClass(1 << 3);
    /// Privileged / system instruction (ECALL, SYSCALL, INT, …).
    pub const PRIVILEGED: InsnClass = InsnClass(1 << 4);
    /// Floating-point operation.
    pub const FLOAT: InsnClass = InsnClass(1 << 5);
    /// Atomic memory operation (AMO / LL-SC / CMPXCHG).
    pub const ATOMIC: InsnClass = InsnClass(1 << 6);
    /// Target is computed at runtime (indirect branch/call/jump).
    pub const INDIRECT: InsnClass = InsnClass(1 << 7);

    /// Return `true` if every bit in `other` is also set in `self`.
    #[inline]
    pub fn contains(self, other: InsnClass) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl core::ops::BitOr for InsnClass {
    type Output = InsnClass;
    fn bitor(self, rhs: InsnClass) -> InsnClass {
        InsnClass(self.0 | rhs.0)
    }
}

/// Metadata about the instruction that fired the trap.
pub struct InstructionInfo {
    /// Guest program counter (byte address, before any base-offset subtraction).
    pub pc: u64,
    /// Byte length of the instruction in the guest ISA.
    pub len: u32,
    /// Which architecture emitted this instruction.
    pub arch: ArchTag,
    /// Semantic class of the instruction (may have multiple bits set).
    pub class: InsnClass,
}

/// Whether the recompiler should proceed normally or suppress the instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapAction {
    /// Continue with normal translation of the instruction body.
    Continue,
    /// Suppress the instruction body; use `skip_snippet` instead.
    Skip,
}

// ── InstructionTrap trait ─────────────────────────────────────────────────────

/// Fires once per translated guest instruction.
///
/// See the [module documentation](self) for the full description of when this
/// fires and what the trap can do.
///
/// # Type parameters
///
/// * `Context` — the recompiler's user context type.
/// * `E` — the error type returned by the recompiler's instruction sink.
/// * `F` — the concrete [`InstructionSink`] the recompiler uses (usually
///   `Reactor<Context, E, OuterF>`).  The trap receives a
///   [`TrapContext<Context, E, F>`] through which it emits wasm.
pub trait InstructionTrap<Context, E, F: InstructionSink<Context, E>> {
    /// Called once per instruction, before the instruction body is emitted.
    ///
    /// Return [`TrapAction::Continue`] to let normal translation proceed, or
    /// [`TrapAction::Skip`] to suppress it.
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E>;

    /// Append wasm **parameter** slots to `params` and store the returned
    /// [`LocalSlot`] handles for later index resolution.
    ///
    /// Parameters (locals `0..total_params-1`) survive `return_call` chains.
    /// Use them for state that must carry over from one translated-instruction
    /// wasm function to the next (e.g. a depth counter).
    ///
    /// Called **once** at trap-installation time by [`TrapConfig::setup`].
    /// The default does nothing (no extra parameters).
    #[allow(unused_variables)]
    fn declare_params(&mut self, params: &mut LocalLayout) {}

    /// Append wasm **local** slots to `locals` and store the returned
    /// [`LocalSlot`] handles for later index resolution.
    ///
    /// Non-parameter locals are reset to zero at the start of each new wasm
    /// function.  Use them for state that is only needed within a single
    /// translated instruction (e.g. a scratch index for a bitmap lookup).
    ///
    /// Called **once** at trap-installation time by
    /// [`TrapConfig::set_instruction_trap`].  The base offset of the locals
    /// layout is updated per function via [`TrapConfig::set_local_base`].
    ///
    /// The default does nothing (no extra locals).
    #[allow(unused_variables)]
    fn declare_locals(&mut self, locals: &mut LocalLayout) {}

    /// Wasm instructions to emit in place of the instruction body when this
    /// trap returns [`TrapAction::Skip`].
    ///
    /// The default emits a single `unreachable`, which is always a valid
    /// function terminator.  Traps that want to redirect control flow (e.g.
    /// jump to a violation handler) should override this.
    fn skip_snippet(
        &self,
        info: &InstructionInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        let _ = info;
        skip_ctx.emit(ctx, &Instruction::Unreachable)
    }
}

// ── Blanket impl for FnMut closures ──────────────────────────────────────────

impl<Context, E, F, Fn> InstructionTrap<Context, E, F> for Fn
where
    F: InstructionSink<Context, E>,
    Fn: FnMut(&InstructionInfo, &mut Context, &mut TrapContext<Context, E, F>)
            -> Result<TrapAction, E>,
{
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        self(info, ctx, trap_ctx)
    }
}

// ── Vec<Box<dyn InstructionTrap>> ─────────────────────────────────────────────

/// `Vec<Box<dyn InstructionTrap<…>>>` implements `InstructionTrap` by running
/// each element in order and short-circuiting on the first `Skip`.
///
/// The `skip_snippet` of the first element that returned `Skip` is used.
///
/// **Note on `declare_locals` / `declare_params`:** each element's declaration
/// methods must be called individually before the vec is installed.  The vec
/// impl delegates both methods to all elements in order, so installing the vec
/// via [`TrapConfig::set_instruction_trap`] will call `declare_*` on each
/// element once and append all their slots to the shared layout.
impl<Context, E, F> InstructionTrap<Context, E, F>
    for Vec<Box<dyn InstructionTrap<Context, E, F> + '_>>
where
    F: InstructionSink<Context, E>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        for trap in self.iter_mut() {
            trap.declare_params(params);
        }
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        for trap in self.iter_mut() {
            trap.declare_locals(locals);
        }
    }

    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        for trap in self.iter_mut() {
            if trap.on_instruction(info, ctx, trap_ctx)? == TrapAction::Skip {
                return Ok(TrapAction::Skip);
            }
        }
        Ok(TrapAction::Continue)
    }
}

/// `Box<dyn InstructionTrap<…>>` simply delegates to the inner value.
impl<Context, E, F> InstructionTrap<Context, E, F>
    for Box<dyn InstructionTrap<Context, E, F> + '_>
where
    F: InstructionSink<Context, E>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        (**self).declare_params(params);
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        (**self).declare_locals(locals);
    }

    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        (**self).on_instruction(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &InstructionInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        (**self).skip_snippet(info, ctx, skip_ctx)
    }
}
