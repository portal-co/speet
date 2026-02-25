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
//! ## Blanket impl for closures
//!
//! Any `FnMut(&InstructionInfo, &mut Context, &mut TrapContext<…>) -> Result<TrapAction, E>`
//! closure automatically implements `InstructionTrap`, so one-off traps can
//! be written inline without declaring a struct.

use alloc::{boxed::Box, vec::Vec};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

use crate::context::TrapContext;
use crate::layout::ExtraParams;
use crate::locals::ExtraLocals;

// ── Supporting types ──────────────────────────────────────────────────────────

/// Which architecture produced the instruction being trapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchTag {
    RiscV,
    Mips,
    X86_64,
    PowerPC,
    Wasm,
    Other,
}

/// Coarse opcode classification bitfield.
///
/// Multiple flags may be set simultaneously (e.g. a `JALR` is both a `BRANCH`
/// and a `CALL`).  Architecture recompilers set these flags when calling
/// [`TrapConfig::on_instruction`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct InsnClass(pub u32);

impl InsnClass {
    /// Instruction has no special classification.
    pub const OTHER: Self = Self(0);
    /// Instruction accesses linear memory (load or store).
    pub const MEMORY: Self = Self(1 << 0);
    /// Conditional or unconditional branch (does not save a return address).
    pub const BRANCH: Self = Self(1 << 1);
    /// Subroutine call (saves a return address).
    pub const CALL: Self = Self(1 << 2);
    /// Return from subroutine.
    pub const RETURN: Self = Self(1 << 3);
    /// Privileged / system instruction (ECALL, SYSCALL, INT, …).
    pub const PRIVILEGED: Self = Self(1 << 4);
    /// Floating-point instruction.
    pub const FLOAT: Self = Self(1 << 5);
    /// Atomic memory operation.
    pub const ATOMIC: Self = Self(1 << 6);
    /// Instruction target is computed at runtime (indirect branch/call).
    pub const INDIRECT: Self = Self(1 << 7);

    /// Returns `true` if all bits of `other` are set in `self`.
    #[inline]
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Combine two classifications (bitwise OR).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl core::ops::BitOr for InsnClass {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Metadata about the guest instruction currently being translated.
///
/// Passed by the recompiler to [`TrapConfig::on_instruction`] and forwarded
/// to the active [`InstructionTrap`].
#[derive(Clone, Debug)]
pub struct InstructionInfo {
    /// Guest program counter (byte address, before any base-offset subtraction).
    pub pc: u64,
    /// Byte length of the instruction in the guest ISA.
    pub len: u32,
    /// Which guest ISA produced this instruction.
    pub arch: ArchTag,
    /// Coarse opcode classification flags.
    pub class: InsnClass,
}

/// What the recompiler should do after an instruction trap fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapAction {
    /// Continue with normal instruction translation.
    Continue,
    /// Suppress the instruction body entirely.
    ///
    /// The recompiler will emit the trap's
    /// [`InstructionTrap::skip_snippet`] in place of the translated
    /// instruction body.  A `skip_snippet` that emits no instructions
    /// causes a bare `unreachable` to be used.
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

    /// The extra wasm locals this trap needs per translated function.
    ///
    /// The default implementation returns [`ExtraLocals::none`] (no extra
    /// locals).  Override this to declare per-function state.
    ///
    /// This method is called once at trap-installation time by
    /// [`TrapConfig::set_instruction_trap`] to obtain the initial
    /// `ExtraLocals` declaration.  The same declaration is re-used for every
    /// subsequent function; `set_base` is called anew each time by
    /// `TrapConfig::set_extra_locals_base`.
    fn extra_locals(&self) -> ExtraLocals {
        ExtraLocals::none()
    }

    /// The extra wasm **parameters** this trap needs per translated function
    /// group.
    ///
    /// Parameters (locals 0..params-1) survive `return_call` chains; use
    /// them for state that must carry over from one translated-instruction
    /// wasm function to the next.  The default returns [`ExtraParams::none`].
    ///
    /// Called once by [`TrapConfig::setup`]; the trap's `ExtraParams` base is
    /// set there.
    fn extra_params(&self) -> ExtraParams {
        ExtraParams::none()
    }

    /// Wasm instructions to emit in place of the instruction body when this
    /// trap returns [`TrapAction::Skip`].
    ///
    /// The default emits a single `unreachable`, which is always a valid
    /// function terminator.  Traps that want to redirect control flow (e.g.
    /// jump to a violation handler) should override this.
    ///
    /// The `skip_ctx` gives full access to [`TrapContext`] so that
    /// the snippet can emit a jump, access the trap's own locals, etc.
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
/// The `skip_snippet` of the first element that returned `Skip` is used; all
/// subsequent elements in the vec that did *not* return `Skip` have already
/// had `on_instruction` called (with `Continue` result) and their
/// `skip_snippet` is not invoked.
///
/// `extra_locals` is the concatenation of all elements' locals in order.
/// The base offsets are maintained by `TrapConfig`; each element's
/// `ExtraLocals` has its own slice of the total.  Because `Vec` owns the
/// elements, `TrapConfig` stores the locals separately (one `ExtraLocals` per
/// trap slot) rather than delegating to the vec here.  This impl therefore
/// returns `ExtraLocals::none()` — the real extra-locals handling for a
/// `Vec`-based trap is done by `TrapConfig::set_instruction_trap_vec`.
impl<Context, E, F> InstructionTrap<Context, E, F>
    for Vec<Box<dyn InstructionTrap<Context, E, F> + '_>>
where
    F: InstructionSink<Context, E>,
{
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
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        (**self).on_instruction(info, ctx, trap_ctx)
    }

    fn extra_locals(&self) -> ExtraLocals {
        (**self).extra_locals()
    }

    fn extra_params(&self) -> ExtraParams {
        (**self).extra_params()
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
