//! Utility trap primitives: [`NullTrap`] and [`ChainedTrap`].
//!
//! Categorised implementations live in the sibling modules:
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`tracing`](crate::tracing) | [`CounterTrap`](crate::CounterTrap), [`TraceLogTrap`](crate::TraceLogTrap) |
//! | [`security`](crate::security) | [`CfiReturnTrap`](crate::CfiReturnTrap) |
//! | [`hardening`](crate::hardening) | [`RopDetectTrap`](crate::RopDetectTrap) |

use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::LocalLayout;

use crate::context::TrapContext;
use crate::insn::{InstructionInfo, InstructionTrap, TrapAction};
use crate::jump::{JumpInfo, JumpTrap};

// ── NullTrap ──────────────────────────────────────────────────────────────────

/// A no-op trap that always returns [`TrapAction::Continue`].
///
/// Use this as a placeholder when the trait bound requires a concrete type but
/// no actual trap behaviour is needed.  When the generic parameter is
/// monomorphised to `NullTrap` the compiler will eliminate all trap overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullTrap;

impl<Context, E, F: InstructionSink<Context, E>> InstructionTrap<Context, E, F> for NullTrap {
    fn on_instruction(
        &mut self,
        _info: &InstructionInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
}

impl<Context, E, F: InstructionSink<Context, E>> JumpTrap<Context, E, F> for NullTrap {
    fn on_jump(
        &mut self,
        _info: &JumpInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
}

// ── ChainedTrap ───────────────────────────────────────────────────────────────

/// Compose two traps of the same kind: run `A` first, then `B`.
///
/// If `A` returns [`TrapAction::Skip`], `B`'s `on_instruction` / `on_jump` is
/// **not** called, and `A`'s `skip_snippet` is used.
///
/// Both traps append their parameter and local slots to the same shared
/// [`LocalLayout`] during [`declare_params`] / [`declare_locals`].  Because
/// they each receive a different [`LocalSlot`] handle, their indices will
/// never conflict regardless of insertion order.
///
/// [`declare_params`]: ChainedTrap::declare_params
/// [`declare_locals`]: ChainedTrap::declare_locals
pub struct ChainedTrap<A, B> {
    /// The first trap to run.
    pub a: A,
    /// The second trap to run (skipped if `a` returns [`TrapAction::Skip`]).
    pub b: B,
}

impl<A, B> ChainedTrap<A, B> {
    /// Construct a `ChainedTrap` that runs `a` first, then `b`.
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<Context, E, F, A, B> InstructionTrap<Context, E, F> for ChainedTrap<A, B>
where
    F: InstructionSink<Context, E>,
    A: InstructionTrap<Context, E, F>,
    B: InstructionTrap<Context, E, F>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        self.a.declare_params(params);
        self.b.declare_params(params);
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.a.declare_locals(locals);
        self.b.declare_locals(locals);
    }

    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.a.on_instruction(info, ctx, trap_ctx)? == TrapAction::Skip {
            return Ok(TrapAction::Skip);
        }
        self.b.on_instruction(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &InstructionInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        // Only A's snippet is used since B never ran (see on_instruction).
        self.a.skip_snippet(info, ctx, skip_ctx)
    }
}

impl<Context, E, F, A, B> JumpTrap<Context, E, F> for ChainedTrap<A, B>
where
    F: InstructionSink<Context, E>,
    A: JumpTrap<Context, E, F>,
    B: JumpTrap<Context, E, F>,
{
    fn declare_params(&mut self, params: &mut LocalLayout) {
        self.a.declare_params(params);
        self.b.declare_params(params);
    }

    fn declare_locals(&mut self, locals: &mut LocalLayout) {
        self.a.declare_locals(locals);
        self.b.declare_locals(locals);
    }

    fn on_jump(
        &mut self,
        info: &JumpInfo,
        ctx: &mut Context,
        trap_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<TrapAction, E> {
        if self.a.on_jump(info, ctx, trap_ctx)? == TrapAction::Skip {
            return Ok(TrapAction::Skip);
        }
        self.b.on_jump(info, ctx, trap_ctx)
    }

    fn skip_snippet(
        &self,
        info: &JumpInfo,
        ctx: &mut Context,
        skip_ctx: &mut TrapContext<Context, E, F>,
    ) -> Result<(), E> {
        self.a.skip_snippet(info, ctx, skip_ctx)
    }
}

