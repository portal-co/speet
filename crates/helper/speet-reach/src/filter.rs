//! [`ReachabilityFilter`] — an [`InstructionTrap`] that skips unreachable
//! instructions.

use speet_traps::{InstructionInfo, InstructionTrap, LocalDeclarator, LocalLayout, TrapAction};
use yecta::layout::CellIdx;

use crate::ReachableSet;

// ── ReachabilityFilter ────────────────────────────────────────────────────────

/// An [`InstructionTrap`] that skips instructions whose PC is not in a
/// [`ReachableSet`].
///
/// # Usage
///
/// ```ignore
/// // 1. Build the reachable set from external-tool seeds.
/// let spec = ReachabilitySpec::single(entry_pc);
/// let reachable = compute_reachable(&spec, &binary_bytes, base_addr, &decoder);
///
/// // 2. Wrap it in a filter and install it.
/// let filter = ReachabilityFilter::new(reachable);
/// recompiler.set_instruction_trap(filter);
///
/// // 3. Translate — unreachable instructions emit `unreachable` stubs.
/// recompiler.translate_bytes(&mut ctx, &binary_bytes, base_addr, &mut sink_factory)?;
/// ```
///
/// # Function count
///
/// Skipped instructions still produce a WASM function (containing a single
/// `unreachable` instruction), so the total function count is unchanged and
/// the linker's two-pass [`FuncSchedule`] remains valid.
///
/// [`FuncSchedule`]: speet_link::FuncSchedule
pub struct ReachabilityFilter {
    reachable: ReachableSet,
}

impl ReachabilityFilter {
    /// Create a new filter from a precomputed [`ReachableSet`].
    pub fn new(reachable: ReachableSet) -> Self {
        Self { reachable }
    }

    /// Consume the filter and return the inner [`ReachableSet`].
    pub fn into_inner(self) -> ReachableSet {
        self.reachable
    }

    /// Borrow the inner [`ReachableSet`].
    pub fn reachable(&self) -> &ReachableSet {
        &self.reachable
    }
}

// ── LocalDeclarator (no-op) ───────────────────────────────────────────────────

impl LocalDeclarator for ReachabilityFilter {
    fn declare_params(&mut self, _cell: CellIdx, _params: &mut LocalLayout) {
        // No WASM parameters needed.
    }

    fn declare_locals(&mut self, _cell: CellIdx, _locals: &mut LocalLayout) {
        // No WASM locals needed.
    }
}

// ── InstructionTrap ───────────────────────────────────────────────────────────

impl<Context, E> InstructionTrap<Context, E> for ReachabilityFilter {
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        _ctx: &mut Context,
        _trap_ctx: &mut speet_traps::TrapContext<Context, E>,
    ) -> Result<TrapAction, E> {
        if self.reachable.contains(info.pc) {
            Ok(TrapAction::Continue)
        } else {
            Ok(TrapAction::Skip)
        }
    }
    // `skip_snippet` is inherited: emits a single `unreachable` instruction.
}
