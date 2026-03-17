//! [`Recompile`] — the trait an arch recompiler implements to participate in
//! the multi-binary linking system.
//!
//! A type implementing `Recompile` encapsulates the arch-specific translation
//! logic.  It does **not** own the reactor or traps — those live in the
//! [`Linker`](crate::linker::Linker), which is passed in as a
//! `&mut impl ReactorContext`.
//!
//! ## Lifecycle (two-pass model)
//!
//! ```ignore
//! // --- Registration phase ---
//! let mut schedule = FuncSchedule::new();
//!
//! // 1. Count functions for each binary (lightweight pre-pass).
//! let n1 = rc.count_fns(&bytes1);
//! let n2 = rc.count_fns(&bytes2);
//!
//! // 2. Register emit closures; layout is final after all pushes.
//! let slot1 = schedule.push(n1, |ctx_rc, ctx| {
//!     rc.reset_for_next_binary(ctx_rc, args1);
//!     translate(&bytes1, &mut rc, ctx_rc);
//!     rc.drain_unit(ctx_rc, entry_points_1)
//! });
//! let slot2 = schedule.push(n2, |ctx_rc, ctx| {
//!     rc.reset_for_next_binary(ctx_rc, args2);
//!     translate(&bytes2, &mut rc, ctx_rc);
//!     rc.drain_unit(ctx_rc, entry_points_2)
//! });
//!
//! // 3. Read cross-binary offsets from the finalised layout.
//! let offsets = IndexOffsets { func: schedule.layout().base(slot2), .. };
//!
//! // --- Emit phase ---
//! schedule.execute(&mut linker, &mut ctx);
//! ```

use crate::context::ReactorContext;
use crate::unit::BinaryUnit;
use alloc::{string::String, vec::Vec};

// ── Recompile ─────────────────────────────────────────────────────────────────

/// Interface for an arch recompiler that participates in the multi-binary
/// linker.
///
/// ## Type parameters
/// - `Context` — the translation context forwarded to the reactor.
/// - `E` — the error type.
/// - `F` — the WASM function type produced (matches the reactor's `F`).
///   Must equal `RC::FnType` for any `ReactorContext` passed to this trait's
///   methods.
pub trait Recompile<Context, E, F> {
    /// Per-binary arguments provided when resetting for a new binary.
    ///
    /// For example, x86-64 uses `u64` (new `base_rip`); RISC-V might use a
    /// small struct carrying `base_pc` and optional new callbacks.
    type BinaryArgs;

    /// Reset per-binary state (addresses, hint tables, parsed input, …)
    /// without touching the caller's reactor or traps.
    ///
    /// This is called between two consecutive binary translations to prepare
    /// the recompiler for the next binary unit.  The `ctx` argument is
    /// provided so the recompiler can, if needed, call methods such as
    /// `ctx.layout_mut()` to rebuild the local layout for the new binary.
    fn reset_for_next_binary(
        &mut self,
        ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        args: Self::BinaryArgs,
    );

    /// Count the WASM functions that would be produced for `bytes`, without
    /// performing code generation.
    ///
    /// This is a lightweight pre-pass (e.g. CFG block count for native arches,
    /// or a WASM Function-section scan).  The result is passed to
    /// [`FuncSchedule::push`](crate::schedule::FuncSchedule::push) to
    /// pre-declare the slot before any translation begins.
    ///
    /// The default implementation panics — native-arch recompilers must
    /// override this with an actual CFG block count.
    fn count_fns(&self, _bytes: &[u8]) -> u32 {
        panic!(
            "count_fns not implemented for {}; override it to use FuncSchedule",
            core::any::type_name::<Self>(),
        )
    }

    /// Drain the reactor (via `ctx`) into a [`BinaryUnit`] and return it.
    ///
    /// Internally this calls `ctx.drain_fns()`, which advances
    /// `ctx.base_func_offset()` by the drained count so the reactor is ready
    /// for the next binary unit.
    ///
    /// The `entry_points` argument carries `(symbol, absolute_wasm_func_index)`
    /// pairs that will become WASM exports in the final module.
    fn drain_unit(
        &mut self,
        ctx: &mut (dyn ReactorContext<Context, E, FnType = F> + '_),
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F>;
}
