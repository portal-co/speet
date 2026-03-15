//! [`Recompile`] — the trait an arch recompiler implements to participate in
//! the multi-binary linking system.
//!
//! A type implementing `Recompile` encapsulates the arch-specific translation
//! logic.  It does **not** own the reactor or traps — those live in the
//! [`Linker`](crate::linker::Linker), which is passed in as a
//! `&mut impl ReactorContext`.
//!
//! ## Lifecycle
//!
//! ```ignore
//! // 1. Create and set up the recompiler.
//! let mut rc = MyRecompiler::new();
//! rc.setup(&mut linker);   // declares arch params / calls declare_trap_params
//!
//! // 2. Translate the first binary.
//! translate(&bytes1, &mut rc, &mut linker);
//!
//! // 3. Commit.
//! linker.commit(&mut rc, entry_points_1);
//!
//! // 4. Reset for the next binary.
//! rc.reset_for_next_binary(&mut linker, args_2);
//!
//! // 5. Translate and commit the second binary.
//! translate(&bytes2, &mut rc, &mut linker);
//! linker.commit(&mut rc, entry_points_2);
//! ```

use alloc::{string::String, vec::Vec};
use crate::context::ReactorContext;
use crate::unit::BinaryUnit;

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
    fn reset_for_next_binary<RC>(
        &mut self,
        ctx: &mut RC,
        args: Self::BinaryArgs,
    )
    where
        RC: ReactorContext<Context, E, FnType = F>;

    /// Drain the reactor (via `ctx`) into a [`BinaryUnit`] and return it.
    ///
    /// Internally this calls `ctx.drain_fns()`, which advances
    /// `ctx.base_func_offset()` by the drained count so the reactor is ready
    /// for the next binary unit.
    ///
    /// The `entry_points` argument carries `(symbol, absolute_wasm_func_index)`
    /// pairs that will become WASM exports in the final module.
    fn drain_unit<RC>(
        &mut self,
        ctx: &mut RC,
        entry_points: Vec<(String, u32)>,
    ) -> BinaryUnit<F>
    where
        RC: ReactorContext<Context, E, FnType = F>;
}
