//! [`LinkerPlugin`] — per-unit callback trait.
//!
//! The concrete [`Linker`] implementation lives in `speet-linker`.

use crate::unit::BinaryUnit;

// ── LinkerPlugin ──────────────────────────────────────────────────────────────

/// Callback invoked once per committed [`BinaryUnit`].
///
/// Implement this to accumulate units into a final module or to perform
/// per-unit analysis.
///
/// Use `()` as a no-op plugin when you do not need to inspect individual units.
pub trait LinkerPlugin<F> {
    /// Called once per [`BinaryUnit`] produced by a schedule item.
    fn on_unit(&mut self, unit: BinaryUnit<F>);
}

/// No-op plugin — discards every unit.
impl<F> LinkerPlugin<F> for () {
    fn on_unit(&mut self, _unit: BinaryUnit<F>) {}
}
