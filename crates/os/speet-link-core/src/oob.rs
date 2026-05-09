//! [`OobConfig`] — out-of-bounds jump dispatch configuration.
//!
//! When a recompiler encounters a static jump target outside the compiled set,
//! it calls [`ReactorContext::oob_jump`] instead of emitting `unreachable`.
//! `OobConfig` provides the runtime indices needed for that dispatch.
//!
//! ## Phase 1 — registration
//!
//! ```ignore
//! let oob = OobConfig::register(&mut linker.inner.entity_space);
//! linker.inner.oob_config = Some(oob);
//! ```
//!
//! ## Phase 2 — fill function indices
//!
//! After the `speet-interp` binary unit has been scheduled and its absolute
//! function indices are known, call:
//! ```ignore
//! if let Some(ref mut oob) = linker.inner.oob_config {
//!     oob.set_func_indices(lookup_stub_abs_idx, interp_abs_idx);
//! }
//! ```

use crate::layout::{EntityIndexSpace, IndexSlot};

/// Out-of-bounds jump dispatch configuration.
///
/// All compiled functions, the lookup stub, and the interpreter share the same
/// WASM function type (arch regs + `target_pc: i64` → arch regs), so they
/// can be tail-called interchangeably.
#[derive(Clone, Debug)]
pub struct OobConfig {
    /// Absolute WASM function index of the lookup stub.
    ///
    /// The lookup stub binary-searches the compiled-PC table and tail-calls
    /// the matching compiled function, or falls through to the interpreter.
    pub lookup_stub_func_idx: u32,

    /// Absolute WASM function index of the soft interpreter.
    ///
    /// Executes one guest instruction per iteration; dispatches back to
    /// compiled code via the function table when it reaches a compiled PC.
    pub interp_func_idx: u32,

    /// Pre-declared WASM table slot (registered in Phase 1).
    ///
    /// The element section populates this table with compiled function indices
    /// keyed by their PC offset, enabling `return_call_indirect` dispatch.
    pub dispatch_table_slot: IndexSlot,
}

impl OobConfig {
    /// Register one WASM table in Phase 1.
    ///
    /// Returns a partially-initialised `OobConfig`; call
    /// [`set_func_indices`](Self::set_func_indices) in Phase 2.
    pub fn register(entity_space: &mut EntityIndexSpace) -> Self {
        let dispatch_table_slot = entity_space.tables.append(1);
        Self {
            lookup_stub_func_idx: 0,
            interp_func_idx: 0,
            dispatch_table_slot,
        }
    }

    /// Set the absolute WASM function indices for the lookup stub and
    /// interpreter once their binary-unit slot bases are known.
    pub fn set_func_indices(&mut self, lookup_stub: u32, interp: u32) {
        self.lookup_stub_func_idx = lookup_stub;
        self.interp_func_idx = interp;
    }

    /// Absolute WASM table index of the dispatch table.
    pub fn table_idx(&self, entity_space: &EntityIndexSpace) -> u32 {
        entity_space.tables.base(self.dispatch_table_slot)
    }
}
