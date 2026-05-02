//! [`FuncSchedule`] — two-pass multi-binary linking coordinator.
//!
//! ## Two-pass model
//!
//! **Phase 1 — Registration:** call [`FuncSchedule::push`] for each binary,
//! supplying the function count and an emit closure.  After all `push` calls
//! the internal [`EntityIndexSpace`] is frozen; absolute index bases for every
//! entity kind are available via [`FuncSchedule::entity_space`].
//!
//! **Phase 2 — Emit:** call [`FuncSchedule::execute`].  Items are processed in
//! registration order; `base_func_offset` is set correctly for each function
//! slot before the emit closure runs, and the resulting [`BinaryUnit`] is
//! forwarded to the plugin.
//!
//! ## Example
//!
//! ```ignore
//! let mut schedule: FuncSchedule<(), Infallible, Function> = FuncSchedule::new();
//!
//! let n_wasm = WasmFrontend::parse_fn_count(&wasm_bytes)?;
//! let n_native = rc.count_fns(&native_bytes);
//!
//! let wasm_slot   = schedule.push(n_wasm,   |ctx_rc, _ctx| { /* translate */ });
//! let native_slot = schedule.push(n_native, |ctx_rc, _ctx| { /* translate */ });
//!
//! // Layout is final — read cross-binary indices from entity_space:
//! let offsets = IndexOffsets { func: schedule.entity_space().functions.base(native_slot), .. };
//!
//! linker.execute_schedule(schedule, &mut ctx);
//! ```

#![no_std]

extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use speet_link_core::context::ReactorContext;
use speet_link_core::layout::{EntityIndexSpace, IndexSlot};
use speet_link_core::linker::LinkerPlugin;
use speet_link_core::unit::BinaryUnit;
use wax_core::build::InstructionSink;

// Keep legacy alias for call sites that still use FuncSlot / FuncLayout.
pub use speet_link_core::layout::{FuncLayout, FuncSlot};

type EmitFn<'a, Context, E, F> = Box<
    dyn FnOnce(&mut dyn ReactorContext<Context, E, FnType = F>, &mut Context) -> BinaryUnit<F> + 'a,
>;

struct ScheduleItem<'a, Context, E, F> {
    fn_slot: IndexSlot,
    emit: EmitFn<'a, Context, E, F>,
}

/// Two-pass multi-binary linking coordinator.
///
/// See the [module documentation](self) for the full two-pass workflow.
pub struct FuncSchedule<'a, Context, E, F> {
    entity_space: EntityIndexSpace,
    items: Vec<ScheduleItem<'a, Context, E, F>>,
}

impl<'a, Context, E, F> FuncSchedule<'a, Context, E, F> {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self { entity_space: EntityIndexSpace::empty(), items: Vec::new() }
    }

    /// Register one binary unit (function-count only).
    ///
    /// `count` is the number of functions this binary will produce.
    ///
    /// `emit` is a closure called during [`execute`](Self::execute) to perform
    /// the actual translation.  It receives `(&mut dyn ReactorContext, &mut
    /// Context)` and must return a [`BinaryUnit`] containing exactly `count`
    /// functions.
    ///
    /// Returns an [`IndexSlot`] into `entity_space().functions` whose base is
    /// available immediately.
    pub fn push(
        &mut self,
        count: u32,
        emit: impl FnOnce(
            &mut dyn ReactorContext<Context, E, FnType = F>,
            &mut Context,
        ) -> BinaryUnit<F>
        + 'a,
    ) -> IndexSlot {
        let slot = self.entity_space.functions.append(count);
        self.items.push(ScheduleItem { fn_slot: slot, emit: Box::new(emit) });
        slot
    }

    /// Read-only access to the unified pre-declaration index space.
    ///
    /// All five entity kinds (types, functions, memories, tables, tags) may be
    /// pre-declared here during Phase 1.  The space is frozen before
    /// [`execute`](Self::execute) is called; emit closures capture any bases
    /// they need from it at registration time.
    pub fn entity_space(&self) -> &EntityIndexSpace {
        &self.entity_space
    }

    /// Mutable access to the unified pre-declaration index space.
    ///
    /// Use this to pre-declare non-function entities (memories, tables, tags,
    /// types) during Phase 1, before [`execute`](Self::execute) is called.
    pub fn entity_space_mut(&mut self) -> &mut EntityIndexSpace {
        &mut self.entity_space
    }

    /// The function sub-space (convenience accessor).
    ///
    /// Equivalent to `entity_space().functions`.  Provided for backward
    /// compatibility with code that called `schedule.layout()`.
    pub fn layout(&self) -> &speet_link_core::layout::IndexSpace {
        &self.entity_space.functions
    }

    /// Execute all registered items in declaration order.
    ///
    /// For each item:
    /// 1. Sets `base_func_offset = entity_space.functions.base(fn_slot)` on `ctx`.
    /// 2. Calls the emit closure.
    /// 3. Asserts the unit contains exactly the declared function count.
    /// 4. Forwards the unit to `plugin`.
    ///
    /// # Panics
    ///
    /// Panics if any emit closure produces a function count that differs from
    /// its declared count.
    #[track_caller]
    pub fn execute<C, P>(
        self,
        ctx: &mut C,
        plugin: &mut P,
        user_ctx: &mut Context,
    ) where
        C: ReactorContext<Context, E, FnType = F>,
        P: LinkerPlugin<F>,
        F: InstructionSink<Context, E>,
    {
        let Self { entity_space, items } = self;
        for item in items {
            ctx.set_base_func_offset(entity_space.functions.base(item.fn_slot));
            let unit = (item.emit)(ctx, user_ctx);
            assert_eq!(
                unit.fns.len() as u32,
                entity_space.functions.count(item.fn_slot),
                "IndexSlot({}) declared {} fns but emit produced {}",
                item.fn_slot.0,
                entity_space.functions.count(item.fn_slot),
                unit.fns.len(),
            );
            plugin.on_unit(unit);
        }
    }
}

#[cfg(test)]
mod tests;

impl<'a, Context, E, F> Default for FuncSchedule<'a, Context, E, F> {
    fn default() -> Self {
        Self::new()
    }
}
