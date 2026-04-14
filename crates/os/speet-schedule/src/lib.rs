//! [`FuncSchedule`] — two-pass multi-binary linking coordinator.
//!
//! ## Two-pass model
//!
//! **Phase 1 — Registration:** call [`FuncSchedule::push`] for each binary,
//! supplying the function count (from [`Recompile::count_fns`] or
//! [`WasmFrontend::parse_fn_count`]) and an emit closure.  After all `push`
//! calls the internal [`FuncLayout`] is final; [`FuncSchedule::layout`] returns
//! absolute function-index bases for every declared slot.
//!
//! **Phase 2 — Emit:** call [`FuncSchedule::execute`].  Items are processed in
//! registration order; `base_func_offset` is set correctly for each slot before
//! the emit closure runs, and the resulting [`BinaryUnit`] is forwarded to the
//! plugin.
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
//! // Layout is final — build cross-binary IndexOffsets before emitting.
//! let offsets = IndexOffsets { func: schedule.layout().base(native_slot), .. };
//!
//! // With a Linker:
//! linker.execute_schedule(schedule, &mut ctx);
//!
//! // Or directly with any ReactorContext + LinkerPlugin:
//! let Linker { ref mut inner, ref mut plugin, .. } = linker;
//! schedule.execute(inner, plugin, &mut ctx);
//! ```

#![no_std]

extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use speet_link_core::context::ReactorContext;
use speet_link_core::layout::{FuncLayout, FuncSlot};
use speet_link_core::linker::LinkerPlugin;
use speet_link_core::unit::BinaryUnit;
use wax_core::build::InstructionSink;

type EmitFn<'a, Context, E, F> = Box<
    dyn FnOnce(&mut dyn ReactorContext<Context, E, FnType = F>, &mut Context) -> BinaryUnit<F> + 'a,
>;

struct ScheduleItem<'a, Context, E, F> {
    slot: FuncSlot,
    emit: EmitFn<'a, Context, E, F>,
}

/// Two-pass multi-binary linking coordinator.
///
/// See the [module documentation](self) for the full two-pass workflow.
pub struct FuncSchedule<'a, Context, E, F> {
    layout: FuncLayout,
    items: Vec<ScheduleItem<'a, Context, E, F>>,
}

impl<'a, Context, E, F> FuncSchedule<'a, Context, E, F> {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self { layout: FuncLayout::empty(), items: Vec::new() }
    }

    /// Register one binary unit.
    ///
    /// `count` is the number of functions this binary will produce.
    ///
    /// `emit` is a closure called during [`execute`](Self::execute) to perform
    /// the actual translation.  It receives `(&mut dyn ReactorContext, &mut
    /// Context)` and must return a [`BinaryUnit`] containing exactly `count`
    /// functions.
    ///
    /// Returns a [`FuncSlot`] whose [`base`](FuncLayout::base) is available
    /// immediately via [`layout`](Self::layout).
    pub fn push(
        &mut self,
        count: u32,
        emit: impl FnOnce(
            &mut dyn ReactorContext<Context, E, FnType = F>,
            &mut Context,
        ) -> BinaryUnit<F>
        + 'a,
    ) -> FuncSlot {
        let slot = self.layout.append(count);
        self.items.push(ScheduleItem { slot, emit: Box::new(emit) });
        slot
    }

    /// The partially- or fully-built function index layout.
    ///
    /// Available at any point during registration.  Safe to call between
    /// `push` calls to read slot bases for constructing index offsets.
    pub fn layout(&self) -> &FuncLayout {
        &self.layout
    }

    /// Execute all registered items in declaration order.
    ///
    /// For each item:
    /// 1. Sets `base_func_offset = layout.base(slot)` on `ctx`.
    /// 2. Calls the emit closure.
    /// 3. Asserts the unit contains exactly `layout.count(slot)` functions.
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
        let Self { layout, items } = self;
        for item in items {
            ctx.set_base_func_offset(layout.base(item.slot));
            let unit = (item.emit)(ctx, user_ctx);
            assert_eq!(
                unit.fns.len() as u32,
                layout.count(item.slot),
                "FuncSlot({}) declared {} fns but emit produced {}",
                item.slot.0,
                layout.count(item.slot),
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
