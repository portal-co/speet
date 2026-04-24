//! [`Linker`] — owns the reactor and traps; implements [`ReactorContext`].
//!
//! The `Linker` is the central object in the multi-binary linking flow.  It:
//!
//! 1. **Owns** the [`Reactor`] and [`TrapConfig`], so recompilers do not need
//!    lifetime parameters for either.
//! 2. **Implements** [`ReactorContext`], acting as the single borrow all
//!    recompiler methods receive.
//! 3. **Forwards** completed [`BinaryUnit`]s to a pluggable
//!    [`LinkerPlugin`] (default: `()`, a no-op).
//!
//! ## Split design
//!
//! [`LinkerInner`] holds all state except the plugin and implements both
//! [`BaseContext`] and [`ReactorContext`].  [`Linker`] wraps `LinkerInner`
//! together with `plugin: Plugin`.  The split enables
//! [`Linker::execute_schedule`] to borrow `inner` and `plugin` disjointly
//! without unsafe code.

#![no_std]

extern crate alloc;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;
use speet_traps::{InstructionInfo, JumpInfo, TrapAction, TrapConfig};
use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;
use yecta::layout::CellIdx;
use yecta::{
    EscapeTag, Fed, FuncIdx, LocalLayout, LocalPool, LocalPoolBackend, Mark, Pool, Reactor,
    layout::CellRegistry,
};

pub use speet_link_core::linker::LinkerPlugin;
pub use speet_link_core::{BaseContext, BinaryUnit, DataSegment, FuncLayout, FuncSlot, FuncType,
    MemWidth, ParamSource, Place, ReactorAdapter, ReactorContext, Recompile, SavePair, ShimSpec,
    emit_shim};

// ── LinkerInner ───────────────────────────────────────────────────────────────

/// All [`Linker`] state except the plugin.
///
/// Implements [`BaseContext`] and [`ReactorContext`] directly; [`Linker`]
/// delegates to it.  The separation allows [`Linker::execute_schedule`] to
/// split borrows of `inner` and `plugin` without unsafe code.
pub struct LinkerInner<'cb, 'ctx, Context, E, F, P = LocalPool>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// The reactor that generates WASM functions.
    pub reactor: Reactor<Context, E, F, P>,
    /// Pluggable instruction-level and jump-level trap hooks.
    pub traps: TrapConfig<'cb, 'ctx, Context, E>,
    /// Unified layout: arch params + trap params, then per-function locals.
    pub layout: LocalLayout,
    /// Mark placed after all parameter slots (captures `total_params`).
    pub locals_mark: Mark,
    /// Indirect-call pool handler and type.
    pub pool: Pool<'cb, Context, E>,
    /// Optional escape tag for exception-based control flow.
    pub escape_tag: Option<EscapeTag>,
    /// Registry mapping unique (function-type params, locals) combinations
    /// to [`CellIdx`] handles.  Populated lazily by [`BaseContext::alloc_cell`].
    pub cell_registry: CellRegistry,
}

// ── BaseContext for LinkerInner ───────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P> BaseContext<Context, E>
    for LinkerInner<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn layout(&self) -> &LocalLayout {
        &self.layout
    }
    fn layout_mut(&mut self) -> &mut LocalLayout {
        &mut self.layout
    }
    fn locals_mark(&self) -> Mark {
        self.locals_mark
    }
    fn set_locals_mark(&mut self, mark: Mark) {
        self.locals_mark = mark;
    }

    fn base_func_offset(&self) -> u32 {
        self.reactor.base_func_offset()
    }
    fn set_base_func_offset(&mut self, n: u32) {
        self.reactor.set_base_func_offset(n);
    }

    fn declare_trap_params(&mut self) {
        self.traps.declare_params(CellIdx(0), &mut self.layout);
    }
    fn declare_trap_locals(&mut self) {
        self.traps.declare_locals(CellIdx(0), &mut self.layout);
    }
    fn alloc_cell(&mut self) -> CellIdx {
        let mark = self.locals_mark;
        self.cell_registry.register(
            self.layout.iter_before(&mark),
            self.layout.iter_since(&mark),
        )
    }
    fn alloc_cell_for_guest(
        &mut self,
        guest_params: &[ValType],
        guest_results: &[ValType],
        guest_locals: &[(u32, ValType)],
    ) -> CellIdx {
        self.cell_registry.register_for_guest(
            guest_params.iter().copied(),
            guest_results.iter().copied(),
            guest_locals.iter().copied(),
        )
    }
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        self.traps.declare_locals(cell, &mut self.layout);
    }
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
    ) -> Result<TrapAction, E> {
        let layout = &self.layout as *const LocalLayout;
        // SAFETY: layout is borrowed immutably; self.traps and self.reactor
        // are disjoint fields.  We use a raw pointer to work around the
        // borrow-checker limitation on partial borrows through &mut self.
        // This is sound because `on_instruction` does not modify `layout`.
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.traps
            .on_instruction(info, ctx, &mut self.reactor, layout_ref)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        let layout = &self.layout as *const LocalLayout;
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.traps
            .on_jump(info, ctx, &mut self.reactor, layout_ref)
    }
}

// ── InstructionSink for LinkerInner ──────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P> InstructionSink<Context, E>
    for LinkerInner<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.tail().instruction(ctx, insn)
    }
}

// ── ReactorContext for LinkerInner ────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P> ReactorContext<Context, E>
    for LinkerInner<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + yecta::LocalPoolApi,
{
    type FnType = F;

    fn fn_count(&self) -> usize {
        self.reactor.fn_count()
    }
    fn drain_fns(&mut self) -> Vec<F> {
        self.reactor.drain_fns()
    }

    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<usize, E> {
        self.reactor.next_with(ctx, f, len)?;
        Ok(self.reactor.fn_count() - 1)
    }
    fn feed(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, insn)
    }
    fn jmp(&self, ctx: &mut Context, tail_idx: usize, target: FuncIdx, params: u32) -> Result<(), E> {
        self.reactor.jmp(tail_idx, ctx, target, params)
    }
    fn seal_fn(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.seal_to(tail_idx, ctx, insn)
    }

    fn pool(&self) -> Pool<'_, Context, E> {
        self.pool
    }
    fn escape_tag(&self) -> Option<EscapeTag> {
        self.escape_tag
    }
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>) {
        self.escape_tag = tag;
    }

    fn ji(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
        params: u32,
        fixups: &alloc::collections::BTreeMap<u32, &dyn yecta::Snippet<Context, E>>,
        target: yecta::Target<Context, E>,
        call: Option<EscapeTag>,
        pool: Pool<'_, Context, E>,
        condition: Option<&dyn yecta::Snippet<Context, E>>,
    ) -> Result<(), E> {
        self.reactor.ji(ctx, params, fixups, target, call, pool, condition, tail_idx)
    }

    fn ji_with_params(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
        params: yecta::JumpCallParams<'_, Context, E>,
    ) -> Result<(), E> {
        self.reactor.ji_with_params(ctx, params, tail_idx)
    }

    fn ret(&self, ctx: &mut Context, tail_idx: usize, params: u32, tag: EscapeTag) -> Result<(), E> {
        self.reactor.ret(tail_idx, ctx, params, tag)
    }

    fn with_local_pool(&self, f: &mut dyn FnMut(&mut dyn yecta::LocalPoolApi)) {
        self.reactor.with_local_pool(|p| f(p as &mut dyn yecta::LocalPoolApi))
    }

    fn flush_bundles(&self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        self.reactor.flush_bundles(ctx, tail_idx)
    }

    fn flush_for_load(
        &self,
        ctx: &mut Context,
        addr_local: u32,
        addr_type: ValType,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.reactor.flush_bundles_for_load(ctx, addr_local, addr_type, tail_idx)
    }

    fn feed_lazy(
        &self,
        ctx: &mut Context,
        addr_type: ValType,
        val_type: ValType,
        insn: &Instruction<'static>,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.reactor.feed_lazy(ctx, addr_type, val_type, insn, tail_idx)
    }
}

// ── Linker ────────────────────────────────────────────────────────────────────

/// Owns the reactor and traps; implements [`ReactorContext`] so recompilers
/// can borrow it as their single environment parameter.
///
/// ## Type parameters
/// - `'cb` — lifetime of the borrowed trap implementations.
/// - `'ctx` — lifetime of data captured inside the trap callbacks.
/// - `Context` — the translation context forwarded to the reactor.
/// - `E` — the error type.
/// - `F` — the WASM function sink type (e.g. `wasm_encoder::Function`).
/// - `P` — local-pool backend (defaults to [`LocalPool`]).
/// - `Plugin` — the [`LinkerPlugin`] (defaults to `()`).
pub struct Linker<'cb, 'ctx, Context, E, F, P = LocalPool, Plugin = ()>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
{
    /// All state except the plugin — implements `BaseContext` + `ReactorContext`.
    pub inner: LinkerInner<'cb, 'ctx, Context, E, F, P>,
    /// Downstream handler for completed [`BinaryUnit`]s.
    pub plugin: Plugin,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P> Linker<'cb, 'ctx, Context, E, F, P, ()>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + Default,
{
    /// Create a `Linker` with the default no-op plugin.
    pub fn new() -> Self {
        Self::with_plugin(())
    }
}

impl<'cb, 'ctx, Context, E, F, P> Default for Linker<'cb, 'ctx, Context, E, F, P, ()>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'cb, 'ctx, Context, E, F, P, Plugin> Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + Default,
    Plugin: LinkerPlugin<F>,
{
    /// Create a `Linker` with the given plugin.
    pub fn with_plugin(plugin: Plugin) -> Self {
        use yecta::{TableIdx, TypeIdx};
        Self {
            inner: LinkerInner {
                reactor: Reactor::default(),
                traps: TrapConfig::new(),
                layout: LocalLayout::empty(),
                locals_mark: Mark { slot_count: 0, total_locals: 0 },
                pool: {
                    static T: TableIdx = TableIdx(0);
                    Pool { handler: &T, ty: TypeIdx(0) }
                },
                escape_tag: None,
                cell_registry: CellRegistry::new(),
            },
            plugin,
        }
    }
}

// ── InstructionSink for Linker ──────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> InstructionSink<Context, E>
    for Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.inner.reactor.tail().instruction(ctx, insn)
    }
}

// ── BaseContext for Linker (delegates to inner) ───────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> BaseContext<Context, E>
    for Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
{
    fn layout(&self) -> &LocalLayout {
        self.inner.layout()
    }
    fn layout_mut(&mut self) -> &mut LocalLayout {
        self.inner.layout_mut()
    }
    fn locals_mark(&self) -> Mark {
        self.inner.locals_mark()
    }
    fn set_locals_mark(&mut self, mark: Mark) {
        self.inner.set_locals_mark(mark);
    }
    fn base_func_offset(&self) -> u32 {
        self.inner.base_func_offset()
    }
    fn set_base_func_offset(&mut self, n: u32) {
        self.inner.set_base_func_offset(n);
    }
    fn declare_trap_params(&mut self) {
        self.inner.declare_trap_params();
    }
    fn declare_trap_locals(&mut self) {
        self.inner.declare_trap_locals();
    }
    fn alloc_cell(&mut self) -> CellIdx {
        self.inner.alloc_cell()
    }
    fn alloc_cell_for_guest(
        &mut self,
        guest_params: &[ValType],
        guest_results: &[ValType],
        guest_locals: &[(u32, ValType)],
    ) -> CellIdx {
        self.inner.alloc_cell_for_guest(guest_params, guest_results, guest_locals)
    }
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        self.inner.declare_trap_locals_with_cell(cell);
    }
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
    ) -> Result<TrapAction, E> {
        self.inner.on_instruction(info, ctx)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        self.inner.on_jump(info, ctx)
    }
}

// ── ReactorContext for Linker (delegates to inner) ────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> ReactorContext<Context, E>
    for Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + yecta::LocalPoolApi,
    Plugin: LinkerPlugin<F>,
{
    type FnType = F;

    fn fn_count(&self) -> usize {
        self.inner.fn_count()
    }
    fn drain_fns(&mut self) -> Vec<F> {
        self.inner.drain_fns()
    }
    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<usize, E> {
        self.inner.next_with(ctx, f, len)
    }
    fn feed(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        self.inner.feed(ctx, tail_idx, insn)
    }
    fn jmp(&self, ctx: &mut Context, tail_idx: usize, target: FuncIdx, params: u32) -> Result<(), E> {
        self.inner.jmp(ctx, tail_idx, target, params)
    }
    fn seal_fn(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        self.inner.seal_fn(ctx, tail_idx, insn)
    }
    fn pool(&self) -> Pool<'_, Context, E> {
        self.inner.pool()
    }
    fn escape_tag(&self) -> Option<EscapeTag> {
        self.inner.escape_tag()
    }
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>) {
        self.inner.set_escape_tag(tag);
    }

    fn ji(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
        params: u32,
        fixups: &alloc::collections::BTreeMap<u32, &dyn yecta::Snippet<Context, E>>,
        target: yecta::Target<Context, E>,
        call: Option<EscapeTag>,
        pool: Pool<'_, Context, E>,
        condition: Option<&dyn yecta::Snippet<Context, E>>,
    ) -> Result<(), E> {
        self.inner.ji(ctx, tail_idx, params, fixups, target, call, pool, condition)
    }

    fn ji_with_params(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
        params: yecta::JumpCallParams<'_, Context, E>,
    ) -> Result<(), E> {
        self.inner.ji_with_params(ctx, tail_idx, params)
    }

    fn ret(&self, ctx: &mut Context, tail_idx: usize, params: u32, tag: EscapeTag) -> Result<(), E> {
        self.inner.ret(ctx, tail_idx, params, tag)
    }

    fn with_local_pool(&self, f: &mut dyn FnMut(&mut dyn yecta::LocalPoolApi)) {
        self.inner.with_local_pool(f)
    }

    fn flush_bundles(&self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        self.inner.flush_bundles(ctx, tail_idx)
    }

    fn flush_for_load(
        &self,
        ctx: &mut Context,
        addr_local: u32,
        addr_type: ValType,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.inner.flush_for_load(ctx, addr_local, addr_type, tail_idx)
    }

    fn feed_lazy(
        &self,
        ctx: &mut Context,
        addr_type: ValType,
        val_type: ValType,
        insn: &Instruction<'static>,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.inner.feed_lazy(ctx, addr_type, val_type, insn, tail_idx)
    }
}

// ── execute_schedule convenience ──────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
{
    /// Execute a [`speet_schedule::FuncSchedule`] using this linker.
    ///
    /// Splits the borrow into `inner` (as `ReactorContext`) and `plugin` (as
    /// `LinkerPlugin`) so `FuncSchedule::execute` can receive them separately.
    pub fn execute_schedule<'a>(
        &mut self,
        schedule: speet_schedule::FuncSchedule<'a, Context, E, F>,
        ctx: &mut Context,
    ) {
        let Linker { ref mut inner, ref mut plugin, .. } = *self;
        schedule.execute(inner, plugin, ctx);
    }
}
