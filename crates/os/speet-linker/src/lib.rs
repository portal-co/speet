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
use wasm_encoder::Instruction;
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
    Reactor<Context, E, F, P>: InstructionSink<Context, E>,
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

// ── ReactorContext for LinkerInner ────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P> ReactorContext<Context, E>
    for LinkerInner<'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Reactor<Context, E, F, P>: InstructionSink<Context, E>,
{
    type FnType = F;

    fn fn_count(&self) -> usize {
        self.reactor.fn_count()
    }
    fn drain_fns(&mut self) -> Vec<F> {
        self.reactor.drain_fns()
    }

    fn feed(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self.reactor.fn_count().saturating_sub(1);
        Fed { reactor: &self.reactor, tail_idx }.instruction(ctx, insn)
    }
    fn jmp(&self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        let tail_idx = self.reactor.fn_count().saturating_sub(1);
        self.reactor.jmp(tail_idx, ctx, target, params)
    }
    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<(), E> {
        self.reactor.next_with(ctx, f, len)
    }
    fn seal_fn(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self.reactor.fn_count().saturating_sub(1);
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

// ── BaseContext for Linker (delegates to inner) ───────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> BaseContext<Context, E>
    for Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
    Reactor<Context, E, F, P>: InstructionSink<Context, E>,
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
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
    Reactor<Context, E, F, P>: InstructionSink<Context, E>,
{
    type FnType = F;

    fn fn_count(&self) -> usize {
        self.inner.fn_count()
    }
    fn drain_fns(&mut self) -> Vec<F> {
        self.inner.drain_fns()
    }
    fn feed(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.inner.feed(ctx, insn)
    }
    fn jmp(&self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.inner.jmp(ctx, target, params)
    }
    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<(), E> {
        self.inner.next_with(ctx, f, len)
    }
    fn seal_fn(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.inner.seal_fn(ctx, insn)
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
}

// ── execute_schedule convenience ──────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
    Reactor<Context, E, F, P>: InstructionSink<Context, E>,
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
