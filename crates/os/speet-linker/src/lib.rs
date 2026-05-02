//! [`Linker`] — shared translation state and multi-binary linking.
//!
//! ## Architecture after the base-reactor split
//!
//! [`LinkerInner`] holds all persistent state (traps, layout, cell registry,
//! entity index space, function signature) but **does not own a reactor**.
//! It implements [`BaseContext`] only.
//!
//! [`ReactorHandle`] wraps `&mut LinkerInner` together with a `&mut Reactor`
//! and implements both [`BaseContext`] and [`ReactorContext`].  Native
//! recompiler emit closures create a [`Reactor`] on the stack and wrap it
//! with a [`ReactorHandle`] for the duration of a single translation unit.
//!
//! [`Linker`] owns a [`LinkerInner`] and a [`LinkerPlugin`].
//! [`Linker::execute_schedule`] creates one [`Reactor`] per
//! [`FuncSchedule`](speet_schedule::FuncSchedule) execution and wraps it in
//! a [`ReactorHandle`] that is passed to every emit closure in the schedule.
//!
//! See `docs/reactor-context-split.md` for the full design rationale.

#![no_std]

extern crate alloc;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;
use speet_traps::{InstructionInfo, JumpInfo, TrapAction, TrapConfig};
use wasm_encoder::{Instruction, ValType};
use yecta::FuncSignature;
use wax_core::build::InstructionSink;
use yecta::layout::CellIdx;
use yecta::{
    EscapeTag, Fed, FuncIdx, LocalLayout, LocalPool, LocalPoolBackend, Mark, Pool, Reactor,
    layout::CellRegistry,
};

pub use speet_link_core::linker::LinkerPlugin;
pub use speet_link_core::{
    BaseContext, BinaryUnit, DataSegment, EntityIndexSpace, FuncLayout, FuncSlot, FuncType,
    IndexSlot, IndexSpace, MemWidth, ParamSource, Place, ReactorAdapter, ReactorContext, Recompile,
    SavePair, ShimSpec, TrapReactorAdapter, emit_shim,
};

// ── LinkerInner ───────────────────────────────────────────────────────────────

/// Persistent translation-unit state — traps, layout, cell registry, entity
/// index space, and function signature.
///
/// Does **not** own a reactor.  Implements [`BaseContext`] only.
/// For full [`ReactorContext`] (instruction emission) use [`ReactorHandle`],
/// which borrows both a `LinkerInner` and a [`Reactor`].
///
/// See `docs/reactor-context-split.md`.
pub struct LinkerInner<'cb, 'ctx, Context, E> {
    /// Pluggable instruction-level and jump-level trap hooks.
    pub traps: TrapConfig<'cb, 'ctx, Context, E>,
    /// Unified layout: arch params + injected/trap params, then per-function locals.
    pub layout: LocalLayout,
    /// Mark placed after all parameter slots (set by `set_locals_mark`).
    pub locals_mark: Mark,
    /// Mark placed before injected/trap params (set by `declare_trap_params`).
    pub injected_start: Mark,
    /// Canonical function signature sealed after `set_locals_mark`.
    pub signature: FuncSignature,
    /// Absolute WASM function index of the first function in the current slot.
    pub base_func_offset: u32,
    /// Indirect-call pool handler and type index.
    pub pool: Pool<'cb, Context, E>,
    /// Optional escape tag for exception-based control flow.
    pub escape_tag: Option<EscapeTag>,
    /// Registry mapping unique `(params, locals)` signatures to [`CellIdx`] handles.
    pub cell_registry: CellRegistry,
    /// Pre-declared index space for all five WASM entity kinds.
    ///
    /// See `docs/entity-index-space.md`.
    pub entity_space: EntityIndexSpace,
}

// ── BaseContext for LinkerInner ───────────────────────────────────────────────

impl<'cb, 'ctx, Context, E> BaseContext<Context, E> for LinkerInner<'cb, 'ctx, Context, E> {
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
        // Seal the FuncSignature from the current params portion.
        let mut params_layout = self.layout.clone();
        params_layout.rewind(&mark);
        self.signature = FuncSignature::seal(params_layout, self.injected_start);
    }

    fn base_func_offset(&self) -> u32 {
        self.base_func_offset
    }
    fn set_base_func_offset(&mut self, n: u32) {
        self.base_func_offset = n;
    }

    fn declare_trap_params(&mut self) {
        self.injected_start = self.layout.mark();
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
    // on_instruction / on_jump: LinkerInner has no reactor, so traps cannot
    // emit instructions.  Real firing happens through ReactorHandle.
    fn on_instruction(
        &mut self,
        _info: &InstructionInfo,
        _ctx: &mut Context,
    ) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
    fn on_jump(&mut self, _info: &JumpInfo, _ctx: &mut Context) -> Result<TrapAction, E> {
        Ok(TrapAction::Continue)
    }
}

// ── ReactorHandle ─────────────────────────────────────────────────────────────

/// A [`ReactorContext`] that borrows a [`LinkerInner`] and a [`Reactor`].
///
/// Created per translation unit (usually inside an emit closure):
///
/// ```ignore
/// let mut reactor = Reactor::<Context, E, F, LocalPool>::default();
/// reactor.set_base_func_offset(inner.base_func_offset);
/// let mut handle = ReactorHandle::new(&mut inner, &mut reactor);
/// recompiler.translate(&mut handle, bytes, ctx)?;
/// ```
///
/// See `docs/reactor-context-split.md`.
pub struct ReactorHandle<'a, 'cb, 'ctx, Context, E, F, P = LocalPool>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// Persistent translation-unit state (traps, layout, cell registry, …).
    pub base: &'a mut LinkerInner<'cb, 'ctx, Context, E>,
    /// The per-translation-unit reactor.
    pub reactor: &'a mut Reactor<Context, E, F, P>,
}

impl<'a, 'cb, 'ctx, Context, E, F, P> ReactorHandle<'a, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// Create a new handle from a `LinkerInner` borrow and a reactor borrow.
    pub fn new(
        base: &'a mut LinkerInner<'cb, 'ctx, Context, E>,
        reactor: &'a mut Reactor<Context, E, F, P>,
    ) -> Self {
        Self { base, reactor }
    }
}

// ── InstructionSink for ReactorHandle ─────────────────────────────────────────

impl<'a, 'cb, 'ctx, Context, E, F, P> InstructionSink<Context, E>
    for ReactorHandle<'a, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.tail().instruction(ctx, insn)
    }
}

// ── BaseContext for ReactorHandle ─────────────────────────────────────────────

impl<'a, 'cb, 'ctx, Context, E, F, P> BaseContext<Context, E>
    for ReactorHandle<'a, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn layout(&self) -> &LocalLayout { self.base.layout() }
    fn layout_mut(&mut self) -> &mut LocalLayout { self.base.layout_mut() }
    fn locals_mark(&self) -> Mark { self.base.locals_mark() }
    fn set_locals_mark(&mut self, mark: Mark) { self.base.set_locals_mark(mark); }

    fn base_func_offset(&self) -> u32 { self.base.base_func_offset }
    fn set_base_func_offset(&mut self, n: u32) {
        self.base.base_func_offset = n;
        self.reactor.set_base_func_offset(n);
    }

    fn declare_trap_params(&mut self) { self.base.declare_trap_params(); }
    fn declare_trap_locals(&mut self) { self.base.declare_trap_locals(); }
    fn alloc_cell(&mut self) -> CellIdx { self.base.alloc_cell() }
    fn alloc_cell_for_guest(
        &mut self,
        guest_params: &[ValType],
        guest_results: &[ValType],
        guest_locals: &[(u32, ValType)],
    ) -> CellIdx {
        self.base.alloc_cell_for_guest(guest_params, guest_results, guest_locals)
    }
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        self.base.declare_trap_locals_with_cell(cell);
    }

    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
    ) -> Result<TrapAction, E> {
        let layout = &self.base.layout as *const LocalLayout;
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.base.traps.on_instruction(info, ctx, &mut *self.reactor, layout_ref)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        let layout = &self.base.layout as *const LocalLayout;
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.base.traps.on_jump(info, ctx, &mut *self.reactor, layout_ref)
    }
}

// ── ReactorContext for ReactorHandle ─────────────────────────────────────────

impl<'a, 'cb, 'ctx, Context, E, F, P> ReactorContext<Context, E>
    for ReactorHandle<'a, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + yecta::LocalPoolApi,
{
    type FnType = F;

    fn fn_count(&self) -> usize { self.reactor.fn_count() }
    fn drain_fns(&mut self) -> Vec<F> { self.reactor.drain_fns() }
    fn seal_remaining(&mut self, ctx: &mut Context) -> Result<(), E> {
        self.reactor.seal_remaining(ctx)
    }

    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<usize, E> {
        self.reactor.next_with(ctx, f, len)?;
        Ok(self.reactor.fn_count() - 1)
    }
    fn feed(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        Fed { reactor: &*self.reactor, tail_idx }.instruction(ctx, insn)
    }
    fn jmp(&self, ctx: &mut Context, tail_idx: usize, target: FuncIdx, params: u32) -> Result<(), E> {
        self.reactor.jmp(tail_idx, ctx, target, params)
    }
    fn seal_fn(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.seal_to(tail_idx, ctx, insn)
    }

    fn pool(&self) -> Pool<'_, Context, E> { self.base.pool }
    fn escape_tag(&self) -> Option<EscapeTag> { self.base.escape_tag }
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>) { self.base.escape_tag = tag; }

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

/// Owns [`LinkerInner`] and a [`LinkerPlugin`]; orchestrates multi-binary
/// linking via [`execute_schedule`](Self::execute_schedule).
///
/// ## Type parameters
/// - `'cb` — lifetime of borrowed trap implementations.
/// - `'ctx` — lifetime of data captured inside trap callbacks.
/// - `Context` — translation context forwarded to the reactor.
/// - `E` — error type.
/// - `Plugin` — the [`LinkerPlugin`] (defaults to `()`).
///
/// The function-sink type `F` is determined per
/// [`execute_schedule`](Self::execute_schedule) call, not at struct
/// construction time.
pub struct Linker<'cb, 'ctx, Context, E, Plugin = ()> {
    /// All state except the plugin.
    pub inner: LinkerInner<'cb, 'ctx, Context, E>,
    /// Downstream handler for completed [`BinaryUnit`]s.
    pub plugin: Plugin,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl<'cb, 'ctx, Context, E> Linker<'cb, 'ctx, Context, E, ()> {
    /// Create a `Linker` with the default no-op plugin.
    pub fn new() -> Self {
        Self::with_plugin(())
    }
}

impl<'cb, 'ctx, Context, E> Default for Linker<'cb, 'ctx, Context, E, ()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'cb, 'ctx, Context, E, Plugin> Linker<'cb, 'ctx, Context, E, Plugin>

{
    /// Create a `Linker` with the given plugin.
    pub fn with_plugin(plugin: Plugin) -> Self {
        use yecta::{TableIdx, TypeIdx};
        Self {
            inner: LinkerInner {
                traps: TrapConfig::new(),
                layout: LocalLayout::empty(),
                locals_mark: Mark { slot_count: 0, total_locals: 0 },
                injected_start: Mark { slot_count: 0, total_locals: 0 },
                signature: FuncSignature::empty(),
                base_func_offset: 0,
                pool: {
                    static T: TableIdx = TableIdx(0);
                    Pool { handler: &T, ty: TypeIdx(0) }
                },
                escape_tag: None,
                cell_registry: CellRegistry::new(),
                entity_space: EntityIndexSpace::empty(),
            },
            plugin,
        }
    }
}

// ── BaseContext for Linker (delegates to inner) ───────────────────────────────

impl<'cb, 'ctx, Context, E, Plugin> BaseContext<Context, E>
    for Linker<'cb, 'ctx, Context, E, Plugin>

{
    fn layout(&self) -> &LocalLayout { self.inner.layout() }
    fn layout_mut(&mut self) -> &mut LocalLayout { self.inner.layout_mut() }
    fn locals_mark(&self) -> Mark { self.inner.locals_mark() }
    fn set_locals_mark(&mut self, mark: Mark) { self.inner.set_locals_mark(mark); }
    fn base_func_offset(&self) -> u32 { self.inner.base_func_offset() }
    fn set_base_func_offset(&mut self, n: u32) { self.inner.set_base_func_offset(n); }
    fn declare_trap_params(&mut self) { self.inner.declare_trap_params(); }
    fn declare_trap_locals(&mut self) { self.inner.declare_trap_locals(); }
    fn alloc_cell(&mut self) -> CellIdx { self.inner.alloc_cell() }
    fn alloc_cell_for_guest(
        &mut self, gp: &[ValType], gr: &[ValType], gl: &[(u32, ValType)],
    ) -> CellIdx {
        self.inner.alloc_cell_for_guest(gp, gr, gl)
    }
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        self.inner.declare_trap_locals_with_cell(cell);
    }
    fn on_instruction(&mut self, info: &InstructionInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        self.inner.on_instruction(info, ctx)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        self.inner.on_jump(info, ctx)
    }
}

// ── execute_schedule ──────────────────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, Plugin> Linker<'cb, 'ctx, Context, E, Plugin>

{
    /// Execute a [`FuncSchedule`](speet_schedule::FuncSchedule) using this linker.
    ///
    /// Creates a fresh [`Reactor`] for the duration of this call.  Every emit
    /// closure in the schedule receives a [`ReactorHandle`] that wraps the
    /// linker's [`LinkerInner`] and the freshly-created reactor.
    ///
    /// After the schedule completes the reactor is dropped; the next call to
    /// `execute_schedule` always starts with a clean reactor.
    pub fn execute_schedule<'a, F>(
        &mut self,
        schedule: speet_schedule::FuncSchedule<'a, Context, E, F>,
        ctx: &mut Context,
    ) where
        F: InstructionSink<Context, E>,
        Plugin: LinkerPlugin<F>,
        LocalPool: LocalPoolBackend + yecta::LocalPoolApi + Default,
    {
        let Linker { ref mut inner, ref mut plugin, .. } = *self;
        let mut reactor: Reactor<Context, E, F, LocalPool> = Reactor::default();
        let mut handle = ReactorHandle::new(inner, &mut reactor);
        schedule.execute(&mut handle, plugin, ctx);
    }
}
