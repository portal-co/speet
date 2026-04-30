//! [`BaseContext`] and [`ReactorContext`] — the environment a recompiler borrows.
//!
//! [`BaseContext`] provides layout management and trap coordination for any
//! translation frontend — including the WASM frontend which does not use a
//! yecta Reactor.
//!
//! [`ReactorContext`] extends [`BaseContext`] with reactor emission methods
//! (`feed`, `jmp`, `next_with`, `seal_fn`) used only by native recompilers
//! backed by yecta.
//!
//! The canonical implementation is [`Linker`](crate::linker::Linker), which
//! owns the reactor and traps.  Recompilers receive a `&mut impl
//! ReactorContext<Context, E>` and call all emission/firing methods through it.
//!
//! ## `FnType` associated type
//!
//! The underlying function type `F` (e.g. `wasm_encoder::Function`) is
//! surfaced as the associated type `FnType`.  This lets [`Recompile::drain_unit`]
//! return `BinaryUnit<RC::FnType>` without the recompiler needing to know
//! `F` directly.
//!
//! ## Context hierarchy
//!
//! Three concrete context types exist, in increasing capability order:
//!
//! | Type | Traps | Cells | Reactor ownership |
//! |------|-------|-------|-------------------|
//! | [`ReactorAdapter`] | no-op | `CellIdx(0)` always | borrowed |
//! | [`TrapReactorAdapter`] | full | real `CellRegistry` | borrowed |
//! | `LinkerInner` (speet-linker) | full | real `CellRegistry` | **owned** |
//!
//! Use `ReactorAdapter` for simple smoke tests and benchmarks that install no
//! traps and do not inspect cells.  Use `TrapReactorAdapter` whenever a trap
//! is installed or correct per-cell local allocation is required (e.g. the e2e
//! trap tests).  Use `LinkerInner` for production multi-binary linking.

use alloc::vec::Vec;
use speet_traps::{InstructionInfo, JumpInfo, TrapAction, TrapConfig};
use wasm_encoder::{Instruction, ValType};
use wax_core::build::InstructionSink;
use yecta::{EscapeTag, Fed, FuncIdx, LocalLayout, LocalPoolBackend, Mark, Pool, Reactor, TableIdx, TypeIdx};
use yecta::layout::{CellIdx, CellRegistry};

// ── BaseContext ───────────────────────────────────────────────────────────────

/// Minimal context for layout management and trap/mapper coordination.
///
/// Does not require a yecta Reactor, so it can be used by the WASM frontend
/// as well as native recompilers.
///
/// ## Type parameters
/// - `Context` — the translation context passed through to instruction sinks.
/// - `E` — the error type.
pub trait BaseContext<Context, E> {
    // ── Layout ────────────────────────────────────────────────────────────

    /// Read-only access to the unified parameter + local layout.
    fn layout(&self) -> &LocalLayout;

    /// Mutable access to the unified parameter + local layout.
    fn layout_mut(&mut self) -> &mut LocalLayout;

    /// The [`Mark`] placed after all parameter slots.
    fn locals_mark(&self) -> Mark;

    /// Overwrite the stored [`Mark`].
    fn set_locals_mark(&mut self, mark: Mark);

    // ── Function offset ───────────────────────────────────────────────────

    /// Absolute WASM function index of the first function compiled so far.
    fn base_func_offset(&self) -> u32;

    /// Set `base_func_offset` to an absolute value.
    ///
    /// Used by [`FuncSchedule`] to position each binary slot at its
    /// pre-computed base before invoking the emit closure.
    fn set_base_func_offset(&mut self, n: u32);

    /// Advance `base_func_offset` by `n`.
    ///
    /// Non-yecta frontends that accumulate `F` values directly call this at
    /// the end of `drain_unit` to keep the linker's offset in sync.
    fn advance_base_func_offset(&mut self, n: u32) {
        self.set_base_func_offset(self.base_func_offset() + n);
    }

    // ── Trap support ──────────────────────────────────────────────────────

    /// **Phase 1** — let installed traps append their parameter groups to the
    /// internal layout.
    ///
    /// Call this after the arch recompiler has appended its own parameter
    /// groups to `layout_mut()`, then call [`set_locals_mark`](Self::set_locals_mark)
    /// with `layout().mark()` to record the total parameter count.
    fn declare_trap_params(&mut self);

    /// **Phase 2** — let installed traps append their per-function local groups
    /// to the internal layout.
    ///
    /// Call this after the arch recompiler has appended its own per-function
    /// locals and before building the output function.
    fn declare_trap_locals(&mut self);

    /// Fire the instruction trap (if any).
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.
    fn on_instruction(
        &mut self,
        info: &InstructionInfo,
        ctx: &mut Context,
    ) -> Result<TrapAction, E>;

    /// Fire the jump trap (if any).
    ///
    /// Returns [`TrapAction::Continue`] when no trap is installed.
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E>;

    /// Allocate (or retrieve) a [`CellIdx`] for the current function's
    /// (params, locals) signature.
    ///
    /// Call this after both [`declare_trap_params`](Self::declare_trap_params)
    /// and [`declare_trap_locals`](Self::declare_trap_locals) have finished so
    /// that the layout is fully populated.  The returned cell uniquely
    /// identifies the combination of function-type params and non-param locals
    /// that the current function uses.
    ///
    /// Implementations that own a [`CellRegistry`] should override this to
    /// return a semantically unique cell.  The default returns `CellIdx(0)`,
    /// which is the legacy placeholder behaviour.
    fn alloc_cell(&mut self) -> CellIdx {
        CellIdx(0)
    }

    /// Pre-allocate a guest-keyed [`CellIdx`] from the guest function type
    /// and guest local declarations.
    ///
    /// Unlike [`alloc_cell`](Self::alloc_cell), this may be called *before*
    /// any `declare_locals` call — as soon as the parsed guest function type
    /// and declared locals are available.  The returned cell can then be
    /// forwarded to every `declare_locals` invocation (mapper and trap) for
    /// that function.
    ///
    /// * `guest_params`  — parameter types from the guest WASM function type.
    /// * `guest_results` — result types from the guest WASM function type.
    /// * `guest_locals`  — `(count, type)` groups from the function body's
    ///   locals section, in declaration order.
    ///
    /// Implementations that own a [`CellRegistry`] should override this.
    /// The default returns `CellIdx(0)`.
    fn alloc_cell_for_guest(
        &mut self,
        _guest_params: &[ValType],
        _guest_results: &[ValType],
        _guest_locals: &[(u32, ValType)],
    ) -> CellIdx {
        CellIdx(0)
    }

    /// Like [`declare_trap_locals`](Self::declare_trap_locals) but threads a
    /// pre-allocated [`CellIdx`] through to each trap's `declare_locals` call.
    ///
    /// Use this instead of `declare_trap_locals` when a real cell has been
    /// pre-allocated via [`alloc_cell_for_guest`](Self::alloc_cell_for_guest)
    /// so that traps receive the semantically correct cell rather than the
    /// `CellIdx(0)` placeholder.
    ///
    /// The default delegates to `declare_trap_locals` (preserving the
    /// placeholder behaviour for contexts that do not override it).
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        let _ = cell;
        self.declare_trap_locals();
    }
}

// ── ReactorContext ────────────────────────────────────────────────────────────

/// The unified environment a native recompiler borrows from its caller.
///
/// Extends [`BaseContext`] with reactor emission methods used only by
/// yecta-backed native frontends.
///
/// See the [module documentation](self) for an overview.
///
/// ## Type parameters
/// - `Context` — the translation context passed through to the reactor.
/// - `E` — the error type.
pub trait ReactorContext<Context, E>: BaseContext<Context, E> + InstructionSink<Context, E> {
    /// The function type produced by the underlying reactor.
    type FnType;

    // ── Reactor state ─────────────────────────────────────────────────────

    /// Number of compiled functions currently held in the reactor.
    fn fn_count(&self) -> usize;

    /// Drain all compiled functions from the reactor.
    ///
    /// After draining, `base_func_offset` is advanced by the drained count
    /// and the reactor is ready for the next binary unit.
    fn drain_fns(&mut self) -> Vec<Self::FnType>;

    /// Seal any functions that were never explicitly terminated.
    ///
    /// Appends `unreachable; end` to each unsealed function so the WASM
    /// binary validator sees properly terminated function bodies.  Call this
    /// once — before `drain_fns` — after finishing a translation pass.
    fn seal_remaining(&mut self, ctx: &mut Context) -> Result<(), E>;

    // ── Reactor operations ────────────────────────────────────────────────

    /// Start a new function in the reactor.
    ///
    /// Returns the `tail_idx` of the newly-created function.  Callers must
    /// save this index and pass it to every subsequent emission call
    /// (`feed`, `jmp`, `seal_fn`, `ji`, `ji_with_params`, `ret`) that
    /// targets this function.  Keeping `tail_idx` explicit enables parallel
    /// emission over multiple simultaneously-live functions.
    fn next_with(&mut self, ctx: &mut Context, f: Self::FnType, len: u32) -> Result<usize, E>;

    /// Emit a single WASM instruction into the function identified by `tail_idx`.
    fn feed(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E>;

    /// Emit an unconditional tail-call jump to `target`, forwarding `params`
    /// parameters, into the function identified by `tail_idx`.
    fn jmp(&self, ctx: &mut Context, tail_idx: usize, target: FuncIdx, params: u32) -> Result<(), E>;

    /// Seal the function group identified by `tail_idx` with a terminal instruction.
    fn seal_fn(&self, ctx: &mut Context, tail_idx: usize, insn: &Instruction<'_>) -> Result<(), E>;

    // ── Configuration ─────────────────────────────────────────────────────

    /// The indirect-call pool configuration.
    ///
    /// Returns a [`Pool`] whose handler lifetime is tied to `&self`.  The
    /// stored handler must outlive the context, so the returned `Pool` is
    /// valid for the duration of the borrow.
    fn pool(&self) -> Pool<'_, Context, E>;

    /// The escape tag used for exception-based control flow, if any.
    fn escape_tag(&self) -> Option<EscapeTag>;

    /// Set the escape tag.
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>);

    /// Emit an indirect jump or call using yecta's `ji` API.
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
    ) -> Result<(), E>;

    /// Emit an indirect jump or call using yecta's `ji_with_params` API.
    fn ji_with_params(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
        params: yecta::JumpCallParams<'_, Context, E>,
    ) -> Result<(), E>;

    /// Emit a return through the escape-tag mechanism.
    fn ret(&self, ctx: &mut Context, tail_idx: usize, params: u32, tag: EscapeTag) -> Result<(), E>;

    /// Seed the underlying local pool.
    ///
    /// The closure receives a `&mut dyn LocalPoolApi` so this method is
    /// dyn-object-safe.  All current uses are side-effectful seeding calls
    /// (`seed_i32` / `seed_i64`) that return no value.
    fn with_local_pool(&self, f: &mut dyn FnMut(&mut dyn yecta::LocalPoolApi));

    /// Flush all pending lazy store bundles (used by FENCE/SYNC handlers).
    ///
    /// Equivalent to `Reactor::flush_bundles`.
    fn flush_bundles(&self, ctx: &mut Context, tail_idx: usize) -> Result<(), E>;

    /// Flush any deferred (lazy) stores that might alias a load at `addr_local`.
    ///
    /// Equivalent to `Reactor::flush_bundles_for_load`.  Used by the ordering
    /// helpers in `speet-ordering` so they can work through a `ReactorContext`
    /// instead of a bare `Reactor`.
    fn flush_for_load(
        &self,
        ctx: &mut Context,
        addr_local: u32,
        addr_type: ValType,
        tail_idx: usize,
    ) -> Result<(), E>;

    /// Defer a store instruction via the lazy-bundle mechanism.
    ///
    /// Equivalent to `Reactor::feed_lazy`.  Used by the ordering helpers.
    fn feed_lazy(
        &self,
        ctx: &mut Context,
        addr_type: ValType,
        val_type: ValType,
        insn: &Instruction<'static>,
        tail_idx: usize,
    ) -> Result<(), E>;
}

// ── Blanket impl for Reactor (provides a minimal ReactorContext without traps) ─

/// A lightweight [`ReactorContext`] wrapper that bundles a [`Reactor`] with
/// fixed pool / escape-tag configuration.
///
/// This is useful for tests or simple translation passes that do not need the
/// full [`Linker`](crate::linker::Linker).
pub struct ReactorAdapter<'a, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// The underlying reactor.
    pub reactor: &'a mut Reactor<Context, E, F, P>,
    /// Layout shared with the calling recompiler.
    pub layout: LocalLayout,
    /// Mark placed after all parameter slots.
    pub locals_mark: Mark,
    /// Indirect-call pool handler and type.
    pub pool: Pool<'a, Context, E>,
    /// Optional escape tag.
    pub escape_tag: Option<EscapeTag>,
}

impl<'a, Context, E, F, P> InstructionSink<Context, E> for ReactorAdapter<'a, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.tail().instruction(ctx, insn)
    }
}

impl<'a, Context, E, F, P> BaseContext<Context, E> for ReactorAdapter<'a, Context, E, F, P>
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

    // No traps — no-op.
    fn declare_trap_params(&mut self) {}
    fn declare_trap_locals(&mut self) {}
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

impl<'a, Context, E, F, P> ReactorContext<Context, E> for ReactorAdapter<'a, Context, E, F, P>
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

// ── FedContext ────────────────────────────────────────────────────────────────

/// A `ReactorContext` reference bound to a saved `tail_idx`.
///
/// Bundles `&RC` with the index of the currently-active function, providing:
/// - `InstructionSink` — for passing to [`ObjectModel`](speet_object) methods
///   and other `&mut dyn InstructionSink` consumers without an unsafe cast.
/// - Thin delegation methods (`feed`, `jmp`, `ret`, `ji`, …) that inject
///   `tail_idx` automatically, so callers never pass it explicitly.
///
/// Create once after each [`ReactorContext::next_with`] call and pass
/// `&FedContext` to pure-emission helpers, `&mut FedContext` where
/// `InstructionSink` is needed.
pub struct FedContext<'a, Context, E, RC: ReactorContext<Context, E> + ?Sized> {
    pub rctx: &'a RC,
    pub tail_idx: usize,
    _phantom: core::marker::PhantomData<(Context, E)>,
}

impl<'a, Context, E, RC: ReactorContext<Context, E> + ?Sized> FedContext<'a, Context, E, RC> {
    pub fn new(rctx: &'a RC, tail_idx: usize) -> Self {
        Self { rctx, tail_idx, _phantom: core::marker::PhantomData }
    }

    pub fn feed(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.rctx.feed(ctx, self.tail_idx, insn)
    }
    pub fn jmp(&self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.rctx.jmp(ctx, self.tail_idx, target, params)
    }
    pub fn ret(&self, ctx: &mut Context, params: u32, tag: EscapeTag) -> Result<(), E> {
        self.rctx.ret(ctx, self.tail_idx, params, tag)
    }
    pub fn ji(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &alloc::collections::BTreeMap<u32, &dyn yecta::Snippet<Context, E>>,
        target: yecta::Target<Context, E>,
        call: Option<EscapeTag>,
        pool: Pool<'_, Context, E>,
        condition: Option<&dyn yecta::Snippet<Context, E>>,
    ) -> Result<(), E> {
        self.rctx.ji(ctx, self.tail_idx, params, fixups, target, call, pool, condition)
    }
    pub fn ji_with_params(
        &self,
        ctx: &mut Context,
        params: yecta::JumpCallParams<'_, Context, E>,
    ) -> Result<(), E> {
        self.rctx.ji_with_params(ctx, self.tail_idx, params)
    }
    pub fn seal_fn(&self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.rctx.seal_fn(ctx, self.tail_idx, insn)
    }

    pub fn pool(&self) -> Pool<'_, Context, E> { self.rctx.pool() }
    pub fn escape_tag(&self) -> Option<EscapeTag> { self.rctx.escape_tag() }
    pub fn locals_mark(&self) -> Mark { self.rctx.locals_mark() }
    pub fn base_func_offset(&self) -> u32 { self.rctx.base_func_offset() }
    pub fn layout(&self) -> &LocalLayout { self.rctx.layout() }
}

impl<'a, Context, E, RC: ReactorContext<Context, E> + ?Sized> InstructionSink<Context, E>
    for FedContext<'a, Context, E, RC>
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.rctx.feed(ctx, self.tail_idx, insn)
    }
}

// ── TrapReactorAdapter ────────────────────────────────────────────────────────

/// A [`ReactorContext`] with a borrowed reactor **and** full trap + cell support.
///
/// This sits between [`ReactorAdapter`] (no traps, no real cell allocation) and
/// `LinkerInner` (owned reactor, production multi-binary flow):
///
/// - Install traps via `rctx.traps.set_instruction_trap(…)` /
///   `rctx.traps.set_jump_trap(…)` **before** calling the recompiler's
///   `setup_traps`.
/// - After translation the `cell_registry` contains one entry per unique
///   `(params, locals)` combination encountered.
///
/// ## Three-phase protocol (same as `LinkerInner`)
///
/// **Phase 1** — `setup_traps` (arch recompiler):
/// ```ignore
/// recompiler.setup_traps(&mut rctx);   // appends arch params, calls declare_trap_params, places mark
/// ```
///
/// **Phase 2** — per function (`init_function` inside the recompiler):
/// ```ignore
/// rctx.layout_mut().rewind(&mark);
/// rctx.layout_mut().append(…);         // arch non-param locals
/// rctx.declare_trap_locals();           // trap appends its locals
/// let cell = rctx.alloc_cell();         // registers (params, locals) → CellIdx
/// ```
///
/// **Phase 3** — firing (inside `translate_instruction` / jump sites):
/// ```ignore
/// rctx.on_instruction(&info, ctx)?;
/// rctx.on_jump(&info, ctx)?;
/// ```
pub struct TrapReactorAdapter<'r, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// The underlying reactor (borrowed, not owned).
    pub reactor: &'r mut Reactor<Context, E, F, P>,
    /// Pluggable instruction-level and jump-level trap hooks.
    pub traps: TrapConfig<'cb, 'ctx, Context, E>,
    /// Unified layout: arch params + trap params, then per-function locals.
    pub layout: LocalLayout,
    /// Mark placed after all parameter slots.
    pub locals_mark: Mark,
    /// Indirect-call pool handler and type.
    pub pool: Pool<'r, Context, E>,
    /// Optional escape tag for exception-based control flow.
    pub escape_tag: Option<EscapeTag>,
    /// Registry mapping unique `(params, locals)` signatures to [`CellIdx`] handles.
    pub cell_registry: CellRegistry,
}

impl<'r, 'cb, 'ctx, Context, E, F, P> TrapReactorAdapter<'r, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    /// Create a new `TrapReactorAdapter` with empty traps and a fresh cell registry.
    ///
    /// Install traps on `rctx.traps` before calling `setup_traps` on the recompiler.
    pub fn new(
        reactor: &'r mut Reactor<Context, E, F, P>,
        pool: Pool<'r, Context, E>,
        escape_tag: Option<EscapeTag>,
    ) -> Self {
        Self {
            reactor,
            traps: TrapConfig::new(),
            layout: LocalLayout::empty(),
            locals_mark: Mark { slot_count: 0, total_locals: 0 },
            pool,
            escape_tag,
            cell_registry: CellRegistry::default(),
        }
    }
}

impl<'r, 'cb, 'ctx, Context, E, F, P> InstructionSink<Context, E>
    for TrapReactorAdapter<'r, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn instruction(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.tail().instruction(ctx, insn)
    }
}

impl<'r, 'cb, 'ctx, Context, E, F, P> BaseContext<Context, E>
    for TrapReactorAdapter<'r, 'cb, 'ctx, Context, E, F, P>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
{
    fn layout(&self) -> &LocalLayout { &self.layout }
    fn layout_mut(&mut self) -> &mut LocalLayout { &mut self.layout }
    fn locals_mark(&self) -> Mark { self.locals_mark }
    fn set_locals_mark(&mut self, mark: Mark) { self.locals_mark = mark; }

    fn base_func_offset(&self) -> u32 { self.reactor.base_func_offset() }
    fn set_base_func_offset(&mut self, n: u32) { self.reactor.set_base_func_offset(n); }

    fn declare_trap_params(&mut self) {
        self.traps.declare_params(CellIdx(0), &mut self.layout);
    }
    fn declare_trap_locals(&mut self) {
        self.traps.declare_locals(CellIdx(0), &mut self.layout);
    }
    fn declare_trap_locals_with_cell(&mut self, cell: CellIdx) {
        self.traps.declare_locals(cell, &mut self.layout);
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
        // SAFETY: layout is borrowed immutably; self.traps and self.reactor are disjoint fields.
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.traps.on_instruction(info, ctx, &mut *self.reactor, layout_ref)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        let layout = &self.layout as *const LocalLayout;
        let layout_ref: &dyn yecta::LocalAllocator = unsafe { &*layout };
        self.traps.on_jump(info, ctx, &mut *self.reactor, layout_ref)
    }
}

impl<'r, 'cb, 'ctx, Context, E, F, P> ReactorContext<Context, E>
    for TrapReactorAdapter<'r, 'cb, 'ctx, Context, E, F, P>
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

    fn pool(&self) -> Pool<'_, Context, E> { self.pool }
    fn escape_tag(&self) -> Option<EscapeTag> { self.escape_tag }
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>) { self.escape_tag = tag; }

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
