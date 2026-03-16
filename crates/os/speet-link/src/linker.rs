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
//! ## Usage
//!
//! ```ignore
//! use speet_link::linker::Linker;
//! use speet_link::builder::MegabinaryBuilder;
//!
//! let mut linker: Linker<_, _, wasm_encoder::Function> =
//!     Linker::with_plugin(MegabinaryBuilder::new());
//!
//! // Install a trap:
//! linker.traps.set_instruction_trap(&mut my_trap);
//!
//! // Translate binary 1:
//! let mut rc = MyRecompiler::new();
//! rc.setup(&mut linker);
//! translate(&bytes1, &mut rc, &mut linker);
//! linker.commit(&mut rc, entry_points_1);
//!
//! // Retrieve result:
//! let output = linker.plugin.finish();
//! ```

use alloc::{string::String, vec::Vec};
use speet_traps::{InstructionInfo, JumpInfo, TrapAction, TrapConfig};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::{EscapeTag, FuncIdx, LocalLayout, LocalPool, LocalPoolBackend, Mark, Pool, Reactor};

use crate::context::{BaseContext, ReactorContext};
use crate::recompiler::Recompile;
use crate::unit::BinaryUnit;

// ── LinkerPlugin ──────────────────────────────────────────────────────────────

/// Callback invoked by the [`Linker`] once per committed [`BinaryUnit`].
///
/// Implement this to accumulate units into a final module
/// ([`MegabinaryBuilder`](crate::builder::MegabinaryBuilder)) or to perform
/// per-unit analysis.
///
/// Use `()` as a no-op plugin when you do not need to inspect individual units.
pub trait LinkerPlugin<F> {
    /// Called after each successful [`Linker::commit`] or
    /// [`Linker::commit_raw`].
    fn on_unit(&mut self, unit: BinaryUnit<F>);
}

/// No-op plugin — discards every unit.
impl<F> LinkerPlugin<F> for () {
    fn on_unit(&mut self, _unit: BinaryUnit<F>) {}
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
    /// The reactor that generates WASM functions.
    pub reactor: Reactor<Context, E, F, P>,
    /// Pluggable instruction-level and jump-level trap hooks.
    pub traps: TrapConfig<'cb, 'ctx, Context, E, Reactor<Context, E, F, P>>,
    /// Unified layout: arch params + trap params, then per-function locals.
    pub layout: LocalLayout,
    /// Mark placed after all parameter slots (captures `total_params`).
    pub locals_mark: Mark,
    /// Indirect-call pool configuration.
    pub pool: Pool,
    /// Optional escape tag for exception-based control flow.
    pub escape_tag: Option<EscapeTag>,
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
            reactor: Reactor::default(),
            traps: TrapConfig::new(),
            layout: LocalLayout::empty(),
            locals_mark: Mark {
                slot_count: 0,
                total_locals: 0,
            },
            pool: Pool {
                table: TableIdx(0),
                ty: TypeIdx(0),
            },
            escape_tag: None,
            plugin,
        }
    }

    // ── Commit helpers ────────────────────────────────────────────────────

    /// Total number of WASM function slots committed so far (including any
    /// currently being compiled).
    pub fn total_fn_count(&self) -> u32 {
        self.reactor.base_func_offset() + self.reactor.fn_count() as u32
    }

    /// Drain a recompiler's functions, build a [`BinaryUnit`], and forward it
    /// to the plugin.
    ///
    /// `entry_points` — `(symbol, absolute_wasm_func_index)` pairs that will
    /// become exports in the final module.
    pub fn commit<B>(
        &mut self,
        rc: &mut (dyn Recompile<Context, E, F, BinaryArgs = B> + '_),
        entry_points: Vec<(String, u32)>,
    ) {
        let unit = rc.drain_unit(self, entry_points);
        self.plugin.on_unit(unit);
    }

    /// Commit a pre-built unit directly (e.g. an ABI shim) without going
    /// through a recompiler.
    pub fn commit_raw(&mut self, unit: BinaryUnit<F>) {
        self.plugin.on_unit(unit);
    }
}

// ── BaseContext impl ──────────────────────────────────────────────────────────

impl<'cb, 'ctx, Context, E, F, P, Plugin> BaseContext<Context, E>
    for Linker<'cb, 'ctx, Context, E, F, P, Plugin>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Plugin: LinkerPlugin<F>,
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
    fn advance_base_func_offset(&mut self, n: u32) {
        self.reactor
            .set_base_func_offset(self.reactor.base_func_offset() + n);
    }

    fn declare_trap_params(&mut self) {
        self.traps.declare_params(&mut self.layout);
    }
    fn declare_trap_locals(&mut self) {
        self.traps.declare_locals(&mut self.layout);
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
        let layout_ref = unsafe { &*layout };
        self.traps
            .on_instruction(info, ctx, &mut self.reactor, layout_ref)
    }
    fn on_jump(&mut self, info: &JumpInfo, ctx: &mut Context) -> Result<TrapAction, E> {
        let layout = &self.layout as *const LocalLayout;
        let layout_ref = unsafe { &*layout };
        self.traps.on_jump(info, ctx, &mut self.reactor, layout_ref)
    }
}

// ── ReactorContext impl ───────────────────────────────────────────────────────

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
        self.reactor.fn_count()
    }
    fn drain_fns(&mut self) -> Vec<F> {
        self.reactor.drain_fns()
    }

    fn feed(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.feed(ctx, insn)
    }
    fn jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        self.reactor.jmp(ctx, target, params)
    }
    fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<(), E> {
        self.reactor.next_with(ctx, f, len)
    }
    fn seal_fn(&mut self, ctx: &mut Context, insn: &Instruction<'_>) -> Result<(), E> {
        self.reactor.seal(ctx, insn)
    }

    fn pool(&self) -> &Pool {
        &self.pool
    }
    fn escape_tag(&self) -> Option<EscapeTag> {
        self.escape_tag
    }
    fn set_escape_tag(&mut self, tag: Option<EscapeTag>) {
        self.escape_tag = tag;
    }
}
