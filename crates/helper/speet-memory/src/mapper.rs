//! [`CallbackContext`] and [`MapperCallback`] — the mapper-callback
//! interface shared by all architecture recompilers.

use yecta::{LocalDeclarator, LocalLayout, layout::CellIdx};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

// ── CallbackContext ────────────────────────────────────────────────────────────

/// A thin wrapper around a `dyn InstructionSink` reference that lets
/// address-mapper callbacks emit wasm instructions without depending on
/// any concrete sink type.
///
/// Recompilers construct this on the stack immediately before calling a
/// mapper and pass `&mut callback_ctx` to the mapper.
pub struct CallbackContext<'a, Context, E> {
    /// The underlying instruction sink.
    pub sink: &'a mut dyn InstructionSink<Context, E>,
}

impl<'a, Context, E> CallbackContext<'a, Context, E> {
    /// Construct a `CallbackContext` wrapping any `InstructionSink`.
    #[inline]
    pub fn new(sink: &'a mut dyn InstructionSink<Context, E>) -> Self {
        Self { sink }
    }

    /// Emit a single wasm instruction via the underlying sink.
    #[inline]
    pub fn emit(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.sink.instruction(ctx, instruction)
    }
}

// ── MapperCallback ─────────────────────────────────────────────────────────────

/// Trait for virtual-to-physical address translation callbacks.
///
/// # Stack contract
/// * **Before call**: virtual address of type `i32` or `i64` is on the stack.
/// * **After call**: physical address of the same type is on the stack.
pub trait MapperCallback<Context, E>: LocalDeclarator {
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) -> Result<(), E>;

    fn chunk_size(&self) -> Option<u64> {
        None
    }
}

// ── ChunkedMapper ──────────────────────────────────────────────────────────────

/// Wraps any [`MapperCallback`] and advertises a fixed page granularity for
/// data-segment chunking via [`MapperCallback::chunk_size`].
pub struct ChunkedMapper<M> {
    pub inner: M,
    pub page_size: u64,
}

impl<M: LocalDeclarator> LocalDeclarator for ChunkedMapper<M> {
    #[inline]
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.inner.declare_params(cell, layout);
    }

    #[inline]
    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.inner.declare_locals(cell, layout);
    }
}

impl<Context, E, M: MapperCallback<Context, E>> MapperCallback<Context, E> for ChunkedMapper<M> {
    #[inline]
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) -> Result<(), E> {
        self.inner.call(ctx, callback_ctx)
    }

    #[inline]
    fn chunk_size(&self) -> Option<u64> {
        Some(self.page_size)
    }
}

/// Blanket impl: any compatible `FnMut` closure is a `MapperCallback`.
impl<Context, E, T> MapperCallback<Context, E> for T
where
    T: FnMut(&mut Context, &mut CallbackContext<Context, E>) -> Result<(), E>
        + LocalDeclarator,
{
    #[inline]
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) -> Result<(), E> {
        self(ctx, callback_ctx)
    }
}

// ── StackedMapper ──────────────────────────────────────────────────────────────

/// Composes two [`MapperCallback`]s in series: the outer mapper runs first,
/// then the inner mapper refines the address further.
///
/// `chunk_size` is the minimum of the two component chunk sizes (with `None`
/// treated as the identity, i.e. "no constraint").
///
/// `LocalDeclarator` delegates to both components — outer first, then inner.
pub struct StackedMapper<Outer, Inner> {
    pub outer: Outer,
    pub inner: Inner,
}

impl<Outer: LocalDeclarator, Inner: LocalDeclarator> LocalDeclarator
    for StackedMapper<Outer, Inner>
{
    #[inline]
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.outer.declare_params(cell, layout);
        self.inner.declare_params(cell, layout);
    }

    #[inline]
    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        self.outer.declare_locals(cell, layout);
        self.inner.declare_locals(cell, layout);
    }
}

impl<Context, E, Outer, Inner> MapperCallback<Context, E>
    for StackedMapper<Outer, Inner>
where
    Outer: MapperCallback<Context, E>,
    Inner: MapperCallback<Context, E>,
{
    #[inline]
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E>,
    ) -> Result<(), E> {
        self.outer.call(ctx, callback_ctx)?;
        self.inner.call(ctx, callback_ctx)
    }

    #[inline]
    fn chunk_size(&self) -> Option<u64> {
        match (self.outer.chunk_size(), self.inner.chunk_size()) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(a.min(b)),
        }
    }
}
