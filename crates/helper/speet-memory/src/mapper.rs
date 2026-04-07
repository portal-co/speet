//! [`CallbackContext`] and [`MapperCallback`] — the generic mapper-callback
//! interface shared by all architecture recompilers.

use yecta::{LocalDeclarator, LocalLayout, layout::CellIdx};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

// ── CallbackContext ────────────────────────────────────────────────────────────

/// A thin wrapper around a mutable reference to any [`InstructionSink`] that
/// lets address-mapper callbacks emit wasm instructions without depending on
/// the concrete sink type.
///
/// Recompilers construct this on the stack immediately before calling a mapper
/// and pass `&mut callback_ctx` to the mapper.
///
/// ```ignore
/// if let Some(mapper) = self.mapper_callback.as_mut() {
///     let mut cb = CallbackContext::new(&mut self.reactor);
///     mapper.call(ctx, &mut cb)?;
/// }
/// ```
///
/// The type parameter `F` is the concrete [`InstructionSink`] stored inside.
/// In practice recompilers always wrap their `Reactor<Context, E, OuterF>`
/// in a `CallbackContext<'_, Context, E, Reactor<Context, E, OuterF>>`.
/// Since `Reactor` implements `InstructionSink`, everything composes cleanly.
pub struct CallbackContext<'a, Context, E, F: InstructionSink<Context, E>> {
    /// The underlying instruction sink (usually a `Reactor`).
    pub sink: &'a mut F,
    // PhantomData for the Context/E parameters so the struct stays generic.
    _pd: core::marker::PhantomData<fn(&mut Context) -> Result<(), E>>,
}

impl<'a, Context, E, F: InstructionSink<Context, E>> CallbackContext<'a, Context, E, F> {
    /// Construct a `CallbackContext` wrapping `sink`.
    #[inline]
    pub fn new(sink: &'a mut F) -> Self {
        Self {
            sink,
            _pd: core::marker::PhantomData,
        }
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
/// A mapper callback transforms a guest virtual address (already on the wasm
/// value stack) into the corresponding physical/host address (also left on the
/// wasm value stack) by emitting any required wasm instructions.
///
/// The type parameter `F` is the concrete [`InstructionSink`] the recompiler
/// uses internally (typically `Reactor<Context, E, OuterF>`).  This matches
/// the `F` inside [`CallbackContext`] so that the mapper can call
/// `callback_ctx.emit(ctx, &instr)` without a double-indirect.
///
/// # Stack contract
/// * **Before call**: virtual address of type `i32` or `i64` is on the stack.
/// * **After call**: physical address of the same type is on the stack.
///
/// # Blanket implementation
///
/// Any `FnMut(&mut Context, &mut CallbackContext<…, F>) -> Result<(), E>`
/// closure automatically implements `MapperCallback`.
pub trait MapperCallback<Context, E, F: InstructionSink<Context, E>>: LocalDeclarator {
    /// Emit the address-translation wasm instructions.
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E>;

    /// Page granularity for data-segment chunking.
    ///
    /// When `Some(n)`, data segments should be split at `n`-byte page
    /// boundaries so that each chunk maps to exactly one physical page.
    /// Returns `None` when no chunking is required (e.g. identity mappers).
    fn chunk_size(&self) -> Option<u64> {
        None
    }
}

// ── ChunkedMapper ──────────────────────────────────────────────────────────────

/// Wraps any [`MapperCallback`] and advertises a fixed page granularity for
/// data-segment chunking via [`MapperCallback::chunk_size`].
///
/// All four page-table helper builders (`standard_page_table_mapper`, etc.)
/// return `ChunkedMapper<impl MapperCallback<…>>` with `page_size = 0x10000`
/// so that callers can derive the chunk size without knowing the concrete
/// mapper type.
pub struct ChunkedMapper<M> {
    /// The underlying mapper callback that performs address translation.
    pub inner: M,
    /// The page granularity advertised by [`MapperCallback::chunk_size`].
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

impl<Context, E, F, M> MapperCallback<Context, E, F> for ChunkedMapper<M>
where
    F: InstructionSink<Context, E>,
    M: MapperCallback<Context, E, F>,
{
    #[inline]
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E> {
        self.inner.call(ctx, callback_ctx)
    }

    #[inline]
    fn chunk_size(&self) -> Option<u64> {
        Some(self.page_size)
    }
}

/// Blanket impl: any compatible `FnMut` closure is a `MapperCallback`.
/// Closures do not need to declare wasm locals, so `LocalDeclarator` is
/// provided automatically via the no-op default implementations.
impl<Context, E, F: InstructionSink<Context, E>, T> MapperCallback<Context, E, F> for T
where
    T: FnMut(&mut Context, &mut CallbackContext<Context, E, F>) -> Result<(), E>
        + LocalDeclarator,
{
    #[inline]
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E> {
        self(ctx, callback_ctx)
    }
}
