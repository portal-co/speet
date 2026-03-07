//! [`CallbackContext`] and [`MapperCallback`] — the generic mapper-callback
//! interface shared by all architecture recompilers.

use wax_core::build::InstructionSink;
use wasm_encoder::Instruction;

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
        Self { sink, _pd: core::marker::PhantomData }
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
pub trait MapperCallback<Context, E, F: InstructionSink<Context, E>> {
    /// Emit the address-translation wasm instructions.
    fn call(
        &mut self,
        ctx: &mut Context,
        callback_ctx: &mut CallbackContext<Context, E, F>,
    ) -> Result<(), E>;
}

/// Blanket impl: any compatible `FnMut` closure is a `MapperCallback`.
impl<Context, E, F: InstructionSink<Context, E>, T> MapperCallback<Context, E, F> for T
where
    T: FnMut(&mut Context, &mut CallbackContext<Context, E, F>) -> Result<(), E>,
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
