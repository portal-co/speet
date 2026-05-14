//! [`InterpBodyBuilder`] — pluggable Thompson-threaded interpreter builder trait.
//!
//! Implementors emit WASM function bodies through [`InstructionSink<Context, E>`]
//! targets, keeping the interpreter on the same abstraction as the rest of the
//! codebase.  Because `wasm_encoder::Function` already implements
//! `InstructionSink`, snippets, `MemoryAccess`, and `TrapConfig` helpers work
//! unchanged inside interpreter handlers.

use alloc::boxed::Box;
use wax_core::build::InstructionSink;

use crate::context::InterpBuildCtx;

// ── InterpBodyBuilder ─────────────────────────────────────────────────────────

/// Pluggable Thompson-threaded interpreter body builder.
///
/// The caller creates one WASM function body per slot and passes them as sinks:
///
/// * `dispatch_sink` — the top-level fetch/decode/dispatch function (slot 1 of
///   the `OobInterp` reservation).
/// * `handler_sinks[i]` — one handler per opcode class (slots 2..2+N).
///
/// `build_interp` must emit exactly `1 + num_handler_fns()` function bodies (one
/// dispatch, N handlers).  Both bodies must end with `End` or `finish()`.
///
/// ## Inheriting context
///
/// [`InterpBuildCtx`] carries optional inherited memory access, trap hooks, and
/// the finalised local layout.  Builders can use these via the adapter types
/// ([`FlatEmitSink`], [`FlatMemorySink`]) exported from [`crate::context`].
pub trait InterpBodyBuilder<Context, E> {
    /// Number of opcode-handler functions beyond the top-level dispatch fn.
    ///
    /// This value must be constant: it is used during Phase 1 slot pre-registration
    /// before any instruction bytes are examined.
    fn num_handler_fns(&self) -> u32;

    /// WASM local declarations for the dispatch function (beyond its params).
    ///
    /// Used by [`OobInterp::emit_with_builder`] to create the `Function` with
    /// the right local groups before calling [`build_interp`].
    fn dispatch_fn_locals(&self) -> alloc::vec::Vec<(u32, wasm_encoder::ValType)> {
        alloc::vec::Vec::new()
    }

    /// WASM local declarations for handler function `i` (beyond its params).
    fn handler_fn_locals(&self, _handler_idx: u32) -> alloc::vec::Vec<(u32, wasm_encoder::ValType)> {
        alloc::vec::Vec::new()
    }

    /// Emit the dispatch function body into `dispatch_sink` and each handler
    /// body into `handler_sinks[i]`.
    ///
    /// `ictx` carries resolved WASM indices (function indices, memory/table
    /// indices) and optional inherited memory/trap configuration.
    fn build_interp(
        &mut self,
        dispatch_sink: &mut dyn InstructionSink<Context, E>,
        handler_sinks: &mut [Box<dyn InstructionSink<Context, E>>],
        ctx: &mut Context,
        ictx: &mut InterpBuildCtx<'_, Context, E>,
    ) -> Result<(), E>;
}

// ── NullInterpBuilder ─────────────────────────────────────────────────────────

/// Zero-cost null builder that emits the original `unreachable` stub.
///
/// Used as the default when no arch-specific interpreter is available.
/// Preserves the pre-existing behaviour of `OobInterp::generate`.
pub struct NullInterpBuilder;

impl<C, E> InterpBodyBuilder<C, E> for NullInterpBuilder {
    fn num_handler_fns(&self) -> u32 {
        0
    }

    fn build_interp(
        &mut self,
        dispatch_sink: &mut dyn InstructionSink<C, E>,
        _handler_sinks: &mut [Box<dyn InstructionSink<C, E>>],
        ctx: &mut C,
        _ictx: &mut InterpBuildCtx<'_, C, E>,
    ) -> Result<(), E> {
        use wasm_encoder::Instruction;
        dispatch_sink.instruction(ctx, &Instruction::Unreachable)?;
        dispatch_sink.instruction(ctx, &Instruction::End)
    }
}
