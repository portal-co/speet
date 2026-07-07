//! [`replay_snippet`] — decodes a [`CodeSnippet`]'s bare instruction-stream
//! bytes and forwards each instruction into a real `dyn InstructionSink`.
//! Shared by every adapter that crosses the snippet boundary (memory,
//! table, object-model — everything except the target adapter, which never
//! produces a `CodeSnippet`, and the arch adapter, added in a later phase).

use alloc::vec::Vec;

use wasm_encoder::reencode::{Reencode, utils};
use wasm_encoder::{Encode, Instruction};
use wasmparser::{BinaryReader, OperatorsReader};
use wax_core::build::InstructionSink;

use crate::CodeSnippet;

/// Zero-sized [`Reencode`] impl: `CodeSnippet`'s indices are already
/// host-relative absolute indices (see its own doc comment), so no
/// index remapping is needed — every `Reencode` hook keeps its identity
/// default.
struct IdentityReencode;
impl Reencode for IdentityReencode {
    type Error = core::convert::Infallible;
}

/// Decode `snippet.wasm` as a bare WASM instruction stream (no length
/// prefix, no trailing `End` required — see [`CodeSnippet`]'s own docs) and
/// call `sink.instruction(ctx, &instr)` for each one in order.
///
/// `on_error` converts a decode failure (malformed bytes from a
/// plugin — always a possibility for the WASM/subprocess hosts) into the
/// caller's `E`, with the failure's `Display` text as context.
pub fn replay_snippet<Context, E>(
    snippet: &CodeSnippet,
    ctx: &mut Context,
    sink: &mut dyn InstructionSink<Context, E>,
    on_error: impl Fn(alloc::string::String) -> E,
) -> Result<(), E> {
    let reader = OperatorsReader::new(BinaryReader::new(&snippet.wasm, 0));
    let mut reencoder = IdentityReencode;
    for op in reader {
        let op = op.map_err(|e| on_error(alloc::format!("{e}")))?;
        let instr = utils::instruction(&mut reencoder, op).map_err(|e| on_error(alloc::format!("{e:?}")))?;
        sink.instruction(ctx, &instr)?;
    }
    Ok(())
}

/// [`CodeRecorder`] — the dual of [`replay_snippet`]: a sink that just
/// encodes every instruction it receives into a byte buffer instead of
/// forwarding it anywhere. Used by the reverse adapters (`reverse` module)
/// to capture a built-in's emitted instructions as a [`CodeSnippet`] a
/// plugin can later replay.
///
/// Implements `InstructionSink<Context, E>` for *any* `Context`/`E` — it
/// never touches `ctx` and never fails — so it can stand in for the sink
/// argument of any internal trait method regardless of the caller's own
/// generic instantiation.
pub struct CodeRecorder {
    bytes: Vec<u8>,
}

impl CodeRecorder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Encode `instr` directly — a plain inherent method, not the generic
    /// `InstructionSink<Context, E>::instruction`, so callers staging a
    /// `LocalGet` before an internal call don't need to pin a fully
    /// unconstrained `E` (nothing else in that staging step would).
    pub fn push(&mut self, instr: &Instruction<'_>) {
        instr.encode(&mut self.bytes);
    }

    pub fn into_snippet(self) -> CodeSnippet {
        CodeSnippet { wasm: self.bytes }
    }
}

impl<Context, E> InstructionSink<Context, E> for CodeRecorder {
    fn instruction(&mut self, _ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        instruction.encode(&mut self.bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use wasm_encoder::Instruction;

    struct RecordingSink {
        seen: alloc::vec::Vec<alloc::string::String>,
    }
    impl InstructionSink<(), Infallible> for RecordingSink {
        fn instruction(&mut self, _ctx: &mut (), instruction: &Instruction<'_>) -> Result<(), Infallible> {
            self.seen.push(alloc::format!("{instruction:?}"));
            Ok(())
        }
    }

    #[test]
    fn replays_bare_instruction_stream() {
        let snippet = CodeSnippet::from_instructions(&[
            Instruction::LocalGet(3),
            Instruction::I64Const(7),
            Instruction::I64Add,
        ]);
        let mut sink = RecordingSink { seen: alloc::vec::Vec::new() };
        replay_snippet(&snippet, &mut (), &mut sink, |e| panic!("{e}")).unwrap();
        assert_eq!(
            sink.seen,
            alloc::vec![
                alloc::format!("{:?}", Instruction::LocalGet(3)),
                alloc::format!("{:?}", Instruction::I64Const(7)),
                alloc::format!("{:?}", Instruction::I64Add),
            ]
        );
    }
}
