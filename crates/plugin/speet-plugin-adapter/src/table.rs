//! Table adapter — bridges [`TablePlugin`] into the real
//! `yecta::IndirectJumpHandler`.
//!
//! `IndirectJumpHandler` is not a `LocalDeclarator` either (see
//! `object_model` module docs for the same constraint), so
//! [`PluginIndirectJumpHandler`] takes its scratch local as a constructor
//! argument. The internal trait's `indirect_jump` fires with the dynamic
//! jump-target value already on the wasm stack (pushed by whatever
//! computed it); the plugin instead takes an explicit `target_local`, so
//! the adapter stages the stack value there before calling the plugin and
//! replaying its snippet — same pattern as the memory adapters.

use alloc::sync::Arc;
use core::marker::PhantomData;

use speet_plugin_api::error::PluginError;
use speet_plugin_api::table::{PluginIndirectJumpKind, TablePlugin};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;
use yecta::{IndirectJumpHandler, IndirectJumpKind, TableIdx};

use crate::replay::replay_snippet;

fn decode_error<E: From<PluginError>>(message: alloc::string::String) -> E {
    E::from(PluginError::decode_failure(message))
}

pub struct PluginIndirectJumpHandler<Context, E> {
    plugin: Arc<dyn TablePlugin>,
    scratch_local: u32,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E> PluginIndirectJumpHandler<Context, E> {
    pub fn new(plugin: Arc<dyn TablePlugin>, scratch_local: u32) -> Self {
        Self {
            plugin,
            scratch_local,
            _marker: PhantomData,
        }
    }
}

impl<Context, E: From<PluginError>> IndirectJumpHandler<Context, E>
    for PluginIndirectJumpHandler<Context, E>
{
    fn indirect_jump(
        &self,
        ctx: &mut Context,
        target: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<IndirectJumpKind, E> {
        target.instruction(ctx, &Instruction::LocalSet(self.scratch_local))?;
        let (kind, snippet) = self.plugin.indirect_jump(self.scratch_local).map_err(E::from)?;
        replay_snippet(&snippet, ctx, target, decode_error)?;
        Ok(match kind {
            PluginIndirectJumpKind::Table(idx) => IndirectJumpKind::Table(TableIdx(idx)),
            PluginIndirectJumpKind::Ref => IndirectJumpKind::Ref,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use speet_plugin_api::snippet::CodeSnippet;
    use speet_plugin_api::PResult;

    #[derive(Debug)]
    #[allow(dead_code)]
    struct TestErr(String);
    impl From<PluginError> for TestErr {
        fn from(e: PluginError) -> Self {
            TestErr(format!("{e}"))
        }
    }

    struct RecordingSink {
        seen: Vec<String>,
    }
    impl InstructionSink<(), TestErr> for RecordingSink {
        fn instruction(&mut self, _ctx: &mut (), instruction: &Instruction<'_>) -> Result<(), TestErr> {
            self.seen.push(format!("{instruction:?}"));
            Ok(())
        }
    }

    /// Identity table dispatch: trusts whatever's already in `target_local`
    /// is a valid table index — mirrors the built-in `TableIdx` handler's
    /// own zero-instruction behavior.
    struct ToyTable;
    impl TablePlugin for ToyTable {
        fn indirect_jump(&self, _target_local: u32) -> PResult<(PluginIndirectJumpKind, CodeSnippet)> {
            Ok((PluginIndirectJumpKind::Table(7), CodeSnippet::empty()))
        }
    }

    const SCRATCH: u32 = 50;
    const DYNAMIC_VALUE_LOCAL: u32 = 51;

    #[test]
    fn stages_and_dispatches() {
        let handler: PluginIndirectJumpHandler<(), TestErr> =
            PluginIndirectJumpHandler::new(Arc::new(ToyTable), SCRATCH);

        let mut recorder = RecordingSink { seen: Vec::new() };
        recorder
            .instruction(&mut (), &Instruction::LocalGet(DYNAMIC_VALUE_LOCAL))
            .unwrap();
        let kind = handler.indirect_jump(&mut (), &mut recorder).unwrap();

        assert_eq!(kind, IndirectJumpKind::Table(TableIdx(7)));
        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(DYNAMIC_VALUE_LOCAL)),
                format!("{:?}", Instruction::LocalSet(SCRATCH)),
            ]
        );
    }
}
