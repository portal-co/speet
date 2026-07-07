//! Object-model adapter — bridges [`ObjectModelPlugin`] into the real
//! `speet_object::ObjectModel`.
//!
//! `ObjectModel` is not a `LocalDeclarator` (no params/locals declare
//! phase), so unlike the memory adapters, [`PluginObjectModel`] takes its
//! scratch locals as constructor arguments — the same convention the
//! *internal* trait itself already uses for `emit_aput`'s `scratch_i32`/
//! `scratch_i64` and `emit_instanceof`/`emit_check_cast`'s `scratch`.
//!
//! Every internal method's stack-contract input (`ref`, `index`/`length`,
//! `value`) is staged into one of these scratch locals immediately before
//! the plugin call, mirroring the memory adapters' approach — see their
//! module docs for the underlying reasoning. `value`'s scratch local is
//! picked between the i32/i64 slot based on `FieldValType`'s register
//! representation (see `speet_object::model` module docs).

use alloc::sync::Arc;
use core::marker::PhantomData;

use speet_object::{FieldValType, ObjectModel, TypeHash};
use speet_plugin_api::error::PluginError;
use speet_plugin_api::object_model::{ObjectModelPlugin, PluginFieldValType, PluginTypeHash};
use wasm_encoder::Instruction;
use wax_core::build::InstructionSink;

use crate::replay::replay_snippet;

fn convert_field_val_type(ty: FieldValType) -> PluginFieldValType {
    match ty {
        FieldValType::I32 => PluginFieldValType::I32,
        FieldValType::I64 => PluginFieldValType::I64,
        FieldValType::F32 => PluginFieldValType::F32,
        FieldValType::F64 => PluginFieldValType::F64,
        FieldValType::I8S => PluginFieldValType::I8S,
        FieldValType::I8U => PluginFieldValType::I8U,
        FieldValType::I16S => PluginFieldValType::I16S,
        FieldValType::I16U => PluginFieldValType::I16U,
        FieldValType::Ref => PluginFieldValType::Ref,
    }
}

fn convert_type_hash(hash: &TypeHash) -> PluginTypeHash {
    PluginTypeHash(hash.0)
}

/// `true` if `ty`'s register representation is an `i64` wasm local (`I64`,
/// `F64` bit-pattern) rather than `i32` — see `speet_object::model` module
/// docs' register/memory convention table.
fn value_is_64(ty: FieldValType) -> bool {
    matches!(ty, FieldValType::I64 | FieldValType::F64)
}

fn decode_error<E: From<PluginError>>(message: alloc::string::String) -> E {
    E::from(PluginError::decode_failure(message))
}

pub struct PluginObjectModel<C, E> {
    plugin: Arc<dyn ObjectModelPlugin>,
    scratch_ref: u32,
    scratch_index: u32,
    scratch_value_i32: u32,
    scratch_value_i64: u32,
    _marker: PhantomData<fn(C, E)>,
}

impl<C, E> PluginObjectModel<C, E> {
    /// `scratch_ref` must be declared with `ref_val_type()`'s wasm type;
    /// `scratch_index` is `i32`; `scratch_value_i32`/`scratch_value_i64`
    /// are `i32`/`i64` respectively. The caller (which already owns the
    /// surrounding function's local layout, since `ObjectModel` has no
    /// declare phase of its own) is responsible for allocating all four.
    pub fn new(
        plugin: Arc<dyn ObjectModelPlugin>,
        scratch_ref: u32,
        scratch_index: u32,
        scratch_value_i32: u32,
        scratch_value_i64: u32,
    ) -> Self {
        Self {
            plugin,
            scratch_ref,
            scratch_index,
            scratch_value_i32,
            scratch_value_i64,
            _marker: PhantomData,
        }
    }

    fn value_scratch(&self, ty: FieldValType) -> u32 {
        if value_is_64(ty) {
            self.scratch_value_i64
        } else {
            self.scratch_value_i32
        }
    }
}

impl<C, E: From<PluginError>> ObjectModel<C, E> for PluginObjectModel<C, E> {
    fn ref_val_type(&self) -> wasm_encoder::ValType {
        self.plugin.ref_val_type().into()
    }

    fn emit_new_object(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        hash: &TypeHash,
        data_size: u32,
    ) -> Result<(), E> {
        let snippet = self
            .plugin
            .emit_new_object(convert_type_hash(hash), data_size)
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_new_array(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        elem_hash: &TypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> Result<(), E> {
        // Stack before: [length: i32].
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_index))?;
        let snippet = self
            .plugin
            .emit_new_array(self.scratch_index, convert_type_hash(elem_hash), dim, elem_bytes)
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_iget(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E> {
        // Stack before: [ref].
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_iget(self.scratch_ref, byte_offset, convert_field_val_type(ty))
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_iput(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        byte_offset: u32,
        ty: FieldValType,
    ) -> Result<(), E> {
        // Stack before: [ref, value] — value (top) popped first.
        let value_scratch = self.value_scratch(ty);
        sink.instruction(ctx, &Instruction::LocalSet(value_scratch))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_iput(self.scratch_ref, value_scratch, byte_offset, convert_field_val_type(ty))
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_aget(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        ty: FieldValType,
    ) -> Result<(), E> {
        // Stack before: [ref, index] — index (top) popped first.
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_index))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_aget(self.scratch_ref, self.scratch_index, convert_field_val_type(ty))
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_aput(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        ty: FieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> Result<(), E> {
        // Stack before: [ref, index, value] — value (top) popped first.
        let value_scratch = self.value_scratch(ty);
        sink.instruction(ctx, &Instruction::LocalSet(value_scratch))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_index))?;
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_aput(
                self.scratch_ref,
                self.scratch_index,
                value_scratch,
                convert_field_val_type(ty),
                scratch_i32,
                scratch_i64,
            )
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_array_length(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self.plugin.emit_array_length(self.scratch_ref).map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_instanceof(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_instanceof(self.scratch_ref, convert_type_hash(hash), dim, scratch)
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_check_cast(
        &self,
        ctx: &mut C,
        sink: &mut (dyn InstructionSink<C, E> + '_),
        hash: &TypeHash,
        dim: u32,
        scratch: u32,
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalSet(self.scratch_ref))?;
        let snippet = self
            .plugin
            .emit_check_cast(self.scratch_ref, convert_type_hash(hash), dim, scratch)
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use speet_plugin_api::object_model::PluginTypeHash;
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

    /// Toy model: `iget`/`aput` just echo their staged locals back, proving
    /// the adapter staged the right value into the right scratch slot —
    /// the snippet itself doesn't need to do anything realistic.
    struct ToyModel;
    impl ObjectModelPlugin for ToyModel {
        fn ref_val_type(&self) -> speet_plugin_api::snippet::PluginValType {
            speet_plugin_api::snippet::PluginValType::I32
        }
        fn emit_new_object(&self, _hash: PluginTypeHash, _data_size: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_new_array(
            &self,
            _length_local: u32,
            _elem_hash: PluginTypeHash,
            _dim: u32,
            _elem_bytes: u32,
        ) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_iget(&self, ref_local: u32, _byte_offset: u32, _ty: PluginFieldValType) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(ref_local)]))
        }
        fn emit_iput(
            &self,
            _ref_local: u32,
            _value_local: u32,
            _byte_offset: u32,
            _ty: PluginFieldValType,
        ) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_aget(&self, _ref_local: u32, _index_local: u32, _ty: PluginFieldValType) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_aput(
            &self,
            ref_local: u32,
            index_local: u32,
            value_local: u32,
            _ty: PluginFieldValType,
            _scratch_i32: u32,
            _scratch_i64: u32,
        ) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[
                Instruction::LocalGet(ref_local),
                Instruction::LocalGet(index_local),
                Instruction::LocalGet(value_local),
            ]))
        }
        fn emit_array_length(&self, _ref_local: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_instanceof(
            &self,
            _ref_local: u32,
            _hash: PluginTypeHash,
            _dim: u32,
            _scratch: u32,
        ) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
        fn emit_check_cast(
            &self,
            _ref_local: u32,
            _hash: PluginTypeHash,
            _dim: u32,
            _scratch: u32,
        ) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::empty())
        }
    }

    const SCRATCH_REF: u32 = 60;
    const SCRATCH_INDEX: u32 = 61;
    const SCRATCH_VALUE_I32: u32 = 62;
    const SCRATCH_VALUE_I64: u32 = 63;
    const REF_LOCAL: u32 = 70;
    const INDEX_LOCAL: u32 = 71;
    const VALUE_LOCAL: u32 = 72;

    fn model() -> PluginObjectModel<(), TestErr> {
        PluginObjectModel::new(
            Arc::new(ToyModel),
            SCRATCH_REF,
            SCRATCH_INDEX,
            SCRATCH_VALUE_I32,
            SCRATCH_VALUE_I64,
        )
    }

    #[test]
    fn emit_iget_stages_single_ref() {
        let model = model();
        let mut recorder = RecordingSink { seen: Vec::new() };
        recorder.instruction(&mut (), &Instruction::LocalGet(REF_LOCAL)).unwrap();
        model
            .emit_iget(&mut (), &mut recorder, 4, FieldValType::I32)
            .unwrap();
        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(REF_LOCAL)),
                format!("{:?}", Instruction::LocalSet(SCRATCH_REF)),
                format!("{:?}", Instruction::LocalGet(SCRATCH_REF)),
            ]
        );
    }

    #[test]
    fn emit_aput_stages_ref_index_value_in_pop_order() {
        let model = model();
        let mut recorder = RecordingSink { seen: Vec::new() };
        recorder.instruction(&mut (), &Instruction::LocalGet(REF_LOCAL)).unwrap();
        recorder.instruction(&mut (), &Instruction::LocalGet(INDEX_LOCAL)).unwrap();
        recorder.instruction(&mut (), &Instruction::LocalGet(VALUE_LOCAL)).unwrap();
        model
            .emit_aput(&mut (), &mut recorder, FieldValType::I32, 90, 91)
            .unwrap();
        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(REF_LOCAL)),
                format!("{:?}", Instruction::LocalGet(INDEX_LOCAL)),
                format!("{:?}", Instruction::LocalGet(VALUE_LOCAL)),
                // pop order: value, then index, then ref
                format!("{:?}", Instruction::LocalSet(SCRATCH_VALUE_I32)),
                format!("{:?}", Instruction::LocalSet(SCRATCH_INDEX)),
                format!("{:?}", Instruction::LocalSet(SCRATCH_REF)),
                // snippet echoes ref, index, value back via the staged scratch locals
                format!("{:?}", Instruction::LocalGet(SCRATCH_REF)),
                format!("{:?}", Instruction::LocalGet(SCRATCH_INDEX)),
                format!("{:?}", Instruction::LocalGet(SCRATCH_VALUE_I32)),
            ]
        );
    }

    #[test]
    fn emit_aput_picks_i64_scratch_for_long_value() {
        let model = model();
        let mut recorder = RecordingSink { seen: Vec::new() };
        recorder.instruction(&mut (), &Instruction::LocalGet(REF_LOCAL)).unwrap();
        recorder.instruction(&mut (), &Instruction::LocalGet(INDEX_LOCAL)).unwrap();
        recorder.instruction(&mut (), &Instruction::LocalGet(VALUE_LOCAL)).unwrap();
        model
            .emit_aput(&mut (), &mut recorder, FieldValType::I64, 90, 91)
            .unwrap();
        assert!(recorder
            .seen
            .contains(&format!("{:?}", Instruction::LocalSet(SCRATCH_VALUE_I64))));
    }
}
