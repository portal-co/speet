//! [`RuntimeLayoutParams`] and [`ParamSlotMap`] — layout values threaded as
//! function parameters (and returned) across all frontends.

use alloc::boxed::Box;
use core::convert::Infallible;
use wasm_encoder::{Instruction, ValType};
use wax_core::build::{InstructionOperatorSink, InstructionOperatorSource, InstructionSink, InstructionSource};
use yecta::{CellIdx, LocalDeclarator, LocalLayout, LocalSlot, Snippet};

/// Named slot handles for runtime layout values (after `declare_params`).
///
/// Stores [`LocalSlot`] handles — **not** raw `u32` local indices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParamSlotMap {
    pub text_base: LocalSlot,
    pub host_mem_base: LocalSlot,
}

/// How the embedder supplies the runtime text base at emission time.
#[derive(Clone, Copy, Debug)]
pub enum TextBaseSource {
    /// Load from a layout param slot (default thin-runtime path).
    ParamLocal(LocalSlot),
    /// Load from a WASM global.
    Global(u32),
    /// Compile-time constant guest VA.
    Constant(u64),
}

impl TextBaseSource {
    /// Default: param-local slot populated by entry_bridge.
    pub fn param_local(slot: LocalSlot) -> Self {
        Self::ParamLocal(slot)
    }
}

/// User-configurable snippet that pushes the runtime text base onto the WASM stack.
#[derive(Clone)]
pub struct TextBaseSnippet {
    source: TextBaseSource,
    /// Resolved absolute local index for `ParamLocal` (set after `declare_params`).
    resolved_local: Option<u32>,
}

impl TextBaseSnippet {
    pub fn new(source: TextBaseSource) -> Self {
        Self { source, resolved_local: None }
    }

    pub fn from_source(source: TextBaseSource) -> Self {
        Self::new(source)
    }

    /// Bind `ParamLocal` to the absolute wasm local index after layout seal.
    pub fn bind_local(&mut self, layout: &LocalLayout, slot: LocalSlot) {
        if matches!(self.source, TextBaseSource::ParamLocal(s) if s == slot) {
            self.resolved_local = Some(layout.local(slot, 0));
        }
    }
}

impl<Context, E> InstructionSource<Context, E> for TextBaseSnippet {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        match self.source {
            TextBaseSource::ParamLocal(slot) => {
                let idx = self.resolved_local.unwrap_or_else(|| {
                    panic!("TextBaseSnippet ParamLocal not bound for slot {slot:?}")
                });
                sink.instruction(ctx, &Instruction::LocalGet(idx))?;
            }
            TextBaseSource::Global(g) => sink.instruction(ctx, &Instruction::GlobalGet(g))?,
            TextBaseSource::Constant(c) => sink.instruction(ctx, &Instruction::I64Const(c as i64))?,
        }
        Ok(())
    }
}

impl<Context, E> InstructionOperatorSource<Context, E> for TextBaseSnippet {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(ctx, sink)
    }
}

/// Emits `((gpr - text_base) >> shift) + base_func_offset` as i64.
pub struct IndirectTableIdxSnippet<'a, Context, E> {
    pub gpr_local: u32,
    pub text_base: &'a dyn Snippet<Context, E>,
    pub slot_shift: u32,
    pub base_func_offset: u32,
}

impl<'a, Context, E> InstructionSource<Context, E> for IndirectTableIdxSnippet<'a, Context, E> {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.gpr_local))?;
        self.text_base.emit_snippet(ctx, &mut |ctx, instr| sink.instruction(ctx, instr))?;
        sink.instruction(ctx, &Instruction::I64Sub)?;
        if self.slot_shift > 0 {
            sink.instruction(ctx, &Instruction::I64Const(self.slot_shift as i64))?;
            sink.instruction(ctx, &Instruction::I64ShrU)?;
        }
        sink.instruction(ctx, &Instruction::I64Const(self.base_func_offset as i64))?;
        sink.instruction(ctx, &Instruction::I64Add)?;
        Ok(())
    }
}

impl<'a, Context, E> InstructionOperatorSource<Context, E> for IndirectTableIdxSnippet<'a, Context, E> {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(ctx, sink)
    }
}

/// Phase-1 registrant for layout params (`text_base`, `host_mem_base`, …).
#[derive(Clone)]
pub struct RuntimeLayoutParams {
    pub slots: ParamSlotMap,
    pub text_base_snippet: TextBaseSnippet,
}

impl RuntimeLayoutParams {
    pub fn new() -> Self {
        Self {
            slots: ParamSlotMap::default(),
            text_base_snippet: TextBaseSnippet::new(TextBaseSource::ParamLocal(LocalSlot::default())),
        }
    }

    pub fn with_text_base_source(source: TextBaseSource) -> Self {
        let mut s = Self::new();
        s.text_base_snippet = TextBaseSnippet::new(source);
        s
    }

    /// Bind snippet locals after layout is sealed.
    pub fn bind_snippets(&mut self, layout: &LocalLayout) {
        self.text_base_snippet
            .bind_local(layout, self.slots.text_base);
        if let TextBaseSource::ParamLocal(slot) = self.text_base_snippet.source {
            self.text_base_snippet.source = TextBaseSource::ParamLocal(self.slots.text_base);
            let _ = slot;
        }
    }
}

impl Default for RuntimeLayoutParams {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalDeclarator for RuntimeLayoutParams {
    fn declare_params(&mut self, _cell: CellIdx, layout: &mut LocalLayout) {
        self.slots.text_base = layout.append(1, ValType::I64);
        self.slots.host_mem_base = layout.append(1, ValType::I64);
        self.text_base_snippet.source = TextBaseSource::ParamLocal(self.slots.text_base);
    }
}

/// Convenience: default thin-runtime layout params (param-local text base).
pub fn default_runtime_layout_params() -> RuntimeLayoutParams {
    RuntimeLayoutParams::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_slot_map_uses_handles() {
        let mut layout = LocalLayout::empty();
        let mut params = RuntimeLayoutParams::new();
        params.declare_params(CellIdx(0), &mut layout);
        assert_ne!(params.slots.text_base, LocalSlot::default());
        assert_ne!(params.slots.host_mem_base, LocalSlot::default());
    }
}
