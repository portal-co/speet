//! Memory adapters — bridge [`AddressMapperPlugin`]/[`MemoryAccessPlugin`]
//! into the real `speet_memory::{AddressMapper, MemoryAccess}` + `LocalDeclarator`.
//!
//! ## The stack-vs-local mismatch
//!
//! The internal traits pass the guest/physical address *on the WASM value
//! stack* (see each method's stack-contract doc in `speet-memory`); the
//! plugin traits instead take an explicit `addr_local: u32` (see
//! `speet_plugin_api::memory` module docs — the plugin's snippet reads it
//! itself wherever it needs the address). Every adapter method here bridges
//! the two by staging the stack value into a private scratch local (one per
//! adapter instance, sized as `addr_val_type`) immediately before the
//! plugin call, then replaying the plugin's snippet — which is responsible
//! for pushing the result back onto the stack per its own documented
//! contract.
//!
//! `emit_store_addr`/`emit_store_insn` are the one exception: the internal
//! contract needs the physical address available *twice* — once on the
//! stack (so the caller can push the store value next) and again as a local
//! for `emit_store_insn`. `emit_store_addr` therefore `local.tee`s (not
//! `local.set`s) the snippet's result into the same scratch local, so it
//! survives on the stack *and* is available as `phys_addr_local` next call.
//!
//! ## Error conversion
//!
//! Every plugin call can fail with a [`PluginError`], but the internal
//! traits return `Result<_, E>` for the caller's own `E`. Adapters in this
//! module therefore require `E: From<PluginError>`.

use alloc::sync::Arc;
use core::marker::PhantomData;

use speet_memory::mem::{LoadKind, StoreKind};
use speet_memory::mapper::{AddressMapper, MemoryAccess};
use speet_ordering::MemorySink;
use speet_plugin_api::error::PluginError;
use speet_plugin_api::memory::{AddressMapperPlugin, MemoryAccessPlugin, PluginLoadKind, PluginStoreKind};
use speet_plugin_api::snippet::PluginValType;
use wasm_encoder::{Instruction, ValType};
use yecta::layout::CellIdx;
use yecta::{LocalDeclarator, LocalLayout};

use crate::replay::replay_snippet;

fn convert_load_kind(kind: LoadKind) -> PluginLoadKind {
    match kind {
        LoadKind::I8S => PluginLoadKind::I8S,
        LoadKind::I8U => PluginLoadKind::I8U,
        LoadKind::I16S => PluginLoadKind::I16S,
        LoadKind::I16U => PluginLoadKind::I16U,
        LoadKind::I32S => PluginLoadKind::I32S,
        LoadKind::I32U => PluginLoadKind::I32U,
        LoadKind::I64 => PluginLoadKind::I64,
        LoadKind::F32 => PluginLoadKind::F32,
        LoadKind::F64 => PluginLoadKind::F64,
    }
}

fn convert_store_kind(kind: StoreKind) -> PluginStoreKind {
    match kind {
        StoreKind::I8 => PluginStoreKind::I8,
        StoreKind::I16 => PluginStoreKind::I16,
        StoreKind::I32 => PluginStoreKind::I32,
        StoreKind::I64 => PluginStoreKind::I64,
        StoreKind::F32 => PluginStoreKind::F32,
        StoreKind::F64 => PluginStoreKind::F64,
    }
}

/// Append each of `types` as its own one-element `LocalLayout` group, in
/// order. Returns the absolute index of the first appended local (the
/// plugin's own params/locals are contiguous from there), or `None` if
/// `types` is empty.
fn append_each(layout: &mut LocalLayout, types: &[PluginValType]) -> Option<u32> {
    let mut base = None;
    for ty in types {
        let slot = layout.append(1, (*ty).into());
        base.get_or_insert_with(|| layout.base(slot));
    }
    base
}

fn decode_error<E: From<PluginError>>(message: alloc::string::String) -> E {
    E::from(PluginError::decode_failure(message))
}

// ── PluginAddressMapper ──────────────────────────────────────────────────────

pub struct PluginAddressMapper<Context, E> {
    plugin: Arc<dyn AddressMapperPlugin>,
    addr_val_type: ValType,
    param_base: Option<u32>,
    scratch_local: Option<u32>,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E> PluginAddressMapper<Context, E> {
    pub fn new(plugin: Arc<dyn AddressMapperPlugin>, addr_val_type: ValType) -> Self {
        Self {
            plugin,
            addr_val_type,
            param_base: None,
            scratch_local: None,
            _marker: PhantomData,
        }
    }
}

impl<Context, E> LocalDeclarator for PluginAddressMapper<Context, E> {
    fn declare_params(&mut self, _cell: CellIdx, layout: &mut LocalLayout) {
        let types = self.plugin.declare_params();
        self.param_base = append_each(layout, &types);
    }

    fn declare_locals(&mut self, _cell: CellIdx, layout: &mut LocalLayout) {
        let types = self.plugin.declare_locals();
        let local_base = append_each(layout, &types);
        let slot = layout.append(1, self.addr_val_type);
        self.scratch_local = Some(layout.base(slot));
        self.plugin
            .bind_slots(self.param_base.unwrap_or(0), local_base.unwrap_or(0));
    }
}

impl<Context, E: From<PluginError>> AddressMapper<Context, E> for PluginAddressMapper<Context, E> {
    fn translate(&mut self, ctx: &mut Context, sink: &mut dyn MemorySink<Context, E>) -> Result<(), E> {
        let scratch = self
            .scratch_local
            .expect("PluginAddressMapper::translate called before declare_locals");
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;
        let snippet = self.plugin.translate(scratch).map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn chunk_size(&self) -> Option<u64> {
        self.plugin.chunk_size()
    }
}

// ── PluginMemoryAccess ────────────────────────────────────────────────────────

pub struct PluginMemoryAccess<Context, E> {
    plugin: Arc<dyn MemoryAccessPlugin>,
    addr_val_type: ValType,
    param_base: Option<u32>,
    scratch_local: Option<u32>,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E> PluginMemoryAccess<Context, E> {
    pub fn new(plugin: Arc<dyn MemoryAccessPlugin>, addr_val_type: ValType) -> Self {
        Self {
            plugin,
            addr_val_type,
            param_base: None,
            scratch_local: None,
            _marker: PhantomData,
        }
    }
}

impl<Context, E> LocalDeclarator for PluginMemoryAccess<Context, E> {
    fn declare_params(&mut self, _cell: CellIdx, layout: &mut LocalLayout) {
        let types = self.plugin.declare_params();
        self.param_base = append_each(layout, &types);
    }

    fn declare_locals(&mut self, _cell: CellIdx, layout: &mut LocalLayout) {
        let types = self.plugin.declare_locals();
        let local_base = append_each(layout, &types);
        let slot = layout.append(1, self.addr_val_type);
        self.scratch_local = Some(layout.base(slot));
        self.plugin
            .bind_slots(self.param_base.unwrap_or(0), local_base.unwrap_or(0));
    }
}

impl<Context, E: From<PluginError>> MemoryAccess<Context, E> for PluginMemoryAccess<Context, E> {
    fn emit_load(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: LoadKind,
    ) -> Result<(), E> {
        let scratch = self
            .scratch_local
            .expect("PluginMemoryAccess::emit_load called before declare_locals");
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;
        let snippet = self
            .plugin
            .emit_load(scratch, convert_load_kind(kind))
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_store_addr(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let scratch = self
            .scratch_local
            .expect("PluginMemoryAccess::emit_store_addr called before declare_locals");
        sink.instruction(ctx, &Instruction::LocalSet(scratch))?;
        let snippet = self.plugin.emit_store_addr(scratch).map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)?;
        // Tee (not set): the physical address must survive on the stack for
        // the caller to push the store value next, *and* be available as
        // `phys_addr_local` for the paired `emit_store_insn` call — see
        // module docs.
        sink.instruction(ctx, &Instruction::LocalTee(scratch))
    }

    fn emit_store_insn(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: StoreKind,
    ) -> Result<(), E> {
        let scratch = self
            .scratch_local
            .expect("PluginMemoryAccess::emit_store_insn called before emit_store_addr");
        let snippet = self
            .plugin
            .emit_store_insn(scratch, convert_store_kind(kind))
            .map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_memory_size(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let snippet = self.plugin.emit_memory_size().map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn emit_memory_grow(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        let snippet = self.plugin.emit_memory_grow().map_err(E::from)?;
        replay_snippet(&snippet, ctx, sink, decode_error)
    }

    fn chunk_size(&self) -> Option<u64> {
        self.plugin.chunk_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use speet_ordering::EagerMemorySink;
    use speet_plugin_api::snippet::CodeSnippet;
    use speet_plugin_api::PResult;
    use wax_core::build::InstructionSink;
    use yecta::LocalLayout;

    #[derive(Debug)]
    #[allow(dead_code)] // field only read via the derived `Debug` on panic
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

    /// Identity address mapper: leaves the staged address unchanged.
    struct IdentityMapper;
    impl AddressMapperPlugin for IdentityMapper {
        fn translate(&self, addr_local: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(addr_local)]))
        }
    }

    fn memarg() -> wasm_encoder::MemArg {
        wasm_encoder::MemArg { offset: 0, align: 2, memory_index: 0 }
    }

    /// Toy memory access plugin: identity address mapping + plain i32 load/store.
    struct ToyMemoryAccess;
    impl MemoryAccessPlugin for ToyMemoryAccess {
        fn emit_load(&self, addr_local: u32, _kind: PluginLoadKind) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[
                Instruction::LocalGet(addr_local),
                Instruction::I32Load(memarg()),
            ]))
        }
        fn emit_store_addr(&self, addr_local: u32) -> PResult<CodeSnippet> {
            Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(addr_local)]))
        }
        fn emit_store_insn(&self, _phys_addr_local: u32, _kind: PluginStoreKind) -> PResult<CodeSnippet> {
            // Common case: consume `[address, value]` already on the stack
            // directly via a plain store — see module docs.
            Ok(CodeSnippet::from_instructions(&[Instruction::I32Store(memarg())]))
        }
    }

    const GUEST_ADDR_LOCAL: u32 = 100;
    const VALUE_LOCAL: u32 = 101;

    #[test]
    fn address_mapper_stages_and_replays() {
        let mut adapter: PluginAddressMapper<(), TestErr> =
            PluginAddressMapper::new(Arc::new(IdentityMapper), ValType::I32);

        let mut layout = LocalLayout::empty();
        adapter.declare_params(CellIdx(0), &mut layout);
        adapter.declare_locals(CellIdx(0), &mut layout);
        let scratch = adapter.scratch_local.unwrap();

        let mut recorder = RecordingSink { seen: Vec::new() };
        let mut sink = EagerMemorySink::new(&mut recorder);
        sink.instruction(&mut (), &Instruction::LocalGet(GUEST_ADDR_LOCAL)).unwrap();
        adapter.translate(&mut (), &mut sink).unwrap();

        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(GUEST_ADDR_LOCAL)),
                format!("{:?}", Instruction::LocalSet(scratch)),
                format!("{:?}", Instruction::LocalGet(scratch)),
            ]
        );
    }

    #[test]
    fn memory_access_load_and_store_round_trip() {
        let mut adapter: PluginMemoryAccess<(), TestErr> =
            PluginMemoryAccess::new(Arc::new(ToyMemoryAccess), ValType::I32);

        let mut layout = LocalLayout::empty();
        adapter.declare_params(CellIdx(0), &mut layout);
        adapter.declare_locals(CellIdx(0), &mut layout);
        let scratch = adapter.scratch_local.unwrap();

        // ── Load ──
        let mut recorder = RecordingSink { seen: Vec::new() };
        {
            let mut sink = EagerMemorySink::new(&mut recorder);
            sink.instruction(&mut (), &Instruction::LocalGet(GUEST_ADDR_LOCAL)).unwrap();
            adapter.emit_load(&mut (), &mut sink, LoadKind::I32U).unwrap();
        }
        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(GUEST_ADDR_LOCAL)),
                format!("{:?}", Instruction::LocalSet(scratch)),
                format!("{:?}", Instruction::LocalGet(scratch)),
                format!("{:?}", Instruction::I32Load(memarg())),
            ]
        );

        // ── Store ──
        let mut recorder = RecordingSink { seen: Vec::new() };
        {
            let mut sink = EagerMemorySink::new(&mut recorder);
            sink.instruction(&mut (), &Instruction::LocalGet(GUEST_ADDR_LOCAL)).unwrap();
            adapter.emit_store_addr(&mut (), &mut sink).unwrap();
            sink.instruction(&mut (), &Instruction::LocalGet(VALUE_LOCAL)).unwrap();
            adapter.emit_store_insn(&mut (), &mut sink, StoreKind::I32).unwrap();
        }
        assert_eq!(
            recorder.seen,
            alloc::vec![
                format!("{:?}", Instruction::LocalGet(GUEST_ADDR_LOCAL)),
                format!("{:?}", Instruction::LocalSet(scratch)),
                format!("{:?}", Instruction::LocalGet(scratch)),
                format!("{:?}", Instruction::LocalTee(scratch)),
                format!("{:?}", Instruction::LocalGet(VALUE_LOCAL)),
                format!("{:?}", Instruction::I32Store(memarg())),
            ]
        );
    }
}
