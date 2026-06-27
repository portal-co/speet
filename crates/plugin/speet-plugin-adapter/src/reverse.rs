//! Reverse adapters — wrap an existing *built-in* internal resource so it
//! can be registered into a `PluginRegistry` and granted to other plugins
//! as a host-entity import (`speet_plugin_api::imports`), indistinguishable
//! from an externally-loaded plugin. See `docs/guides/plugin-api.md` §7.
//!
//! These are the mirror image of `memory`/`table`/`object_model`/`target`:
//! those wrap a plugin trait object to satisfy an *internal* trait; these
//! wrap a concrete *internal* value to satisfy a *plugin* trait.
//!
//! ## Manufacturing `Context` and capturing snippets
//!
//! A plugin method has no `ctx`/live sink to receive — it must return a
//! self-contained [`CodeSnippet`]. So every snippet-shaped reverse adapter:
//! 1. Manufactures a throwaway `Context::default()` (requiring
//!    `Context: Default` — true of the production `Context = ()`).
//! 2. Calls the wrapped value's internal method against a [`CodeRecorder`]
//!    (optionally wrapped in `EagerMemorySink` for the `MemorySink`-shaped
//!    methods), which just encodes instructions instead of forwarding them.
//! 3. Converts the internal `Result<_, E>` into [`PResult`] via `E: Into<PluginError>`.
//!
//! `declare_params`/`declare_locals` are handled the same way, but against
//! a throwaway `LocalLayout`: call the internal `LocalDeclarator` method,
//! then read back what it appended via `LocalLayout::iter_since`.
//!
//! `AddressMapper`/`MemoryAccess` methods take `&mut self`, so their
//! reverse adapters hold the wrapped value behind a `spin::Mutex` (the
//! `no_std`-compatible mutex already used elsewhere in this workspace,
//! e.g. `yecta`) to expose `&self` for `Arc`-sharing. `IndirectJumpHandler`
//! and `ObjectModel` are already `&self` internally, so no lock is needed.

use alloc::vec::Vec;
use core::marker::PhantomData;

use speet_memory::mapper::{AddressMapper, MemoryAccess};
use speet_memory::mem::{LoadKind, StoreKind};
use speet_object::{FieldValType, ObjectModel, TypeHash};
use speet_ordering::EagerMemorySink;
use speet_plugin_api::error::PluginError;
use speet_plugin_api::memory::{AddressMapperPlugin, MemoryAccessPlugin, PluginLoadKind, PluginStoreKind};
use speet_plugin_api::object_model::{ObjectModelPlugin, PluginFieldValType, PluginTypeHash};
use speet_plugin_api::snippet::{CodeSnippet, PluginValType};
use speet_plugin_api::target::{ModuleManifest, PluginSyscallTable};
use speet_plugin_api::target::TargetPlugin;
use speet_plugin_api::PResult;
use spin::Mutex;
use wasm_encoder::Instruction;
use yecta::layout::CellIdx;
use yecta::{LocalDeclarator, LocalLayout};

use crate::replay::CodeRecorder;

/// Run `f` against a fresh `LocalLayout`/throwaway cell and return the
/// `PluginValType`s it appended, flattening `(count, ty)` groups.
fn capture_declared<D: LocalDeclarator>(d: &mut D, run: impl FnOnce(&mut D, CellIdx, &mut LocalLayout)) -> Vec<PluginValType> {
    let mut layout = LocalLayout::empty();
    let mark = layout.mark();
    run(d, CellIdx(0), &mut layout);
    layout
        .iter_since(&mark)
        .flat_map(|(count, ty)| {
            let plugin_ty = PluginValType::try_from(ty)
                .expect("built-in declared a local type outside the plugin-representable subset");
            core::iter::repeat_n(plugin_ty, count as usize)
        })
        .collect()
}

fn into_plugin_error<E: Into<PluginError>>(e: E) -> PluginError {
    e.into()
}

// ── BuiltinAddressMapperAsPlugin ─────────────────────────────────────────────

pub struct BuiltinAddressMapperAsPlugin<Context, E, M> {
    inner: Mutex<M>,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E, M> BuiltinAddressMapperAsPlugin<Context, E, M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner: Mutex::new(inner),
            _marker: PhantomData,
        }
    }
}

impl<Context, E, M> AddressMapperPlugin for BuiltinAddressMapperAsPlugin<Context, E, M>
where
    Context: Default + Send,
    E: Into<PluginError> + Send,
    M: AddressMapper<Context, E> + Send,
{
    fn translate(&self, addr_local: u32) -> PResult<CodeSnippet> {
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(addr_local));
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner.translate(&mut ctx, &mut sink).map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn declare_params(&self) -> Vec<PluginValType> {
        capture_declared(&mut *self.inner.lock(), |m, cell, layout| m.declare_params(cell, layout))
    }

    fn declare_locals(&self) -> Vec<PluginValType> {
        capture_declared(&mut *self.inner.lock(), |m, cell, layout| m.declare_locals(cell, layout))
    }

    fn chunk_size(&self) -> Option<u64> {
        self.inner.lock().chunk_size()
    }
}

// ── BuiltinMemoryAccessAsPlugin ───────────────────────────────────────────────

pub struct BuiltinMemoryAccessAsPlugin<Context, E, M> {
    inner: Mutex<M>,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E, M> BuiltinMemoryAccessAsPlugin<Context, E, M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner: Mutex::new(inner),
            _marker: PhantomData,
        }
    }
}

fn convert_load_kind(kind: PluginLoadKind) -> LoadKind {
    match kind {
        PluginLoadKind::I8S => LoadKind::I8S,
        PluginLoadKind::I8U => LoadKind::I8U,
        PluginLoadKind::I16S => LoadKind::I16S,
        PluginLoadKind::I16U => LoadKind::I16U,
        PluginLoadKind::I32S => LoadKind::I32S,
        PluginLoadKind::I32U => LoadKind::I32U,
        PluginLoadKind::I64 => LoadKind::I64,
        PluginLoadKind::F32 => LoadKind::F32,
        PluginLoadKind::F64 => LoadKind::F64,
    }
}

fn convert_store_kind(kind: PluginStoreKind) -> StoreKind {
    match kind {
        PluginStoreKind::I8 => StoreKind::I8,
        PluginStoreKind::I16 => StoreKind::I16,
        PluginStoreKind::I32 => StoreKind::I32,
        PluginStoreKind::I64 => StoreKind::I64,
        PluginStoreKind::F32 => StoreKind::F32,
        PluginStoreKind::F64 => StoreKind::F64,
    }
}

impl<Context, E, M> MemoryAccessPlugin for BuiltinMemoryAccessAsPlugin<Context, E, M>
where
    Context: Default + Send,
    E: Into<PluginError> + Send,
    M: MemoryAccess<Context, E> + Send,
{
    fn emit_load(&self, addr_local: u32, kind: PluginLoadKind) -> PResult<CodeSnippet> {
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(addr_local));
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner
            .emit_load(&mut ctx, &mut sink, convert_load_kind(kind))
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_store_addr(&self, addr_local: u32) -> PResult<CodeSnippet> {
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(addr_local));
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner.emit_store_addr(&mut ctx, &mut sink).map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_store_insn(&self, phys_addr_local: u32, kind: PluginStoreKind) -> PResult<CodeSnippet> {
        // Common case (see `speet-plugin-adapter::memory` module docs): the
        // internal contract expects `[address, value]` already on the
        // stack; stage both back onto our recording sink's virtual stack
        // from the addresses the caller gave us (`phys_addr_local` for the
        // address — the value, per the plugin contract, is not given a
        // local, so we forward only the address and let the wrapped
        // `MemoryAccess` impl's own instruction sequence read the value via
        // whatever convention it already used when first recorded as a
        // plugin — see the type-level note below).
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(phys_addr_local));
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner
            .emit_store_insn(&mut ctx, &mut sink, convert_store_kind(kind))
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_memory_size(&self) -> PResult<CodeSnippet> {
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner.emit_memory_size(&mut ctx, &mut sink).map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_memory_grow(&self) -> PResult<CodeSnippet> {
        let mut inner = self.inner.lock();
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        let mut sink = EagerMemorySink::new(&mut recorder);
        inner.emit_memory_grow(&mut ctx, &mut sink).map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn declare_params(&self) -> Vec<PluginValType> {
        capture_declared(&mut *self.inner.lock(), |m, cell, layout| m.declare_params(cell, layout))
    }

    fn declare_locals(&self) -> Vec<PluginValType> {
        capture_declared(&mut *self.inner.lock(), |m, cell, layout| m.declare_locals(cell, layout))
    }

    fn chunk_size(&self) -> Option<u64> {
        self.inner.lock().chunk_size()
    }
}

// ── BuiltinTableAsPlugin ──────────────────────────────────────────────────────

pub struct BuiltinTableAsPlugin<Context, E, H> {
    inner: H,
    _marker: PhantomData<fn(Context, E)>,
}

impl<Context, E, H> BuiltinTableAsPlugin<Context, E, H> {
    pub fn new(inner: H) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<Context, E, H> speet_plugin_api::table::TablePlugin for BuiltinTableAsPlugin<Context, E, H>
where
    Context: Default + Send + Sync,
    E: Into<PluginError> + Send + Sync,
    H: yecta::IndirectJumpHandler<Context, E> + Send + Sync,
{
    fn indirect_jump(
        &self,
        target_local: u32,
    ) -> PResult<(speet_plugin_api::table::PluginIndirectJumpKind, CodeSnippet)> {
        let mut ctx = Context::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(target_local));
        let kind = self.inner.indirect_jump(&mut ctx, &mut recorder).map_err(into_plugin_error)?;
        let kind = match kind {
            yecta::IndirectJumpKind::Table(yecta::TableIdx(idx)) => {
                speet_plugin_api::table::PluginIndirectJumpKind::Table(idx)
            }
            yecta::IndirectJumpKind::Ref => speet_plugin_api::table::PluginIndirectJumpKind::Ref,
        };
        Ok((kind, recorder.into_snippet()))
    }
}

// ── BuiltinObjectModelAsPlugin ────────────────────────────────────────────────

pub struct BuiltinObjectModelAsPlugin<C, E, O> {
    inner: O,
    _marker: PhantomData<fn(C, E)>,
}

impl<C, E, O> BuiltinObjectModelAsPlugin<C, E, O> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

fn convert_field_val_type(ty: PluginFieldValType) -> FieldValType {
    match ty {
        PluginFieldValType::I32 => FieldValType::I32,
        PluginFieldValType::I64 => FieldValType::I64,
        PluginFieldValType::F32 => FieldValType::F32,
        PluginFieldValType::F64 => FieldValType::F64,
        PluginFieldValType::I8S => FieldValType::I8S,
        PluginFieldValType::I8U => FieldValType::I8U,
        PluginFieldValType::I16S => FieldValType::I16S,
        PluginFieldValType::I16U => FieldValType::I16U,
        PluginFieldValType::Ref => FieldValType::Ref,
    }
}

fn convert_type_hash(hash: PluginTypeHash) -> TypeHash {
    TypeHash(hash.0)
}

impl<C, E, O> ObjectModelPlugin for BuiltinObjectModelAsPlugin<C, E, O>
where
    C: Default + Send + Sync,
    E: Into<PluginError> + Send + Sync,
    O: ObjectModel<C, E> + Send + Sync,
{
    fn ref_val_type(&self) -> PluginValType {
        PluginValType::try_from(self.inner.ref_val_type())
            .expect("ObjectModel::ref_val_type returned a type outside the plugin-representable subset")
    }

    fn emit_new_object(&self, hash: PluginTypeHash, data_size: u32) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        self.inner
            .emit_new_object(&mut ctx, &mut recorder, &convert_type_hash(hash), data_size)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_new_array(
        &self,
        length_local: u32,
        elem_hash: PluginTypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(length_local));
        self.inner
            .emit_new_array(&mut ctx, &mut recorder, &convert_type_hash(elem_hash), dim, elem_bytes)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_iget(&self, ref_local: u32, byte_offset: u32, ty: PluginFieldValType) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        self.inner
            .emit_iget(&mut ctx, &mut recorder, byte_offset, convert_field_val_type(ty))
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_iput(
        &self,
        ref_local: u32,
        value_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    ) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        recorder.push(&Instruction::LocalGet(value_local));
        self.inner
            .emit_iput(&mut ctx, &mut recorder, byte_offset, convert_field_val_type(ty))
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_aget(&self, ref_local: u32, index_local: u32, ty: PluginFieldValType) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        recorder.push(&Instruction::LocalGet(index_local));
        self.inner
            .emit_aget(&mut ctx, &mut recorder, convert_field_val_type(ty))
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_aput(
        &self,
        ref_local: u32,
        index_local: u32,
        value_local: u32,
        ty: PluginFieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        recorder.push(&Instruction::LocalGet(index_local));
        recorder.push(&Instruction::LocalGet(value_local));
        self.inner
            .emit_aput(&mut ctx, &mut recorder, convert_field_val_type(ty), scratch_i32, scratch_i64)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_array_length(&self, ref_local: u32) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        self.inner
            .emit_array_length(&mut ctx, &mut recorder)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_instanceof(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        self.inner
            .emit_instanceof(&mut ctx, &mut recorder, &convert_type_hash(hash), dim, scratch)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }

    fn emit_check_cast(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> PResult<CodeSnippet> {
        let mut ctx = C::default();
        let mut recorder = CodeRecorder::new();
        recorder.push(&Instruction::LocalGet(ref_local));
        self.inner
            .emit_check_cast(&mut ctx, &mut recorder, &convert_type_hash(hash), dim, scratch)
            .map_err(into_plugin_error)?;
        Ok(recorder.into_snippet())
    }
}

// ── BuiltinTargetAsPlugin ─────────────────────────────────────────────────────

/// Wraps an already-materialized `(ModuleManifest, PluginSyscallTable)` pair
/// — e.g. produced once, up front, from a hand-written environment crate
/// like `speet-linux-wasi` — as a [`TargetPlugin`]. Unlike the other four
/// reverse adapters, there is no generic internal *trait* to wrap here:
/// `ModuleTarget` is a declaration *sink*, not a source, so the data must
/// already be in `ModuleManifest`/`PluginSyscallTable` form before
/// wrapping.
pub struct BuiltinTargetAsPlugin {
    manifest: ModuleManifest,
    syscall_table: PluginSyscallTable,
}

impl BuiltinTargetAsPlugin {
    pub fn new(manifest: ModuleManifest, syscall_table: PluginSyscallTable) -> Self {
        Self { manifest, syscall_table }
    }
}

impl TargetPlugin for BuiltinTargetAsPlugin {
    fn module_manifest(&self) -> ModuleManifest {
        self.manifest.clone()
    }

    fn syscall_table(&self) -> PluginSyscallTable {
        self.syscall_table.clone()
    }
}
