//! The six `Dylib*Plugin` adapters: each implements one `speet-plugin-api`
//! trait by encoding a `remote::XRequest`, driving it through
//! [`DylibPlugin::call_raw`], and decoding the `remote::XResponse`. Mirrors
//! `speet-plugin-host-wasm::adapters` almost exactly — same wire-envelope
//! types, same transport-failure-handling policy (see that module's doc
//! comment for the rationale): infallible methods panic on a transport
//! failure, `PResult`-returning methods convert it to `Err(PluginError)`.

use speet_plugin_api::arch::ArchOp;
use speet_plugin_api::error::PluginError;
use speet_plugin_api::imports::HostImports;
use speet_plugin_api::memory::{PluginLoadKind, PluginStoreKind};
use speet_plugin_api::object_model::{PluginFieldValType, PluginTypeHash};
use speet_plugin_api::remote::{
    AddressMapperRequest, AddressMapperResponse, ArchRequest, ArchResponse, MemoryAccessRequest,
    MemoryAccessResponse, ObjectModelRequest, ObjectModelResponse, TableRequest, TableResponse,
    TargetRequest, TargetResponse,
};
use speet_plugin_api::snippet::{CodeSnippet, PluginValType};
use speet_plugin_api::table::PluginIndirectJumpKind;
use speet_plugin_api::target::{ModuleManifest, PluginSyscallTable};
use speet_plugin_api::wire::{WireDecode, WireEncode};
use speet_plugin_api::{
    AddressMapperPlugin, ArchPlugin, MemoryAccessPlugin, ObjectModelPlugin, TablePlugin,
    TargetPlugin,
};

use super::engine::DylibPlugin;

fn roundtrip<Req: WireEncode, Resp: WireDecode>(engine: &DylibPlugin, req: Req) -> Resp {
    let mut payload = Vec::new();
    req.encode(&mut payload);
    let resp_bytes = engine
        .call_raw(&payload)
        .expect("dylib plugin call failed (transport-level)");
    let (resp, _) = Resp::decode(&resp_bytes).expect("dylib plugin returned malformed response");
    resp
}

// ── Arch ─────────────────────────────────────────────────────────────────

pub struct DylibArchPlugin {
    engine: DylibPlugin,
}
impl DylibArchPlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl ArchPlugin for DylibArchPlugin {
    fn reset_for_next_binary(&self, args: &[u8]) {
        let resp: ArchResponse = roundtrip(
            &self.engine,
            ArchRequest::ResetForNextBinary {
                args: args.to_vec(),
            },
        );
        assert!(matches!(resp, ArchResponse::ResetForNextBinary));
    }
    fn count_fns(&self, bytes: &[u8]) -> u32 {
        match roundtrip(&self.engine, ArchRequest::CountFns { bytes: bytes.to_vec() }) {
            ArchResponse::CountFns { count } => count,
            other => panic!("unexpected ArchResponse variant: {other:?}"),
        }
    }
    fn step(&self, feedback: Option<&[u8]>) -> speet_plugin_api::PResult<ArchOp> {
        match roundtrip(
            &self.engine,
            ArchRequest::Step {
                feedback: feedback.map(|b| b.to_vec()),
            },
        ) {
            ArchResponse::Step { op } => op,
            other => Err(PluginError::new(0, format!("unexpected ArchResponse variant: {other:?}"))),
        }
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, ArchRequest::DeclareParams) {
            ArchResponse::DeclareParams { params } => params,
            other => panic!("unexpected ArchResponse variant: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {
        // No-op: dylib mode binds imports at `speet_plugin_create_<role>`
        // time (the `HostImportsFfi` passed into `DylibPlugin::load`), not
        // via a separate post-load call — see `crate::dylib::ffi`.
    }
}

// ── AddressMapper ────────────────────────────────────────────────────────

pub struct DylibAddressMapperPlugin {
    engine: DylibPlugin,
}
impl DylibAddressMapperPlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl AddressMapperPlugin for DylibAddressMapperPlugin {
    fn translate(&self, addr_local: u32) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, AddressMapperRequest::Translate { addr_local }) {
            AddressMapperResponse::Translate { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, AddressMapperRequest::DeclareParams) {
            AddressMapperResponse::DeclareParams { params } => params,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn declare_locals(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, AddressMapperRequest::DeclareLocals) {
            AddressMapperResponse::DeclareLocals { locals } => locals,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn bind_slots(&self, param_base: u32, local_base: u32) {
        let resp: AddressMapperResponse = roundtrip(
            &self.engine,
            AddressMapperRequest::BindSlots {
                param_base,
                local_base,
            },
        );
        assert!(matches!(resp, AddressMapperResponse::BindSlots));
    }
    fn chunk_size(&self) -> Option<u64> {
        match roundtrip(&self.engine, AddressMapperRequest::ChunkSize) {
            AddressMapperResponse::ChunkSize { size } => size,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

// ── MemoryAccess ─────────────────────────────────────────────────────────

pub struct DylibMemoryAccessPlugin {
    engine: DylibPlugin,
}
impl DylibMemoryAccessPlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl MemoryAccessPlugin for DylibMemoryAccessPlugin {
    fn emit_load(&self, addr_local: u32, kind: PluginLoadKind) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, MemoryAccessRequest::EmitLoad { addr_local, kind }) {
            MemoryAccessResponse::EmitLoad { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn emit_store_addr(&self, addr_local: u32) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, MemoryAccessRequest::EmitStoreAddr { addr_local }) {
            MemoryAccessResponse::EmitStoreAddr { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn emit_store_insn(
        &self,
        phys_addr_local: u32,
        kind: PluginStoreKind,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(
            &self.engine,
            MemoryAccessRequest::EmitStoreInsn {
                phys_addr_local,
                kind,
            },
        ) {
            MemoryAccessResponse::EmitStoreInsn { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn emit_memory_size(&self) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, MemoryAccessRequest::EmitMemorySize) {
            MemoryAccessResponse::EmitMemorySize { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn emit_memory_grow(&self) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, MemoryAccessRequest::EmitMemoryGrow) {
            MemoryAccessResponse::EmitMemoryGrow { snippet } => snippet,
            other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
        }
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, MemoryAccessRequest::DeclareParams) {
            MemoryAccessResponse::DeclareParams { params } => params,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn declare_locals(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, MemoryAccessRequest::DeclareLocals) {
            MemoryAccessResponse::DeclareLocals { locals } => locals,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn bind_slots(&self, param_base: u32, local_base: u32) {
        let resp: MemoryAccessResponse = roundtrip(
            &self.engine,
            MemoryAccessRequest::BindSlots {
                param_base,
                local_base,
            },
        );
        assert!(matches!(resp, MemoryAccessResponse::BindSlots));
    }
    fn chunk_size(&self) -> Option<u64> {
        match roundtrip(&self.engine, MemoryAccessRequest::ChunkSize) {
            MemoryAccessResponse::ChunkSize { size } => size,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

// ── Table ────────────────────────────────────────────────────────────────

pub struct DylibTablePlugin {
    engine: DylibPlugin,
}
impl DylibTablePlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl TablePlugin for DylibTablePlugin {
    fn indirect_jump(
        &self,
        target_local: u32,
    ) -> speet_plugin_api::PResult<(PluginIndirectJumpKind, CodeSnippet)> {
        match roundtrip(&self.engine, TableRequest::IndirectJump { target_local }) {
            TableResponse::IndirectJump { result } => result,
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

// ── ObjectModel ──────────────────────────────────────────────────────────

pub struct DylibObjectModelPlugin {
    engine: DylibPlugin,
}
impl DylibObjectModelPlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl ObjectModelPlugin for DylibObjectModelPlugin {
    fn ref_val_type(&self) -> PluginValType {
        match roundtrip(&self.engine, ObjectModelRequest::RefValType) {
            ObjectModelResponse::RefValType { ty } => ty,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn emit_new_object(&self, hash: PluginTypeHash, data_size: u32) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(&self.engine, ObjectModelRequest::EmitNewObject { hash, data_size })
    }
    fn emit_new_array(
        &self,
        length_local: u32,
        elem_hash: PluginTypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitNewArray {
                length_local,
                elem_hash,
                dim,
                elem_bytes,
            },
        )
    }
    fn emit_iget(
        &self,
        ref_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitIGet {
                ref_local,
                byte_offset,
                ty,
            },
        )
    }
    fn emit_iput(
        &self,
        ref_local: u32,
        value_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitIPut {
                ref_local,
                value_local,
                byte_offset,
                ty,
            },
        )
    }
    fn emit_aget(
        &self,
        ref_local: u32,
        index_local: u32,
        ty: PluginFieldValType,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitAGet {
                ref_local,
                index_local,
                ty,
            },
        )
    }
    fn emit_aput(
        &self,
        ref_local: u32,
        index_local: u32,
        value_local: u32,
        ty: PluginFieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitAPut {
                ref_local,
                index_local,
                value_local,
                ty,
                scratch_i32,
                scratch_i64,
            },
        )
    }
    fn emit_array_length(&self, ref_local: u32) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(&self.engine, ObjectModelRequest::EmitArrayLength { ref_local })
    }
    fn emit_instanceof(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitInstanceof {
                ref_local,
                hash,
                dim,
                scratch,
            },
        )
    }
    fn emit_check_cast(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
        snippet_call(
            &self.engine,
            ObjectModelRequest::EmitCheckCast {
                ref_local,
                hash,
                dim,
                scratch,
            },
        )
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

fn snippet_call(engine: &DylibPlugin, req: ObjectModelRequest) -> speet_plugin_api::PResult<CodeSnippet> {
    match roundtrip(engine, req) {
        ObjectModelResponse::Snippet { snippet } => snippet,
        other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
    }
}

// ── Target ───────────────────────────────────────────────────────────────

pub struct DylibTargetPlugin {
    engine: DylibPlugin,
}
impl DylibTargetPlugin {
    pub fn new(engine: DylibPlugin) -> Self {
        Self { engine }
    }
}
impl TargetPlugin for DylibTargetPlugin {
    fn module_manifest(&self) -> ModuleManifest {
        match roundtrip(&self.engine, TargetRequest::ModuleManifest) {
            TargetResponse::ModuleManifest { manifest } => manifest,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn syscall_table(&self) -> PluginSyscallTable {
        match roundtrip(&self.engine, TargetRequest::SyscallTable) {
            TargetResponse::SyscallTable { table } => table,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}
