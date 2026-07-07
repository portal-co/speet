//! The six `Subprocess*Plugin` adapters: each implements one `speet-plugin-api`
//! trait by encoding a `remote::XRequest`, driving it through
//! [`SubprocessPlugin::call_raw`], and decoding the `remote::XResponse`.
//!
//! Methods that the in-process traits declare as infallible (returning
//! plain data, not `PResult`) have no error channel to report a transport
//! failure (a process crash, a closed pipe, a malformed response) through.
//! Such a failure means the plugin process is fundamentally broken, so
//! these adapters `expect`/`panic!` rather than inventing a fake zero value
//! — consistent with
//! `speet_link_core::Recompile::count_fns`'s own default impl, which
//! panics rather than guessing. Fallible methods (`PResult`-returning)
//! convert a transport failure into a normal `Err(PluginError)` instead.

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
use speet_plugin_api::{AddressMapperPlugin, ArchPlugin, MemoryAccessPlugin, ObjectModelPlugin, TablePlugin, TargetPlugin};

use crate::process::SubprocessPlugin;

fn roundtrip<Req: WireEncode, Resp: WireDecode>(engine: &SubprocessPlugin, req: Req) -> Resp {
    let mut payload = Vec::new();
    req.encode(&mut payload);
    let resp_bytes = engine
        .call_raw(&payload)
        .expect("subprocess plugin call failed (transport-level)");
    let (resp, _) = Resp::decode(&resp_bytes).expect("subprocess plugin returned malformed response");
    resp
}

// ── Arch ─────────────────────────────────────────────────────────────────

pub struct SubprocessArchPlugin {
    engine: SubprocessPlugin,
}
impl SubprocessArchPlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl ArchPlugin for SubprocessArchPlugin {
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
            other => Err(PluginError::new(
                0,
                format!("unexpected ArchResponse variant: {other:?}"),
            )),
        }
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, ArchRequest::DeclareParams) {
            ArchResponse::DeclareParams { params } => params,
            other => panic!("unexpected ArchResponse variant: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {
        // No-op: the subprocess realization of §2.7 resolves each import
        // per-call, against the `Arc<dyn HostImports>` already captured by
        // `SubprocessPlugin::spawn`'s `ImportRequest` dispatch loop — there
        // is no separate post-load binding step to perform here. See
        // `crate::frame` and `crate::process` module docs.
    }
}

// ── AddressMapper ────────────────────────────────────────────────────────

pub struct SubprocessAddressMapperPlugin {
    engine: SubprocessPlugin,
}
impl SubprocessAddressMapperPlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl AddressMapperPlugin for SubprocessAddressMapperPlugin {
    fn translate(&self, addr_local: u32) -> speet_plugin_api::PResult<CodeSnippet> {
        match roundtrip(&self.engine, AddressMapperRequest::Translate { addr_local }) {
            AddressMapperResponse::Translate { snippet } => snippet,
            other => Err(PluginError::new(
                0,
                format!("unexpected AddressMapperResponse variant: {other:?}"),
            )),
        }
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, AddressMapperRequest::DeclareParams) {
            AddressMapperResponse::DeclareParams { params } => params,
            other => panic!("unexpected AddressMapperResponse variant: {other:?}"),
        }
    }
    fn declare_locals(&self) -> Vec<PluginValType> {
        match roundtrip(&self.engine, AddressMapperRequest::DeclareLocals) {
            AddressMapperResponse::DeclareLocals { locals } => locals,
            other => panic!("unexpected AddressMapperResponse variant: {other:?}"),
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
            other => panic!("unexpected AddressMapperResponse variant: {other:?}"),
        }
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

// ── MemoryAccess ─────────────────────────────────────────────────────────

pub struct SubprocessMemoryAccessPlugin {
    engine: SubprocessPlugin,
}
impl SubprocessMemoryAccessPlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl MemoryAccessPlugin for SubprocessMemoryAccessPlugin {
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

pub struct SubprocessTablePlugin {
    engine: SubprocessPlugin,
}
impl SubprocessTablePlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl TablePlugin for SubprocessTablePlugin {
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

pub struct SubprocessObjectModelPlugin {
    engine: SubprocessPlugin,
}
impl SubprocessObjectModelPlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl ObjectModelPlugin for SubprocessObjectModelPlugin {
    fn ref_val_type(&self) -> PluginValType {
        match roundtrip(&self.engine, ObjectModelRequest::RefValType) {
            ObjectModelResponse::RefValType { ty } => ty,
            other => panic!("unexpected response: {other:?}"),
        }
    }
    fn emit_new_object(
        &self,
        hash: PluginTypeHash,
        data_size: u32,
    ) -> speet_plugin_api::PResult<CodeSnippet> {
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

fn snippet_call(engine: &SubprocessPlugin, req: ObjectModelRequest) -> speet_plugin_api::PResult<CodeSnippet> {
    match roundtrip(engine, req) {
        ObjectModelResponse::Snippet { snippet } => snippet,
        other => Err(PluginError::new(0, format!("unexpected response: {other:?}"))),
    }
}

// ── Target ───────────────────────────────────────────────────────────────

pub struct SubprocessTargetPlugin {
    engine: SubprocessPlugin,
}
impl SubprocessTargetPlugin {
    pub fn new(engine: SubprocessPlugin) -> Self {
        Self { engine }
    }
}
impl TargetPlugin for SubprocessTargetPlugin {
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
