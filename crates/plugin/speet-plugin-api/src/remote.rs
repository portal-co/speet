//! Shared request/response wire envelopes for the WASM and subprocess
//! transports — both marshal calls across a real process/sandbox boundary,
//! so both need the exact same "encode these args, decode this result"
//! logic. Defining it once here means neither transport crate (nor a
//! non-Rust plugin author implementing the guest/subprocess side by hand)
//! has to reverse-engineer the other's wire shape — this module *is* the
//! spec. See `docs/guides/plugin-api.md` §6.
//!
//! Each `XRequest`/`XResponse` pair mirrors `XPlugin` 1:1, one variant per
//! trait method (`bind_imports` excluded — it has no remote-call form, see
//! below). The request variant's wire tag reuses the matching `XMethod`
//! enum's discriminant, so "method tag 2" means the same thing whether
//! you're reading the in-process dispatch docs or sniffing wire bytes.
//!
//! **`bind_imports` has no `XRequest` variant.** For the in-process host
//! it's a direct `Arc` handoff (no wire call at all). For WASM, the import
//! mechanism is realized as ordinary WASM host-function imports resolved at
//! *instantiation* time — by the time a guest can call anything, its
//! imports are already linked. For the subprocess host, there is similarly
//! no separate "bind" step: the plugin issues a nested `plugin→host` frame
//! (see `speet-plugin-host-subprocess`) whenever it wants to use a granted
//! import, addressed by `(kind, name)` every time rather than through a
//! prior handle. `bind_imports` stays a pure in-process convenience.

use alloc::vec::Vec;

use crate::arch::{ArchOp, ArchPlugin};
use crate::error::PResult;
use crate::memory::{AddressMapperPlugin, MemoryAccessPlugin, PluginLoadKind, PluginStoreKind};
use crate::object_model::{ObjectModelPlugin, PluginFieldValType, PluginTypeHash};
use crate::snippet::{CodeSnippet, PluginValType};
use crate::table::{PluginIndirectJumpKind, TablePlugin};
use crate::target::{ModuleManifest, PluginSyscallTable, TargetPlugin};
use crate::wire::{WireDecode, WireEncode, WireError};

// ── Arch ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ArchRequest {
    ResetForNextBinary { args: Vec<u8> },
    CountFns { bytes: Vec<u8> },
    Step { feedback: Option<Vec<u8>> },
    DeclareParams,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArchResponse {
    ResetForNextBinary,
    CountFns { count: u32 },
    Step { op: PResult<ArchOp> },
    DeclareParams { params: Vec<PluginValType> },
}

impl WireEncode for ArchRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ArchRequest::ResetForNextBinary { args } => {
                0u8.encode(out);
                args.encode(out);
            }
            ArchRequest::CountFns { bytes } => {
                1u8.encode(out);
                bytes.encode(out);
            }
            ArchRequest::Step { feedback } => {
                2u8.encode(out);
                feedback.encode(out);
            }
            ArchRequest::DeclareParams => 3u8.encode(out),
        }
    }
}
impl WireDecode for ArchRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (args, rest) = Vec::<u8>::decode(rest)?;
                (ArchRequest::ResetForNextBinary { args }, rest)
            }
            1 => {
                let (bytes, rest) = Vec::<u8>::decode(rest)?;
                (ArchRequest::CountFns { bytes }, rest)
            }
            2 => {
                let (feedback, rest) = Option::<Vec<u8>>::decode(rest)?;
                (ArchRequest::Step { feedback }, rest)
            }
            3 => (ArchRequest::DeclareParams, rest),
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for ArchResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ArchResponse::ResetForNextBinary => 0u8.encode(out),
            ArchResponse::CountFns { count } => {
                1u8.encode(out);
                count.encode(out);
            }
            ArchResponse::Step { op } => {
                2u8.encode(out);
                op.encode(out);
            }
            ArchResponse::DeclareParams { params } => {
                3u8.encode(out);
                params.encode(out);
            }
        }
    }
}
impl WireDecode for ArchResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => (ArchResponse::ResetForNextBinary, rest),
            1 => {
                let (count, rest) = u32::decode(rest)?;
                (ArchResponse::CountFns { count }, rest)
            }
            2 => {
                let (op, rest) = PResult::<ArchOp>::decode(rest)?;
                (ArchResponse::Step { op }, rest)
            }
            3 => {
                let (params, rest) = Vec::<PluginValType>::decode(rest)?;
                (ArchResponse::DeclareParams { params }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

/// Decode `req`, call the matching method on `plugin`, encode the result.
/// Shared by every remote transport so the "match tag, call trait method"
/// logic lives in exactly one place.
pub fn dispatch_arch(plugin: &dyn ArchPlugin, req: ArchRequest) -> ArchResponse {
    match req {
        ArchRequest::ResetForNextBinary { args } => {
            plugin.reset_for_next_binary(&args);
            ArchResponse::ResetForNextBinary
        }
        ArchRequest::CountFns { bytes } => ArchResponse::CountFns {
            count: plugin.count_fns(&bytes),
        },
        ArchRequest::Step { feedback } => ArchResponse::Step {
            op: plugin.step(feedback.as_deref()),
        },
        ArchRequest::DeclareParams => ArchResponse::DeclareParams {
            params: plugin.declare_params(),
        },
    }
}

// ── AddressMapper ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AddressMapperRequest {
    Translate { addr_local: u32 },
    DeclareParams,
    DeclareLocals,
    BindSlots { param_base: u32, local_base: u32 },
    ChunkSize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AddressMapperResponse {
    Translate { snippet: PResult<CodeSnippet> },
    DeclareParams { params: Vec<PluginValType> },
    DeclareLocals { locals: Vec<PluginValType> },
    BindSlots,
    ChunkSize { size: Option<u64> },
}

impl WireEncode for AddressMapperRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            AddressMapperRequest::Translate { addr_local } => {
                0u8.encode(out);
                addr_local.encode(out);
            }
            AddressMapperRequest::DeclareParams => 1u8.encode(out),
            AddressMapperRequest::DeclareLocals => 2u8.encode(out),
            AddressMapperRequest::BindSlots {
                param_base,
                local_base,
            } => {
                3u8.encode(out);
                param_base.encode(out);
                local_base.encode(out);
            }
            AddressMapperRequest::ChunkSize => 4u8.encode(out),
        }
    }
}
impl WireDecode for AddressMapperRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (addr_local, rest) = u32::decode(rest)?;
                (AddressMapperRequest::Translate { addr_local }, rest)
            }
            1 => (AddressMapperRequest::DeclareParams, rest),
            2 => (AddressMapperRequest::DeclareLocals, rest),
            3 => {
                let (param_base, rest) = u32::decode(rest)?;
                let (local_base, rest) = u32::decode(rest)?;
                (
                    AddressMapperRequest::BindSlots {
                        param_base,
                        local_base,
                    },
                    rest,
                )
            }
            4 => (AddressMapperRequest::ChunkSize, rest),
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for AddressMapperResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            AddressMapperResponse::Translate { snippet } => {
                0u8.encode(out);
                snippet.encode(out);
            }
            AddressMapperResponse::DeclareParams { params } => {
                1u8.encode(out);
                params.encode(out);
            }
            AddressMapperResponse::DeclareLocals { locals } => {
                2u8.encode(out);
                locals.encode(out);
            }
            AddressMapperResponse::BindSlots => 3u8.encode(out),
            AddressMapperResponse::ChunkSize { size } => {
                4u8.encode(out);
                size.encode(out);
            }
        }
    }
}
impl WireDecode for AddressMapperResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (AddressMapperResponse::Translate { snippet }, rest)
            }
            1 => {
                let (params, rest) = Vec::<PluginValType>::decode(rest)?;
                (AddressMapperResponse::DeclareParams { params }, rest)
            }
            2 => {
                let (locals, rest) = Vec::<PluginValType>::decode(rest)?;
                (AddressMapperResponse::DeclareLocals { locals }, rest)
            }
            3 => (AddressMapperResponse::BindSlots, rest),
            4 => {
                let (size, rest) = Option::<u64>::decode(rest)?;
                (AddressMapperResponse::ChunkSize { size }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

pub fn dispatch_address_mapper(
    plugin: &dyn AddressMapperPlugin,
    req: AddressMapperRequest,
) -> AddressMapperResponse {
    match req {
        AddressMapperRequest::Translate { addr_local } => AddressMapperResponse::Translate {
            snippet: plugin.translate(addr_local),
        },
        AddressMapperRequest::DeclareParams => AddressMapperResponse::DeclareParams {
            params: plugin.declare_params(),
        },
        AddressMapperRequest::DeclareLocals => AddressMapperResponse::DeclareLocals {
            locals: plugin.declare_locals(),
        },
        AddressMapperRequest::BindSlots {
            param_base,
            local_base,
        } => {
            plugin.bind_slots(param_base, local_base);
            AddressMapperResponse::BindSlots
        }
        AddressMapperRequest::ChunkSize => AddressMapperResponse::ChunkSize {
            size: plugin.chunk_size(),
        },
    }
}

// ── MemoryAccess ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessRequest {
    EmitLoad {
        addr_local: u32,
        kind: PluginLoadKind,
    },
    EmitStoreAddr {
        addr_local: u32,
    },
    EmitStoreInsn {
        phys_addr_local: u32,
        kind: PluginStoreKind,
    },
    EmitMemorySize,
    EmitMemoryGrow,
    DeclareParams,
    DeclareLocals,
    BindSlots {
        param_base: u32,
        local_base: u32,
    },
    ChunkSize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessResponse {
    EmitLoad { snippet: PResult<CodeSnippet> },
    EmitStoreAddr { snippet: PResult<CodeSnippet> },
    EmitStoreInsn { snippet: PResult<CodeSnippet> },
    EmitMemorySize { snippet: PResult<CodeSnippet> },
    EmitMemoryGrow { snippet: PResult<CodeSnippet> },
    DeclareParams { params: Vec<PluginValType> },
    DeclareLocals { locals: Vec<PluginValType> },
    BindSlots,
    ChunkSize { size: Option<u64> },
}

impl WireEncode for MemoryAccessRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            MemoryAccessRequest::EmitLoad { addr_local, kind } => {
                0u8.encode(out);
                addr_local.encode(out);
                kind.encode(out);
            }
            MemoryAccessRequest::EmitStoreAddr { addr_local } => {
                1u8.encode(out);
                addr_local.encode(out);
            }
            MemoryAccessRequest::EmitStoreInsn {
                phys_addr_local,
                kind,
            } => {
                2u8.encode(out);
                phys_addr_local.encode(out);
                kind.encode(out);
            }
            MemoryAccessRequest::EmitMemorySize => 3u8.encode(out),
            MemoryAccessRequest::EmitMemoryGrow => 4u8.encode(out),
            MemoryAccessRequest::DeclareParams => 5u8.encode(out),
            MemoryAccessRequest::DeclareLocals => 6u8.encode(out),
            MemoryAccessRequest::BindSlots {
                param_base,
                local_base,
            } => {
                7u8.encode(out);
                param_base.encode(out);
                local_base.encode(out);
            }
            MemoryAccessRequest::ChunkSize => 8u8.encode(out),
        }
    }
}
impl WireDecode for MemoryAccessRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (addr_local, rest) = u32::decode(rest)?;
                let (kind, rest) = PluginLoadKind::decode(rest)?;
                (MemoryAccessRequest::EmitLoad { addr_local, kind }, rest)
            }
            1 => {
                let (addr_local, rest) = u32::decode(rest)?;
                (MemoryAccessRequest::EmitStoreAddr { addr_local }, rest)
            }
            2 => {
                let (phys_addr_local, rest) = u32::decode(rest)?;
                let (kind, rest) = PluginStoreKind::decode(rest)?;
                (
                    MemoryAccessRequest::EmitStoreInsn {
                        phys_addr_local,
                        kind,
                    },
                    rest,
                )
            }
            3 => (MemoryAccessRequest::EmitMemorySize, rest),
            4 => (MemoryAccessRequest::EmitMemoryGrow, rest),
            5 => (MemoryAccessRequest::DeclareParams, rest),
            6 => (MemoryAccessRequest::DeclareLocals, rest),
            7 => {
                let (param_base, rest) = u32::decode(rest)?;
                let (local_base, rest) = u32::decode(rest)?;
                (
                    MemoryAccessRequest::BindSlots {
                        param_base,
                        local_base,
                    },
                    rest,
                )
            }
            8 => (MemoryAccessRequest::ChunkSize, rest),
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for MemoryAccessResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            MemoryAccessResponse::EmitLoad { snippet } => {
                0u8.encode(out);
                snippet.encode(out);
            }
            MemoryAccessResponse::EmitStoreAddr { snippet } => {
                1u8.encode(out);
                snippet.encode(out);
            }
            MemoryAccessResponse::EmitStoreInsn { snippet } => {
                2u8.encode(out);
                snippet.encode(out);
            }
            MemoryAccessResponse::EmitMemorySize { snippet } => {
                3u8.encode(out);
                snippet.encode(out);
            }
            MemoryAccessResponse::EmitMemoryGrow { snippet } => {
                4u8.encode(out);
                snippet.encode(out);
            }
            MemoryAccessResponse::DeclareParams { params } => {
                5u8.encode(out);
                params.encode(out);
            }
            MemoryAccessResponse::DeclareLocals { locals } => {
                6u8.encode(out);
                locals.encode(out);
            }
            MemoryAccessResponse::BindSlots => 7u8.encode(out),
            MemoryAccessResponse::ChunkSize { size } => {
                8u8.encode(out);
                size.encode(out);
            }
        }
    }
}
impl WireDecode for MemoryAccessResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (MemoryAccessResponse::EmitLoad { snippet }, rest)
            }
            1 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (MemoryAccessResponse::EmitStoreAddr { snippet }, rest)
            }
            2 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (MemoryAccessResponse::EmitStoreInsn { snippet }, rest)
            }
            3 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (MemoryAccessResponse::EmitMemorySize { snippet }, rest)
            }
            4 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (MemoryAccessResponse::EmitMemoryGrow { snippet }, rest)
            }
            5 => {
                let (params, rest) = Vec::<PluginValType>::decode(rest)?;
                (MemoryAccessResponse::DeclareParams { params }, rest)
            }
            6 => {
                let (locals, rest) = Vec::<PluginValType>::decode(rest)?;
                (MemoryAccessResponse::DeclareLocals { locals }, rest)
            }
            7 => (MemoryAccessResponse::BindSlots, rest),
            8 => {
                let (size, rest) = Option::<u64>::decode(rest)?;
                (MemoryAccessResponse::ChunkSize { size }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

pub fn dispatch_memory_access(
    plugin: &dyn MemoryAccessPlugin,
    req: MemoryAccessRequest,
) -> MemoryAccessResponse {
    match req {
        MemoryAccessRequest::EmitLoad { addr_local, kind } => MemoryAccessResponse::EmitLoad {
            snippet: plugin.emit_load(addr_local, kind),
        },
        MemoryAccessRequest::EmitStoreAddr { addr_local } => {
            MemoryAccessResponse::EmitStoreAddr {
                snippet: plugin.emit_store_addr(addr_local),
            }
        }
        MemoryAccessRequest::EmitStoreInsn {
            phys_addr_local,
            kind,
        } => MemoryAccessResponse::EmitStoreInsn {
            snippet: plugin.emit_store_insn(phys_addr_local, kind),
        },
        MemoryAccessRequest::EmitMemorySize => MemoryAccessResponse::EmitMemorySize {
            snippet: plugin.emit_memory_size(),
        },
        MemoryAccessRequest::EmitMemoryGrow => MemoryAccessResponse::EmitMemoryGrow {
            snippet: plugin.emit_memory_grow(),
        },
        MemoryAccessRequest::DeclareParams => MemoryAccessResponse::DeclareParams {
            params: plugin.declare_params(),
        },
        MemoryAccessRequest::DeclareLocals => MemoryAccessResponse::DeclareLocals {
            locals: plugin.declare_locals(),
        },
        MemoryAccessRequest::BindSlots {
            param_base,
            local_base,
        } => {
            plugin.bind_slots(param_base, local_base);
            MemoryAccessResponse::BindSlots
        }
        MemoryAccessRequest::ChunkSize => MemoryAccessResponse::ChunkSize {
            size: plugin.chunk_size(),
        },
    }
}

// ── Table ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TableRequest {
    IndirectJump { target_local: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableResponse {
    IndirectJump {
        result: PResult<(PluginIndirectJumpKind, CodeSnippet)>,
    },
}

impl WireEncode for TableRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            TableRequest::IndirectJump { target_local } => {
                0u8.encode(out);
                target_local.encode(out);
            }
        }
    }
}
impl WireDecode for TableRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (target_local, rest) = u32::decode(rest)?;
                (TableRequest::IndirectJump { target_local }, rest)
            }
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for TableResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            TableResponse::IndirectJump { result } => {
                0u8.encode(out);
                result.encode(out);
            }
        }
    }
}
impl WireDecode for TableResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (result, rest) =
                    PResult::<(PluginIndirectJumpKind, CodeSnippet)>::decode(rest)?;
                (TableResponse::IndirectJump { result }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

pub fn dispatch_table(plugin: &dyn TablePlugin, req: TableRequest) -> TableResponse {
    match req {
        TableRequest::IndirectJump { target_local } => TableResponse::IndirectJump {
            result: plugin.indirect_jump(target_local),
        },
    }
}

// ── ObjectModel ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectModelRequest {
    RefValType,
    EmitNewObject {
        hash: PluginTypeHash,
        data_size: u32,
    },
    EmitNewArray {
        length_local: u32,
        elem_hash: PluginTypeHash,
        dim: u32,
        elem_bytes: u32,
    },
    EmitIGet {
        ref_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    },
    EmitIPut {
        ref_local: u32,
        value_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    },
    EmitAGet {
        ref_local: u32,
        index_local: u32,
        ty: PluginFieldValType,
    },
    EmitAPut {
        ref_local: u32,
        index_local: u32,
        value_local: u32,
        ty: PluginFieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    },
    EmitArrayLength {
        ref_local: u32,
    },
    EmitInstanceof {
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    },
    EmitCheckCast {
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectModelResponse {
    RefValType { ty: PluginValType },
    Snippet { snippet: PResult<CodeSnippet> },
}

impl WireEncode for ObjectModelRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ObjectModelRequest::RefValType => 0u8.encode(out),
            ObjectModelRequest::EmitNewObject { hash, data_size } => {
                1u8.encode(out);
                hash.encode(out);
                data_size.encode(out);
            }
            ObjectModelRequest::EmitNewArray {
                length_local,
                elem_hash,
                dim,
                elem_bytes,
            } => {
                2u8.encode(out);
                length_local.encode(out);
                elem_hash.encode(out);
                dim.encode(out);
                elem_bytes.encode(out);
            }
            ObjectModelRequest::EmitIGet {
                ref_local,
                byte_offset,
                ty,
            } => {
                3u8.encode(out);
                ref_local.encode(out);
                byte_offset.encode(out);
                ty.encode(out);
            }
            ObjectModelRequest::EmitIPut {
                ref_local,
                value_local,
                byte_offset,
                ty,
            } => {
                4u8.encode(out);
                ref_local.encode(out);
                value_local.encode(out);
                byte_offset.encode(out);
                ty.encode(out);
            }
            ObjectModelRequest::EmitAGet {
                ref_local,
                index_local,
                ty,
            } => {
                5u8.encode(out);
                ref_local.encode(out);
                index_local.encode(out);
                ty.encode(out);
            }
            ObjectModelRequest::EmitAPut {
                ref_local,
                index_local,
                value_local,
                ty,
                scratch_i32,
                scratch_i64,
            } => {
                6u8.encode(out);
                ref_local.encode(out);
                index_local.encode(out);
                value_local.encode(out);
                ty.encode(out);
                scratch_i32.encode(out);
                scratch_i64.encode(out);
            }
            ObjectModelRequest::EmitArrayLength { ref_local } => {
                7u8.encode(out);
                ref_local.encode(out);
            }
            ObjectModelRequest::EmitInstanceof {
                ref_local,
                hash,
                dim,
                scratch,
            } => {
                8u8.encode(out);
                ref_local.encode(out);
                hash.encode(out);
                dim.encode(out);
                scratch.encode(out);
            }
            ObjectModelRequest::EmitCheckCast {
                ref_local,
                hash,
                dim,
                scratch,
            } => {
                9u8.encode(out);
                ref_local.encode(out);
                hash.encode(out);
                dim.encode(out);
                scratch.encode(out);
            }
        }
    }
}
impl WireDecode for ObjectModelRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => (ObjectModelRequest::RefValType, rest),
            1 => {
                let (hash, rest) = PluginTypeHash::decode(rest)?;
                let (data_size, rest) = u32::decode(rest)?;
                (ObjectModelRequest::EmitNewObject { hash, data_size }, rest)
            }
            2 => {
                let (length_local, rest) = u32::decode(rest)?;
                let (elem_hash, rest) = PluginTypeHash::decode(rest)?;
                let (dim, rest) = u32::decode(rest)?;
                let (elem_bytes, rest) = u32::decode(rest)?;
                (
                    ObjectModelRequest::EmitNewArray {
                        length_local,
                        elem_hash,
                        dim,
                        elem_bytes,
                    },
                    rest,
                )
            }
            3 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (byte_offset, rest) = u32::decode(rest)?;
                let (ty, rest) = PluginFieldValType::decode(rest)?;
                (
                    ObjectModelRequest::EmitIGet {
                        ref_local,
                        byte_offset,
                        ty,
                    },
                    rest,
                )
            }
            4 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (value_local, rest) = u32::decode(rest)?;
                let (byte_offset, rest) = u32::decode(rest)?;
                let (ty, rest) = PluginFieldValType::decode(rest)?;
                (
                    ObjectModelRequest::EmitIPut {
                        ref_local,
                        value_local,
                        byte_offset,
                        ty,
                    },
                    rest,
                )
            }
            5 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (index_local, rest) = u32::decode(rest)?;
                let (ty, rest) = PluginFieldValType::decode(rest)?;
                (
                    ObjectModelRequest::EmitAGet {
                        ref_local,
                        index_local,
                        ty,
                    },
                    rest,
                )
            }
            6 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (index_local, rest) = u32::decode(rest)?;
                let (value_local, rest) = u32::decode(rest)?;
                let (ty, rest) = PluginFieldValType::decode(rest)?;
                let (scratch_i32, rest) = u32::decode(rest)?;
                let (scratch_i64, rest) = u32::decode(rest)?;
                (
                    ObjectModelRequest::EmitAPut {
                        ref_local,
                        index_local,
                        value_local,
                        ty,
                        scratch_i32,
                        scratch_i64,
                    },
                    rest,
                )
            }
            7 => {
                let (ref_local, rest) = u32::decode(rest)?;
                (ObjectModelRequest::EmitArrayLength { ref_local }, rest)
            }
            8 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (hash, rest) = PluginTypeHash::decode(rest)?;
                let (dim, rest) = u32::decode(rest)?;
                let (scratch, rest) = u32::decode(rest)?;
                (
                    ObjectModelRequest::EmitInstanceof {
                        ref_local,
                        hash,
                        dim,
                        scratch,
                    },
                    rest,
                )
            }
            9 => {
                let (ref_local, rest) = u32::decode(rest)?;
                let (hash, rest) = PluginTypeHash::decode(rest)?;
                let (dim, rest) = u32::decode(rest)?;
                let (scratch, rest) = u32::decode(rest)?;
                (
                    ObjectModelRequest::EmitCheckCast {
                        ref_local,
                        hash,
                        dim,
                        scratch,
                    },
                    rest,
                )
            }
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for ObjectModelResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ObjectModelResponse::RefValType { ty } => {
                0u8.encode(out);
                ty.encode(out);
            }
            ObjectModelResponse::Snippet { snippet } => {
                1u8.encode(out);
                snippet.encode(out);
            }
        }
    }
}
impl WireDecode for ObjectModelResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (ty, rest) = PluginValType::decode(rest)?;
                (ObjectModelResponse::RefValType { ty }, rest)
            }
            1 => {
                let (snippet, rest) = PResult::<CodeSnippet>::decode(rest)?;
                (ObjectModelResponse::Snippet { snippet }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

pub fn dispatch_object_model(
    plugin: &dyn ObjectModelPlugin,
    req: ObjectModelRequest,
) -> ObjectModelResponse {
    match req {
        ObjectModelRequest::RefValType => ObjectModelResponse::RefValType {
            ty: plugin.ref_val_type(),
        },
        ObjectModelRequest::EmitNewObject { hash, data_size } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_new_object(hash, data_size),
        },
        ObjectModelRequest::EmitNewArray {
            length_local,
            elem_hash,
            dim,
            elem_bytes,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_new_array(length_local, elem_hash, dim, elem_bytes),
        },
        ObjectModelRequest::EmitIGet {
            ref_local,
            byte_offset,
            ty,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_iget(ref_local, byte_offset, ty),
        },
        ObjectModelRequest::EmitIPut {
            ref_local,
            value_local,
            byte_offset,
            ty,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_iput(ref_local, value_local, byte_offset, ty),
        },
        ObjectModelRequest::EmitAGet {
            ref_local,
            index_local,
            ty,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_aget(ref_local, index_local, ty),
        },
        ObjectModelRequest::EmitAPut {
            ref_local,
            index_local,
            value_local,
            ty,
            scratch_i32,
            scratch_i64,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_aput(ref_local, index_local, value_local, ty, scratch_i32, scratch_i64),
        },
        ObjectModelRequest::EmitArrayLength { ref_local } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_array_length(ref_local),
        },
        ObjectModelRequest::EmitInstanceof {
            ref_local,
            hash,
            dim,
            scratch,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_instanceof(ref_local, hash, dim, scratch),
        },
        ObjectModelRequest::EmitCheckCast {
            ref_local,
            hash,
            dim,
            scratch,
        } => ObjectModelResponse::Snippet {
            snippet: plugin.emit_check_cast(ref_local, hash, dim, scratch),
        },
    }
}

// ── Target ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRequest {
    ModuleManifest,
    SyscallTable,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetResponse {
    ModuleManifest { manifest: ModuleManifest },
    SyscallTable { table: PluginSyscallTable },
}

impl WireEncode for TargetRequest {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            TargetRequest::ModuleManifest => 0u8.encode(out),
            TargetRequest::SyscallTable => 1u8.encode(out),
        }
    }
}
impl WireDecode for TargetRequest {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => (TargetRequest::ModuleManifest, rest),
            1 => (TargetRequest::SyscallTable, rest),
            _ => return Err(WireError),
        })
    }
}
impl WireEncode for TargetResponse {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            TargetResponse::ModuleManifest { manifest } => {
                0u8.encode(out);
                manifest.encode(out);
            }
            TargetResponse::SyscallTable { table } => {
                1u8.encode(out);
                table.encode(out);
            }
        }
    }
}
impl WireDecode for TargetResponse {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (manifest, rest) = ModuleManifest::decode(rest)?;
                (TargetResponse::ModuleManifest { manifest }, rest)
            }
            1 => {
                let (table, rest) = PluginSyscallTable::decode(rest)?;
                (TargetResponse::SyscallTable { table }, rest)
            }
            _ => return Err(WireError),
        })
    }
}

pub fn dispatch_target(plugin: &dyn TargetPlugin, req: TargetRequest) -> TargetResponse {
    match req {
        TargetRequest::ModuleManifest => TargetResponse::ModuleManifest {
            manifest: plugin.module_manifest(),
        },
        TargetRequest::SyscallTable => TargetResponse::SyscallTable {
            table: plugin.syscall_table(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn arch_request_roundtrip() {
        let reqs = [
            ArchRequest::ResetForNextBinary {
                args: vec![1, 2, 3],
            },
            ArchRequest::CountFns { bytes: vec![9] },
            ArchRequest::Step {
                feedback: Some(vec![4, 5]),
            },
            ArchRequest::Step { feedback: None },
            ArchRequest::DeclareParams,
        ];
        for req in reqs {
            let mut out = Vec::new();
            req.encode(&mut out);
            let (decoded, rest) = ArchRequest::decode(&out).unwrap();
            assert_eq!(decoded, req);
            assert!(rest.is_empty());
        }
    }

    #[test]
    fn target_roundtrip() {
        let req = TargetRequest::ModuleManifest;
        let mut out = Vec::new();
        req.encode(&mut out);
        let (decoded, rest) = TargetRequest::decode(&out).unwrap();
        assert_eq!(decoded, req);
        assert!(rest.is_empty());

        let resp = TargetResponse::SyscallTable {
            table: PluginSyscallTable::default(),
        };
        let mut out = Vec::new();
        resp.encode(&mut out);
        let (decoded, rest) = TargetResponse::decode(&out).unwrap();
        assert_eq!(decoded, resp);
        assert!(rest.is_empty());
    }

    #[test]
    fn address_mapper_roundtrip() {
        let req = AddressMapperRequest::BindSlots {
            param_base: 3,
            local_base: 9,
        };
        let mut out = Vec::new();
        req.encode(&mut out);
        let (decoded, rest) = AddressMapperRequest::decode(&out).unwrap();
        assert_eq!(decoded, req);
        assert!(rest.is_empty());
    }
}
