//! Object model plugin — mirrors `speet_object::ObjectModel`.

use alloc::vec::Vec;

use crate::error::PResult;
use crate::imports::HostImports;
use crate::snippet::{CodeSnippet, PluginValType};
use crate::wire::{WireDecode, WireEncode, WireError};

/// Mirrors `speet_object::TypeHash` (SHA3-256 of a class name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginTypeHash(pub [u8; 32]);

impl WireEncode for PluginTypeHash {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
}
impl WireDecode for PluginTypeHash {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        if input.len() < 32 {
            return Err(WireError);
        }
        let (head, tail) = input.split_at(32);
        let mut buf = [0u8; 32];
        buf.copy_from_slice(head);
        Ok((PluginTypeHash(buf), tail))
    }
}

/// Mirrors `speet_object::FieldValType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginFieldValType {
    I32,
    I64,
    F32,
    F64,
    I8S,
    I8U,
    I16S,
    I16U,
    Ref,
}

macro_rules! wire_c_enum {
    ($name:ident { $($variant:ident = $tag:literal),+ $(,)? }) => {
        impl WireEncode for $name {
            fn encode(&self, out: &mut Vec<u8>) {
                let tag: u8 = match self { $(Self::$variant => $tag,)+ };
                tag.encode(out);
            }
        }
        impl WireDecode for $name {
            fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
                let (tag, rest) = u8::decode(input)?;
                let v = match tag { $($tag => Self::$variant,)+ _ => return Err(WireError) };
                Ok((v, rest))
            }
        }
    };
}

wire_c_enum!(PluginFieldValType {
    I32 = 0, I64 = 1, F32 = 2, F64 = 3, I8S = 4, I8U = 5, I16S = 6, I16U = 7, Ref = 8,
});

/// Method tags for the `ObjectModelPlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectModelMethod {
    RefValType = 0,
    EmitNewObject = 1,
    EmitNewArray = 2,
    EmitIGet = 3,
    EmitIPut = 4,
    EmitAGet = 5,
    EmitAPut = 6,
    EmitArrayLength = 7,
    EmitInstanceof = 8,
    EmitCheckCast = 9,
}

/// Heap object layout and operations. Mirrors `speet_object::ObjectModel`,
/// `&self` exactly like the internal trait — no signature widening needed
/// (see `memory` module docs for why every plugin trait here is `&self`).
///
/// Every internal `ObjectModel` method documents a stack contract (e.g.
/// `emit_iget`'s "Stack before: `[ref]`"); those stack-only inputs have no
/// named parameter in the *internal* trait, since the internal trait's
/// caller (the adapter) controls the wasm value stack directly. A plugin
/// has no such access, so each stack input gets an explicit `_local`
/// parameter here — the adapter stages the stack value into that local
/// (one of a fixed set of scratch locals supplied when the adapter is
/// constructed, since `ObjectModel` is not a `LocalDeclarator`) before
/// calling the plugin. See `speet-plugin-adapter::object_model` and
/// `docs/guides/plugin-api.md`.
pub trait ObjectModelPlugin: Send + Sync {
    fn ref_val_type(&self) -> PluginValType;
    fn emit_new_object(&self, hash: PluginTypeHash, data_size: u32) -> PResult<CodeSnippet>;
    fn emit_new_array(
        &self,
        length_local: u32,
        elem_hash: PluginTypeHash,
        dim: u32,
        elem_bytes: u32,
    ) -> PResult<CodeSnippet>;
    fn emit_iget(&self, ref_local: u32, byte_offset: u32, ty: PluginFieldValType) -> PResult<CodeSnippet>;
    fn emit_iput(
        &self,
        ref_local: u32,
        value_local: u32,
        byte_offset: u32,
        ty: PluginFieldValType,
    ) -> PResult<CodeSnippet>;
    fn emit_aget(&self, ref_local: u32, index_local: u32, ty: PluginFieldValType) -> PResult<CodeSnippet>;
    fn emit_aput(
        &self,
        ref_local: u32,
        index_local: u32,
        value_local: u32,
        ty: PluginFieldValType,
        scratch_i32: u32,
        scratch_i64: u32,
    ) -> PResult<CodeSnippet>;
    fn emit_array_length(&self, ref_local: u32) -> PResult<CodeSnippet>;
    fn emit_instanceof(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> PResult<CodeSnippet>;
    fn emit_check_cast(
        &self,
        ref_local: u32,
        hash: PluginTypeHash,
        dim: u32,
        scratch: u32,
    ) -> PResult<CodeSnippet>;

    fn bind_imports(&self, _imports: &dyn HostImports) {}
}
