//! Memory plugin — mirrors `speet_memory::{AddressMapper, MemoryAccess}`.
//!
//! All methods take `&self`, not `&mut self`: plugin trait objects are
//! stored as `Arc<dyn Trait>` in `PluginRegistry` so they can be shared as
//! host-entity imports across multiple callers (see `imports` module docs).
//! Implementations needing per-call mutable state must use a thread-safe
//! interior-mutability primitive (`Mutex`/`RwLock`/atomics) — never
//! `Cell`/`RefCell`, since every plugin trait here requires `Send + Sync`.
//! This mirrors the "replace `&mut self` on hook traits with interior
//! mutability" rule already in `docs/guides/parallel-api.md`.

use alloc::vec::Vec;

use crate::error::PResult;
use crate::imports::HostImports;
use crate::snippet::{CodeSnippet, PluginValType};
use crate::wire::{WireDecode, WireEncode, WireError};

/// Mirrors `speet_memory::LoadKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginLoadKind {
    I8S,
    I8U,
    I16S,
    I16U,
    I32S,
    I32U,
    I64,
    F32,
    F64,
}

/// Mirrors `speet_memory::StoreKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginStoreKind {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
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

wire_c_enum!(PluginLoadKind {
    I8S = 0, I8U = 1, I16S = 2, I16U = 3, I32S = 4, I32U = 5, I64 = 6, F32 = 7, F64 = 8,
});

wire_c_enum!(PluginStoreKind {
    I8 = 0, I16 = 1, I32 = 2, I64 = 3, F32 = 4, F64 = 5,
});

/// Method tags for the `AddressMapperPlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMapperMethod {
    Translate = 0,
    DeclareParams = 1,
    DeclareLocals = 2,
    BindSlots = 3,
    ChunkSize = 4,
}

/// Method tags for the `MemoryAccessPlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessMethod {
    EmitLoad = 0,
    EmitStoreAddr = 1,
    EmitStoreInsn = 2,
    EmitMemorySize = 3,
    EmitMemoryGrow = 4,
    DeclareParams = 5,
    DeclareLocals = 6,
    BindSlots = 7,
    ChunkSize = 8,
}

/// Virtual-address translation: guest address (on a known local) → physical
/// address (left on the value stack). Mirrors `speet_memory::AddressMapper`.
pub trait AddressMapperPlugin: Send + Sync {
    /// Emit the translation snippet. `addr_local` holds the guest virtual
    /// address on entry; the snippet must leave the translated physical
    /// address on the value stack.
    fn translate(&self, addr_local: u32) -> PResult<CodeSnippet>;

    fn declare_params(&self) -> Vec<PluginValType> {
        Vec::new()
    }
    fn declare_locals(&self) -> Vec<PluginValType> {
        Vec::new()
    }
    /// Called once after allocation so the plugin knows its own slots'
    /// absolute indices for use inside `translate`'s snippet.
    fn bind_slots(&self, _param_base: u32, _local_base: u32) {}

    fn chunk_size(&self) -> Option<u64> {
        None
    }

    /// Bind host (or other-plugin) entities this plugin imports. No-op
    /// default — most memory plugins are self-contained. See
    /// `docs/guides/plugin-api.md` §7 ("host entity imports").
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

/// Full load/store sequencing. Mirrors `speet_memory::MemoryAccess`.
pub trait MemoryAccessPlugin: Send + Sync {
    fn emit_load(&self, addr_local: u32, kind: PluginLoadKind) -> PResult<CodeSnippet>;
    fn emit_store_addr(&self, addr_local: u32) -> PResult<CodeSnippet>;
    fn emit_store_insn(&self, phys_addr_local: u32, kind: PluginStoreKind) -> PResult<CodeSnippet>;
    fn emit_memory_size(&self) -> PResult<CodeSnippet> {
        Err(crate::error::PluginError::unsupported("emit_memory_size"))
    }
    fn emit_memory_grow(&self) -> PResult<CodeSnippet> {
        Err(crate::error::PluginError::unsupported("emit_memory_grow"))
    }
    fn declare_params(&self) -> Vec<PluginValType> {
        Vec::new()
    }
    fn declare_locals(&self) -> Vec<PluginValType> {
        Vec::new()
    }
    fn bind_slots(&self, _param_base: u32, _local_base: u32) {}
    fn chunk_size(&self) -> Option<u64> {
        None
    }
    fn bind_imports(&self, _imports: &dyn HostImports) {}
}
