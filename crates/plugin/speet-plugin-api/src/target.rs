//! Target plugin — mirrors `speet_module_target::ModuleTarget` +
//! `speet_syscall::{SyscallTable, SyscallEntry}`.
//!
//! The most data-shaped plugin kind: both methods return plain owned data,
//! no [`CodeSnippet`](crate::snippet::CodeSnippet)/[`PResult`](crate::error::PResult)
//! machinery at all, because `WasmSyscallDispatcher` (generic, already
//! built-in) does the actual dispatch codegen from this data.

use alloc::string::String;
use alloc::vec::Vec;

use crate::imports::HostImports;
use crate::snippet::PluginValType;
use crate::wire::{WireDecode, WireEncode, WireError};

// ── Module-level declarations (mirrors `speet_module_target::ModuleTarget`) ──

/// The module-level declarations a target needs: imports, memories, tables,
/// tags, and segments. Computed once per recompile.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModuleManifest {
    pub func_imports: Vec<FuncImportDecl>,
    pub globals: Vec<GlobalDecl>,
    pub memories: Vec<MemoryDecl>,
    pub tables: Vec<TableDecl>,
    pub tags: Vec<TagDecl>,
    pub memory_data: Vec<MemoryDataDecl>,
    pub passive_memory_data: Vec<Vec<u8>>,
    pub element_segments: Vec<ElementSegmentDecl>,
    pub passive_element_segments: Vec<Vec<PluginElement>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncImportDecl {
    pub module: String,
    pub field: String,
    pub params: Vec<PluginValType>,
    pub results: Vec<PluginValType>,
}

/// A plain-data mirror of `wasm_encoder::ConstExpr`, covering the common
/// instantiation-time-constant cases. Scoped down for v1 — see
/// `docs/guides/plugin-api.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PluginConstExpr {
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),
    RefNull,
    GlobalGet(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalDecl {
    pub val_type: PluginValType,
    pub mutable: bool,
    pub init: PluginConstExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryDecl {
    pub minimum: u64,
    pub maximum: Option<u64>,
    pub memory64: bool,
    pub shared: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableDecl {
    pub minimum: u64,
    pub maximum: Option<u64>,
    pub table64: bool,
    pub init: Option<PluginConstExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagDecl {
    pub func_type_params: Vec<PluginValType>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDataDecl {
    pub memory_index: u32,
    pub offset: PluginConstExpr,
    pub data: Vec<u8>,
}

/// A plain-data mirror of `wasm_encoder::Elements`, scoped to the common
/// function-reference-table case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginElement {
    Func(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementSegmentDecl {
    pub table_index: u32,
    pub offset: PluginConstExpr,
    pub elements: Vec<PluginElement>,
}

// ── Syscall dispatch table (mirrors `speet_syscall::{SyscallTable, SyscallEntry}`) ──

/// Mirrors `speet_syscall::ParamSource`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginParamSource {
    LocalI64AsI32(u32),
    LocalI32(u32),
    ConstI32(i32),
    ConstI64(i64),
}

/// Mirrors `speet_syscall::SavePair`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginSavePair {
    pub local_idx: u32,
    pub global_idx: u32,
}

/// Mirrors `speet_syscall::MemoryStore`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginMemoryStore {
    pub addr: u32,
    pub value_local: u32,
    pub value_is_i64: bool,
}

/// Mirrors `speet_syscall::SyscallEntry`. `import_idx` indexes
/// `ModuleManifest::func_imports` rather than naming a raw absolute WASM
/// function index, so the plugin never needs the host's global index space —
/// the adapter resolves it after declaring the manifest's imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginSyscallEntry {
    pub import_idx: u32,
    pub param_map: Vec<PluginParamSource>,
    pub saves: Vec<PluginSavePair>,
    pub result_local: Option<u32>,
    pub negate_nonzero_result: bool,
    pub has_return: bool,
    pub terminates: bool,
    pub memory_stores: Vec<PluginMemoryStore>,
    pub load_mem_on_success: Option<u32>,
}

/// Mirrors `speet_syscall::SyscallTable`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginSyscallTable {
    pub entries: Vec<(u64, PluginSyscallEntry)>,
}

/// Method tags for the `TargetPlugin` wire protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetMethod {
    ModuleManifest = 0,
    SyscallTable = 1,
}

/// OS/ABI environment: what a target needs declared in the module, and how
/// syscalls dispatch. Almost entirely declarative — see module docs.
///
/// `&self` throughout, matching every other plugin trait in this crate (see
/// `memory` module docs for why: shared `Arc<dyn Trait>` storage for
/// host-entity imports).
pub trait TargetPlugin: Send + Sync {
    fn module_manifest(&self) -> ModuleManifest;
    fn syscall_table(&self) -> PluginSyscallTable;

    fn bind_imports(&self, _imports: &dyn HostImports) {}
}

// ── Wire codec ──────────────────────────────────────────────────────────────

macro_rules! wire_struct {
    ($name:ident { $($field:ident : $ty:ty),+ $(,)? }) => {
        impl WireEncode for $name {
            fn encode(&self, out: &mut Vec<u8>) {
                $(self.$field.encode(out);)+
            }
        }
        impl WireDecode for $name {
            fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
                let rest = input;
                $(let ($field, rest): ($ty, &[u8]) = WireDecode::decode(rest)?;)+
                Ok((Self { $($field),+ }, rest))
            }
        }
    };
}

wire_struct!(FuncImportDecl {
    module: String, field: String, params: Vec<PluginValType>, results: Vec<PluginValType>
});

impl WireEncode for PluginConstExpr {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            PluginConstExpr::I32Const(v) => {
                0u8.encode(out);
                v.encode(out);
            }
            PluginConstExpr::I64Const(v) => {
                1u8.encode(out);
                v.encode(out);
            }
            PluginConstExpr::F32Const(v) => {
                2u8.encode(out);
                v.encode(out);
            }
            PluginConstExpr::F64Const(v) => {
                3u8.encode(out);
                v.encode(out);
            }
            PluginConstExpr::RefNull => 4u8.encode(out),
            PluginConstExpr::GlobalGet(idx) => {
                5u8.encode(out);
                idx.encode(out);
            }
        }
    }
}
impl WireDecode for PluginConstExpr {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (v, rest) = i32::decode(rest)?;
                (PluginConstExpr::I32Const(v), rest)
            }
            1 => {
                let (v, rest) = i64::decode(rest)?;
                (PluginConstExpr::I64Const(v), rest)
            }
            2 => {
                let (v, rest) = f32::decode(rest)?;
                (PluginConstExpr::F32Const(v), rest)
            }
            3 => {
                let (v, rest) = f64::decode(rest)?;
                (PluginConstExpr::F64Const(v), rest)
            }
            4 => (PluginConstExpr::RefNull, rest),
            5 => {
                let (idx, rest) = u32::decode(rest)?;
                (PluginConstExpr::GlobalGet(idx), rest)
            }
            _ => return Err(WireError),
        })
    }
}

wire_struct!(GlobalDecl {
    val_type: PluginValType, mutable: bool, init: PluginConstExpr
});
wire_struct!(MemoryDecl {
    minimum: u64, maximum: Option<u64>, memory64: bool, shared: bool
});
wire_struct!(TableDecl {
    minimum: u64, maximum: Option<u64>, table64: bool, init: Option<PluginConstExpr>
});
wire_struct!(TagDecl { func_type_params: Vec<PluginValType> });
wire_struct!(MemoryDataDecl {
    memory_index: u32, offset: PluginConstExpr, data: Vec<u8>
});

impl WireEncode for PluginElement {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            PluginElement::Func(idx) => idx.encode(out),
        }
    }
}
impl WireDecode for PluginElement {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (idx, rest) = u32::decode(input)?;
        Ok((PluginElement::Func(idx), rest))
    }
}

wire_struct!(ElementSegmentDecl {
    table_index: u32, offset: PluginConstExpr, elements: Vec<PluginElement>
});

wire_struct!(ModuleManifest {
    func_imports: Vec<FuncImportDecl>,
    globals: Vec<GlobalDecl>,
    memories: Vec<MemoryDecl>,
    tables: Vec<TableDecl>,
    tags: Vec<TagDecl>,
    memory_data: Vec<MemoryDataDecl>,
    passive_memory_data: Vec<Vec<u8>>,
    element_segments: Vec<ElementSegmentDecl>,
    passive_element_segments: Vec<Vec<PluginElement>>,
});

impl WireEncode for PluginParamSource {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            PluginParamSource::LocalI64AsI32(v) => {
                0u8.encode(out);
                v.encode(out);
            }
            PluginParamSource::LocalI32(v) => {
                1u8.encode(out);
                v.encode(out);
            }
            PluginParamSource::ConstI32(v) => {
                2u8.encode(out);
                v.encode(out);
            }
            PluginParamSource::ConstI64(v) => {
                3u8.encode(out);
                v.encode(out);
            }
        }
    }
}
impl WireDecode for PluginParamSource {
    fn decode(input: &[u8]) -> Result<(Self, &[u8]), WireError> {
        let (tag, rest) = u8::decode(input)?;
        Ok(match tag {
            0 => {
                let (v, rest) = u32::decode(rest)?;
                (PluginParamSource::LocalI64AsI32(v), rest)
            }
            1 => {
                let (v, rest) = u32::decode(rest)?;
                (PluginParamSource::LocalI32(v), rest)
            }
            2 => {
                let (v, rest) = i32::decode(rest)?;
                (PluginParamSource::ConstI32(v), rest)
            }
            3 => {
                let (v, rest) = i64::decode(rest)?;
                (PluginParamSource::ConstI64(v), rest)
            }
            _ => return Err(WireError),
        })
    }
}

wire_struct!(PluginSavePair { local_idx: u32, global_idx: u32 });
wire_struct!(PluginMemoryStore {
    addr: u32,
    value_local: u32,
    value_is_i64: bool
});
wire_struct!(PluginSyscallEntry {
    import_idx: u32,
    param_map: Vec<PluginParamSource>,
    saves: Vec<PluginSavePair>,
    result_local: Option<u32>,
    negate_nonzero_result: bool,
    has_return: bool,
    terminates: bool,
    memory_stores: Vec<PluginMemoryStore>,
    load_mem_on_success: Option<u32>,
});
wire_struct!(PluginSyscallTable { entries: Vec<(u64, PluginSyscallEntry)> });

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn module_manifest_roundtrip() {
        let manifest = ModuleManifest {
            func_imports: vec![FuncImportDecl {
                module: "wasi_snapshot_preview1".into(),
                field: "proc_exit".into(),
                params: vec![PluginValType::I32],
                results: vec![],
            }],
            ..Default::default()
        };
        let mut out = Vec::new();
        manifest.encode(&mut out);
        let (decoded, rest) = ModuleManifest::decode(&out).unwrap();
        assert_eq!(decoded, manifest);
        assert!(rest.is_empty());
    }

    #[test]
    fn syscall_table_roundtrip() {
        let table = PluginSyscallTable {
            entries: vec![(
                93,
                PluginSyscallEntry {
                    import_idx: 0,
                    param_map: vec![PluginParamSource::LocalI64AsI32(10)],
                    saves: vec![],
                    result_local: None,
                    negate_nonzero_result: false,
                    has_return: false,
                    terminates: true,
                    memory_stores: vec![],
                    load_mem_on_success: None,
                },
            )],
        };
        let mut out = Vec::new();
        table.encode(&mut out);
        let (decoded, rest) = PluginSyscallTable::decode(&out).unwrap();
        assert_eq!(decoded, table);
        assert!(rest.is_empty());
    }
}
