//! End-to-end tests against real `.wasm` guest modules, built by hand with
//! `wasm-encoder` (no toolchain/wasm32-target dependency for the test
//! suite). Each fixture implements the ABI documented in
//! `speet_plugin_host_wasm::abi` for exactly one plugin trait method, just
//! enough to prove the host's marshalling convention round-trips real data
//! — not a general-purpose guest SDK.

use std::sync::Arc;

use speet_plugin_api::imports::NoImports;
use speet_plugin_api::memory::AddressMapperPlugin;
use speet_plugin_api::remote::{AddressMapperResponse, TargetResponse};
use speet_plugin_api::snippet::{CodeSnippet, PluginValType};
use speet_plugin_api::target::{FuncImportDecl, ModuleManifest, PluginSyscallTable, TargetPlugin};
use speet_plugin_api::wire::WireEncode;
use speet_plugin_api::{HostImports, PResult, PluginError, PluginKind};
use speet_plugin_host::PluginRegistry;
use speet_plugin_host_wasm::abi::{pack, ImportRole};
use speet_plugin_host_wasm::{WasmAddressMapperPlugin, WasmPlugin, WasmTargetPlugin};
use wasm_encoder::{
    BlockType, CodeSection, DataSection, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType, ImportSection, Instruction, MemArg, MemorySection,
    MemoryType, Module, TypeSection, ValType,
};

// ── Fixture 1: a constant-data TargetPlugin (no request parsing needed —
// `module_manifest`/`syscall_table` take no arguments) ─────────────────────

fn sample_manifest() -> ModuleManifest {
    ModuleManifest {
        func_imports: vec![
            FuncImportDecl {
                module: "wasi_snapshot_preview1".into(),
                field: "proc_exit".into(),
                params: vec![PluginValType::I32],
                results: vec![],
            },
            FuncImportDecl {
                module: "wasi_snapshot_preview1".into(),
                field: "fd_write".into(),
                params: vec![
                    PluginValType::I32,
                    PluginValType::I32,
                    PluginValType::I32,
                    PluginValType::I32,
                ],
                results: vec![PluginValType::I32],
            },
        ],
        ..Default::default()
    }
}

fn sample_syscall_table() -> PluginSyscallTable {
    PluginSyscallTable::default()
}

/// `speet_plugin_call` ignores its request entirely and returns whichever
/// of two fixed data segments the request's leading tag byte selects.
fn build_target_module(manifest_bytes: &[u8], table_bytes: &[u8]) -> Vec<u8> {
    let manifest_addr = 0u32;
    let table_addr = 4096u32;
    let bump_init = 8192i32;

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]); // alloc
    types.ty().function([ValType::I32, ValType::I32], [ValType::I64]); // call

    let mut funcs = FunctionSection::new();
    funcs.function(0);
    funcs.function(1);

    let mut mem = MemorySection::new();
    mem.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut globals = GlobalSection::new();
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm_encoder::ConstExpr::i32_const(bump_init),
    );

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("speet_plugin_alloc", ExportKind::Func, 0);
    exports.export("speet_plugin_call", ExportKind::Func, 1);

    let mut code = CodeSection::new();
    code.function(&bump_alloc_fn(0));

    let mut call_fn = Function::new([]);
    call_fn.instruction(&Instruction::LocalGet(0)); // req_ptr
    call_fn.instruction(&Instruction::I32Load8U(MemArg {
        offset: 0,
        align: 0,
        memory_index: 0,
    }));
    call_fn.instruction(&Instruction::I32Eqz); // tag == 0 (ModuleManifest)?
    call_fn.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    call_fn.instruction(&Instruction::I64Const(pack(manifest_addr, manifest_bytes.len() as u32)));
    call_fn.instruction(&Instruction::Else);
    call_fn.instruction(&Instruction::I64Const(pack(table_addr, table_bytes.len() as u32)));
    call_fn.instruction(&Instruction::End);
    call_fn.instruction(&Instruction::End);
    code.function(&call_fn);

    let mut data = DataSection::new();
    data.active(0, &wasm_encoder::ConstExpr::i32_const(manifest_addr as i32), manifest_bytes.iter().copied());
    data.active(0, &wasm_encoder::ConstExpr::i32_const(table_addr as i32), table_bytes.iter().copied());

    assemble(types, None, funcs, mem, globals, exports, code, data)
}

// ── Fixture 2: AddressMapperPlugin::translate with a dynamic argument ──────

/// `speet_plugin_call` reads `addr_local`'s low byte straight out of the
/// request (offset 1 — past the 1-byte request tag), patches it into a
/// pre-encoded response template's last byte, and returns that template.
/// Only correct for `addr_local < 128` (single-byte LEB128) — fine for a
/// test fixture proving the marshalling convention, not a general guest SDK.
fn build_translate_module(template: &[u8]) -> Vec<u8> {
    let template_addr = 0u32;
    let bump_init = 8192i32;
    let patch_offset = (template.len() - 1) as u64;

    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I32]);
    types.ty().function([ValType::I32, ValType::I32], [ValType::I64]);

    let mut funcs = FunctionSection::new();
    funcs.function(0);
    funcs.function(1);

    let mut mem = MemorySection::new();
    mem.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut globals = GlobalSection::new();
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm_encoder::ConstExpr::i32_const(bump_init),
    );

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("speet_plugin_alloc", ExportKind::Func, 0);
    exports.export("speet_plugin_call", ExportKind::Func, 1);

    let mut code = CodeSection::new();
    code.function(&bump_alloc_fn(0));

    let mut call_fn = Function::new([(1, ValType::I32)]);
    let b_local = 2u32; // params occupy 0,1; first declared local is index 2
    call_fn.instruction(&Instruction::LocalGet(0)); // req_ptr
    call_fn.instruction(&Instruction::I32Load8U(MemArg {
        offset: 1,
        align: 0,
        memory_index: 0,
    }));
    call_fn.instruction(&Instruction::LocalSet(b_local));
    call_fn.instruction(&Instruction::I32Const(template_addr as i32));
    call_fn.instruction(&Instruction::LocalGet(b_local));
    call_fn.instruction(&Instruction::I32Store8(MemArg {
        offset: patch_offset,
        align: 0,
        memory_index: 0,
    }));
    call_fn.instruction(&Instruction::I64Const(pack(template_addr, template.len() as u32)));
    call_fn.instruction(&Instruction::End);
    code.function(&call_fn);

    let mut data = DataSection::new();
    data.active(0, &wasm_encoder::ConstExpr::i32_const(template_addr as i32), template.iter().copied());

    assemble(types, None, funcs, mem, globals, exports, code, data)
}

// ── Fixture 3: a forwarding AddressMapperPlugin that delegates `translate`
// to a host-granted import named "host-mmu" (§2.7's WASM realization) ──────

/// Forwards every `speet_plugin_call` verbatim to
/// `host.speet_host_call(AddressMapper, "host-mmu", ...)`. If the import
/// resolves to nothing (denied or unknown — `speet_host_call` returns the
/// `(0, 0)` sentinel, see `host_calls` module docs), returns a pre-encoded
/// `Err(PluginError)` response instead of forwarding garbage — what a real,
/// well-behaved guest should do.
fn build_forwarding_module(name: &[u8], denied_template: &[u8]) -> Vec<u8> {
    let name_addr = 0u32;
    let denied_addr = 1024u32;
    let bump_init = 8192i32;

    let mut types = TypeSection::new();
    types.ty().function(
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [ValType::I64],
    ); // speet_host_call
    types.ty().function([ValType::I32], [ValType::I32]); // alloc
    types.ty().function([ValType::I32, ValType::I32], [ValType::I64]); // call

    let mut imports = ImportSection::new();
    imports.import("host", "speet_host_call", EntityType::Function(0));

    let mut funcs = FunctionSection::new();
    funcs.function(1);
    funcs.function(2);

    let mut mem = MemorySection::new();
    mem.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut globals = GlobalSection::new();
    globals.global(
        GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm_encoder::ConstExpr::i32_const(bump_init),
    );

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("speet_plugin_alloc", ExportKind::Func, 1);
    exports.export("speet_plugin_call", ExportKind::Func, 2);

    let mut code = CodeSection::new();
    code.function(&bump_alloc_fn(0));

    let mut call_fn = Function::new([(1, ValType::I64)]);
    let packed_local = 2u32; // params occupy 0,1
    call_fn.instruction(&Instruction::I32Const(ImportRole::AddressMapper as i32));
    call_fn.instruction(&Instruction::I32Const(name_addr as i32));
    call_fn.instruction(&Instruction::I32Const(name.len() as i32));
    call_fn.instruction(&Instruction::LocalGet(0)); // req_ptr
    call_fn.instruction(&Instruction::LocalGet(1)); // req_len
    call_fn.instruction(&Instruction::Call(0)); // imported speet_host_call
    call_fn.instruction(&Instruction::LocalSet(packed_local));
    call_fn.instruction(&Instruction::LocalGet(packed_local));
    call_fn.instruction(&Instruction::I64Eqz);
    call_fn.instruction(&Instruction::If(BlockType::Result(ValType::I64)));
    call_fn.instruction(&Instruction::I64Const(pack(denied_addr, denied_template.len() as u32)));
    call_fn.instruction(&Instruction::Else);
    call_fn.instruction(&Instruction::LocalGet(packed_local));
    call_fn.instruction(&Instruction::End);
    call_fn.instruction(&Instruction::End);
    code.function(&call_fn);

    let mut data = DataSection::new();
    data.active(0, &wasm_encoder::ConstExpr::i32_const(name_addr as i32), name.iter().copied());
    data.active(0, &wasm_encoder::ConstExpr::i32_const(denied_addr as i32), denied_template.iter().copied());

    assemble(types, Some(imports), funcs, mem, globals, exports, code, data)
}

/// Shared bump allocator: `(len) -> ptr`, advancing global 0 by `len` each
/// call. `type_idx` is the alloc function's TypeSection index.
fn bump_alloc_fn(_type_idx: u32) -> Function {
    let mut f = Function::new([(1, ValType::I32)]);
    let ptr_local = 1u32; // param 0 is `len`
    f.instruction(&Instruction::GlobalGet(0));
    f.instruction(&Instruction::LocalSet(ptr_local));
    f.instruction(&Instruction::GlobalGet(0));
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::GlobalSet(0));
    f.instruction(&Instruction::LocalGet(ptr_local));
    f.instruction(&Instruction::End);
    f
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    types: TypeSection,
    imports: Option<ImportSection>,
    funcs: FunctionSection,
    mem: MemorySection,
    globals: GlobalSection,
    exports: ExportSection,
    code: CodeSection,
    data: DataSection,
) -> Vec<u8> {
    let mut module = Module::new();
    module.section(&types);
    if let Some(imports) = &imports {
        module.section(imports);
    }
    module.section(&funcs);
    module.section(&mem);
    module.section(&globals);
    module.section(&exports);
    module.section(&code);
    module.section(&data);
    let bytes = module.finish();
    wasmparser::validate(&bytes).expect("fixture module failed wasm validation");
    bytes
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn target_plugin_constant_data_roundtrip() {
    let manifest = sample_manifest();
    let table = sample_syscall_table();

    let mut manifest_bytes = Vec::new();
    TargetResponse::ModuleManifest {
        manifest: manifest.clone(),
    }
    .encode(&mut manifest_bytes);
    let mut table_bytes = Vec::new();
    TargetResponse::SyscallTable { table: table.clone() }.encode(&mut table_bytes);

    let wasm = build_target_module(&manifest_bytes, &table_bytes);
    let engine = WasmPlugin::load(&wasm, Arc::new(NoImports)).expect("load");
    let plugin = WasmTargetPlugin::new(engine);

    assert_eq!(plugin.module_manifest(), manifest);
    assert_eq!(plugin.syscall_table(), table);
}

#[test]
fn address_mapper_translate_uses_dynamic_argument() {
    let placeholder = AddressMapperResponse::Translate {
        snippet: Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(0)])),
    };
    let mut template = Vec::new();
    placeholder.encode(&mut template);

    let wasm = build_translate_module(&template);
    let engine = WasmPlugin::load(&wasm, Arc::new(NoImports)).expect("load");
    let plugin = WasmAddressMapperPlugin::new(engine);

    let snippet = plugin.translate(7).expect("translate");
    assert_eq!(
        snippet,
        CodeSnippet::from_instructions(&[Instruction::LocalGet(7)])
    );
}

struct ToyMemory;
impl AddressMapperPlugin for ToyMemory {
    fn translate(&self, addr_local: u32) -> PResult<CodeSnippet> {
        Ok(CodeSnippet::from_instructions(&[Instruction::LocalGet(addr_local)]))
    }
}

fn denied_template() -> Vec<u8> {
    let mut out = Vec::new();
    AddressMapperResponse::Translate {
        snippet: Err(PluginError::new(1, "import denied")),
    }
    .encode(&mut out);
    out
}

#[test]
fn host_entity_import_granted_forwards_to_host_plugin() {
    let mut registry = PluginRegistry::new();
    registry.register_static(
        "host-mmu",
        speet_plugin_host::PluginHandle::address_mapper(Arc::new(ToyMemory)),
    );
    let granted = [(PluginKind::Memory, "host-mmu".to_string())];
    let imports: Arc<dyn HostImports> = Arc::new(owned_restricted_view(registry, &granted));

    let wasm = build_forwarding_module(b"host-mmu", &denied_template());
    let engine = WasmPlugin::load(&wasm, imports).expect("load");
    let plugin = WasmAddressMapperPlugin::new(engine);

    let snippet = plugin.translate(9).expect("translate via import");
    assert_eq!(
        snippet,
        CodeSnippet::from_instructions(&[Instruction::LocalGet(9)])
    );
}

#[test]
fn host_entity_import_denied_when_not_granted() {
    let mut registry = PluginRegistry::new();
    registry.register_static(
        "host-mmu",
        speet_plugin_host::PluginHandle::address_mapper(Arc::new(ToyMemory)),
    );
    let granted: [(PluginKind, String); 0] = [];
    let imports: Arc<dyn HostImports> = Arc::new(owned_restricted_view(registry, &granted));

    let wasm = build_forwarding_module(b"host-mmu", &denied_template());
    let engine = WasmPlugin::load(&wasm, imports).expect("load");
    let plugin = WasmAddressMapperPlugin::new(engine);

    let err = plugin.translate(9).expect_err("import must be denied");
    assert_eq!(err, PluginError::new(1, "import denied"));
}

/// Bundles a [`PluginRegistry`] and a `'static` allowlist into one owned,
/// `Send + Sync` [`HostImports`] — `RestrictedHostImports` borrows both, so
/// this self-owning wrapper is what lets the registry outlive this
/// function's scope inside an `Arc<dyn HostImports>`.
struct OwnedRestrictedView {
    registry: PluginRegistry,
    granted: Vec<(PluginKind, String)>,
}
impl HostImports for OwnedRestrictedView {
    fn address_mapper(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::memory::AddressMapperPlugin>> {
        self.registry.restricted_view(&self.granted).address_mapper(name)
    }
    fn memory_access(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::memory::MemoryAccessPlugin>> {
        self.registry.restricted_view(&self.granted).memory_access(name)
    }
    fn table(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::table::TablePlugin>> {
        self.registry.restricted_view(&self.granted).table(name)
    }
    fn object_model(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::object_model::ObjectModelPlugin>> {
        self.registry.restricted_view(&self.granted).object_model(name)
    }
    fn target(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::target::TargetPlugin>> {
        self.registry.restricted_view(&self.granted).target(name)
    }
    fn arch(&self, name: &str) -> Option<Arc<dyn speet_plugin_api::arch::ArchPlugin>> {
        self.registry.restricted_view(&self.granted).arch(name)
    }
}

fn owned_restricted_view(registry: PluginRegistry, granted: &[(PluginKind, String)]) -> OwnedRestrictedView {
    OwnedRestrictedView {
        registry,
        granted: granted.to_vec(),
    }
}
