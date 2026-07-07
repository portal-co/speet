//! Target adapter — replays a [`TargetPlugin`]'s declarative data into a
//! real `speet_module_target::ModuleTarget` and a real
//! `speet_syscall::SyscallTable`. The most data-shaped adapter: no
//! `CodeSnippet` replay needed, since `TargetPlugin` returns plain owned
//! data (see `speet_plugin_api::target` module docs).

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use speet_module_target::ModuleTarget;
use speet_plugin_api::target::{
    PluginConstExpr, PluginElement, PluginParamSource, TargetPlugin,
};
use speet_syscall::{MemoryStore, ParamSource, SavePair, SyscallEntry, SyscallTable};
use wasm_encoder::{ConstExpr, Elements, GlobalType, HeapType, MemoryType, RefType, TableType, ValType};

fn convert_const_expr(c: &PluginConstExpr) -> ConstExpr {
    match *c {
        PluginConstExpr::I32Const(v) => ConstExpr::i32_const(v),
        PluginConstExpr::I64Const(v) => ConstExpr::i64_const(v),
        PluginConstExpr::F32Const(v) => ConstExpr::f32_const(v.into()),
        PluginConstExpr::F64Const(v) => ConstExpr::f64_const(v.into()),
        PluginConstExpr::RefNull => ConstExpr::ref_null(HeapType::FUNC),
        PluginConstExpr::GlobalGet(idx) => ConstExpr::global_get(idx),
    }
}

fn convert_elements(elements: &[PluginElement]) -> Elements<'static> {
    let funcs: Vec<u32> = elements
        .iter()
        .map(|e| match e {
            PluginElement::Func(idx) => *idx,
        })
        .collect();
    Elements::Functions(Cow::Owned(funcs))
}

fn convert_param_source(p: PluginParamSource) -> ParamSource {
    match p {
        PluginParamSource::LocalI64AsI32(v) => ParamSource::LocalI64AsI32(v),
        PluginParamSource::LocalI32(v) => ParamSource::LocalI32(v),
        PluginParamSource::ConstI32(v) => ParamSource::ConstI32(v),
        PluginParamSource::ConstI64(v) => ParamSource::ConstI64(v),
    }
}

/// Replays a [`TargetPlugin`]'s [`ModuleManifest`](speet_plugin_api::target::ModuleManifest)
/// into any `dyn ModuleTarget<Ctx, Err>` (e.g. a `MegabinaryBuilder` or
/// `ModuleBuilder`), the same way a hand-written environment crate like
/// `speet-linux-wasi` declares its own imports.
pub struct PluginModuleTargetDeclarator<Ctx, Err> {
    plugin: Arc<dyn TargetPlugin>,
    _marker: PhantomData<fn(Ctx, Err)>,
}

impl<Ctx, Err> PluginModuleTargetDeclarator<Ctx, Err> {
    pub fn new(plugin: Arc<dyn TargetPlugin>) -> Self {
        Self {
            plugin,
            _marker: PhantomData,
        }
    }

    /// Declare every entity in the plugin's `module_manifest()` into
    /// `module`. Returns the allocated index of each `FuncImportDecl`, in
    /// `module_manifest().func_imports` order — `materialize_syscall_table`
    /// needs this to resolve `PluginSyscallEntry::import_idx` (an index into
    /// that same list) into a real absolute WASM function index.
    pub fn declare_into(
        &self,
        ctx: &mut Ctx,
        module: &mut (dyn ModuleTarget<Ctx, Err> + '_),
    ) -> Result<Vec<u32>, Err> {
        let manifest = self.plugin.module_manifest();

        let mut import_indices = Vec::with_capacity(manifest.func_imports.len());
        for decl in &manifest.func_imports {
            let params: Vec<ValType> = decl.params.iter().map(|v| (*v).into()).collect();
            let results: Vec<ValType> = decl.results.iter().map(|v| (*v).into()).collect();
            let idx =
                module.declare_func_import(ctx, &decl.module, &decl.field, &params, &results)?;
            import_indices.push(idx);
        }

        for decl in &manifest.globals {
            let ty = GlobalType {
                val_type: decl.val_type.into(),
                mutable: decl.mutable,
                shared: false,
            };
            module.declare_global(ctx, ty, &convert_const_expr(&decl.init))?;
        }

        for decl in &manifest.memories {
            let ty = MemoryType {
                minimum: decl.minimum,
                maximum: decl.maximum,
                memory64: decl.memory64,
                shared: decl.shared,
                page_size_log2: None,
            };
            module.declare_memory(ctx, ty)?;
        }

        for decl in &manifest.tables {
            // `element_type` always FUNCREF: `TableDecl` is scoped to the
            // common function-reference-table case for v1, mirroring
            // `PluginElement`'s own scope-down — see module docs.
            let ty = TableType {
                element_type: RefType::FUNCREF,
                minimum: decl.minimum,
                maximum: decl.maximum,
                table64: decl.table64,
                shared: false,
            };
            let init = decl.init.as_ref().map(convert_const_expr);
            module.declare_table(ctx, ty, init.as_ref())?;
        }

        if !manifest.tags.is_empty() {
            // `ModuleTarget` has no primitive to intern a bare function
            // type, so a `TagType.func_type_idx` can't be derived from
            // `TagDecl::func_type_params` alone without changing that trait
            // (out of scope — see docs/guides/plugin-api.md). Loud failure
            // rather than emitting a wrong/dangling type index.
            unimplemented!(
                "TargetPlugin tags are not yet supported by the plugin adapter — see \
                 docs/guides/plugin-api.md"
            );
        }

        for decl in &manifest.memory_data {
            module.add_memory_data(
                ctx,
                decl.memory_index,
                &convert_const_expr(&decl.offset),
                &decl.data,
            )?;
        }
        for data in &manifest.passive_memory_data {
            module.add_passive_memory_data(ctx, data)?;
        }

        for decl in &manifest.element_segments {
            let elements = convert_elements(&decl.elements);
            module.add_element_segment(
                ctx,
                decl.table_index,
                &convert_const_expr(&decl.offset),
                elements,
            )?;
        }
        for elements in &manifest.passive_element_segments {
            module.add_passive_element_segment(ctx, convert_elements(elements))?;
        }

        Ok(import_indices)
    }
}

/// Produce a real [`SyscallTable`] from a [`TargetPlugin`]'s
/// `syscall_table()`. `import_indices` must be
/// [`PluginModuleTargetDeclarator::declare_into`]'s return value for the
/// *same* plugin — `PluginSyscallEntry::import_idx` indexes into
/// `module_manifest().func_imports`, not a raw absolute WASM function
/// index, so the plugin never needs the host's global index space.
pub fn materialize_syscall_table(plugin: &dyn TargetPlugin, import_indices: &[u32]) -> SyscallTable {
    let table = plugin.syscall_table();
    let entries = table
        .entries
        .into_iter()
        .map(|(num, entry)| {
            let func_idx = import_indices[entry.import_idx as usize];
            let param_map = entry
                .param_map
                .into_iter()
                .map(convert_param_source)
                .collect();
            let saves = entry
                .saves
                .into_iter()
                .map(|s| SavePair {
                    local_idx: s.local_idx,
                    global_idx: s.global_idx,
                })
                .collect();
            let memory_stores = entry
                .memory_stores
                .into_iter()
                .map(|m| MemoryStore {
                    addr: m.addr,
                    value_local: m.value_local,
                    value_is_i64: m.value_is_i64,
                })
                .collect();
            (
                num,
                SyscallEntry {
                    func_idx,
                    param_map,
                    saves,
                    result_local: entry.result_local,
                    negate_nonzero_result: entry.negate_nonzero_result,
                    has_return: entry.has_return,
                    terminates: entry.terminates,
                    memory_stores,
                    load_mem_on_success: entry.load_mem_on_success,
                },
            )
        })
        .collect();
    SyscallTable::new(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use core::convert::Infallible;
    use speet_link_core::EntityIndexSpace;
    use speet_linux_wasi::{LinuxToWasi, WasiImports};
    use speet_module_builder::ModuleBuilder;
    use speet_plugin_api::snippet::PluginValType;
    use speet_plugin_api::target::{
        FuncImportDecl, ModuleManifest, PluginMemoryStore, PluginSyscallEntry, PluginSyscallTable,
    };

    /// Reproduces a subset of `speet-linux-wasi`'s `WasiImports` +
    /// `LinuxToWasi` (read/write/close/exit/exit_group over RV64 registers
    /// x10-x12) as a `TargetPlugin`, for byte-identical-output comparison
    /// against the hand-written crate.
    struct PocWasiTarget;

    const IOVEC_SCRATCH_OFFSET: u32 = 0x200;
    // Default RV64 layout: locals 0-31 = x0-x31.
    const A0: u32 = 10;
    const A1: u32 = 11;
    const A2: u32 = 12;

    impl TargetPlugin for PocWasiTarget {
        fn module_manifest(&self) -> ModuleManifest {
            let import = |field: &str, params: &[PluginValType], results: &[PluginValType]| {
                FuncImportDecl {
                    module: String::from("wasi_snapshot_preview1"),
                    field: String::from(field),
                    params: params.to_vec(),
                    results: results.to_vec(),
                }
            };
            ModuleManifest {
                func_imports: alloc::vec![
                    import(
                        "fd_write",
                        &[
                            PluginValType::I32,
                            PluginValType::I32,
                            PluginValType::I32,
                            PluginValType::I32
                        ],
                        &[PluginValType::I32]
                    ),
                    import(
                        "fd_read",
                        &[
                            PluginValType::I32,
                            PluginValType::I32,
                            PluginValType::I32,
                            PluginValType::I32
                        ],
                        &[PluginValType::I32]
                    ),
                    import("fd_close", &[PluginValType::I32], &[PluginValType::I32]),
                    import("proc_exit", &[PluginValType::I32], &[]),
                ],
                ..Default::default()
            }
        }

        fn syscall_table(&self) -> PluginSyscallTable {
            // import_idx: 0=fd_write 1=fd_read 2=fd_close 3=proc_exit
            // (declaration order from module_manifest(), above)
            let rw_entry = |import_idx: u32| PluginSyscallEntry {
                import_idx,
                param_map: alloc::vec![
                    PluginParamSource::LocalI64AsI32(A0),
                    PluginParamSource::ConstI32(IOVEC_SCRATCH_OFFSET as i32),
                    PluginParamSource::ConstI32(1),
                    PluginParamSource::ConstI32((IOVEC_SCRATCH_OFFSET + 8) as i32),
                ],
                saves: Vec::new(),
                result_local: Some(A0),
                negate_nonzero_result: true,
                has_return: true,
                terminates: false,
                memory_stores: alloc::vec![
                    PluginMemoryStore {
                        addr: IOVEC_SCRATCH_OFFSET,
                        value_local: A1,
                        value_is_i64: true,
                    },
                    PluginMemoryStore {
                        addr: IOVEC_SCRATCH_OFFSET + 4,
                        value_local: A2,
                        value_is_i64: true,
                    },
                ],
                load_mem_on_success: Some(IOVEC_SCRATCH_OFFSET + 8),
            };
            let exit_entry = || PluginSyscallEntry {
                import_idx: 3,
                param_map: alloc::vec![PluginParamSource::LocalI64AsI32(A0)],
                saves: Vec::new(),
                result_local: None,
                negate_nonzero_result: false,
                has_return: false,
                terminates: true,
                memory_stores: Vec::new(),
                load_mem_on_success: None,
            };
            PluginSyscallTable {
                entries: alloc::vec![
                    (63, rw_entry(1)), // read
                    (64, rw_entry(0)), // write
                    (
                        57,
                        PluginSyscallEntry {
                            import_idx: 2,
                            param_map: alloc::vec![PluginParamSource::LocalI64AsI32(A0)],
                            saves: Vec::new(),
                            result_local: Some(A0),
                            negate_nonzero_result: true,
                            has_return: true,
                            terminates: false,
                            memory_stores: Vec::new(),
                            load_mem_on_success: None,
                        },
                    ), // close
                    (93, exit_entry()),  // exit
                    (94, exit_entry()),  // exit_group
                ],
            }
        }
    }

    #[test]
    fn module_manifest_matches_hand_written_speet_linux_wasi() {
        // Hand-written side.
        let mut entity_space = EntityIndexSpace::empty();
        let imports = WasiImports::register(&mut entity_space);
        let mut hand_written = ModuleBuilder::new();
        imports
            .declare(&mut hand_written, &mut ())
            .unwrap();

        // Plugin-adapter side.
        let plugin: Arc<dyn TargetPlugin> = Arc::new(PocWasiTarget);
        let declarator: PluginModuleTargetDeclarator<(), Infallible> =
            PluginModuleTargetDeclarator::new(plugin);
        let mut from_plugin = ModuleBuilder::new();
        let import_indices = declarator.declare_into(&mut (), &mut from_plugin).unwrap();
        assert_eq!(import_indices, alloc::vec![0u32, 1, 2, 3]);

        // Byte-identical resulting WASM module bytes — the headline parity
        // check (mirrors the plan's three-host cross-check methodology).
        assert_eq!(
            hand_written.finish().finish(),
            from_plugin.finish().finish()
        );
    }

    #[test]
    fn syscall_table_matches_hand_written_speet_linux_wasi() {
        let mut entity_space = EntityIndexSpace::empty();
        let imports = WasiImports::register(&mut entity_space);
        let import_indices = alloc::vec![
            imports.fd_write,
            imports.fd_read,
            imports.fd_close,
            imports.proc_exit,
        ];
        let hand_written = LinuxToWasi::new(imports).build_table(|n| n as u32);

        let plugin = PocWasiTarget;
        let from_plugin = materialize_syscall_table(&plugin, &import_indices);

        // `SyscallTable`/`SyscallEntry` don't derive `PartialEq` (and adding
        // it is out of scope — zero changes to `speet-syscall`), so compare
        // via `Debug` formatting, which is deterministic for these types.
        assert_eq!(
            format!("{:?}", hand_written.entries()),
            format!("{:?}", from_plugin.entries())
        );
    }
}
