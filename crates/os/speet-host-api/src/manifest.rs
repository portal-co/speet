//! Import manifest and link recipe types.

use binary_io::{BinArch, BinOs};

/// A WASM value type, kept neutral (no `wasm_encoder` dependency in this
/// crate) so both `speet-recompile` (building real WASM types) and
/// `speet-rt` (which only cares about names) can consume the same
/// [`FuncImport`] without a to-`wasm_encoder` conversion living outside
/// `speet-recompile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmValType {
    I32,
    I64,
    F32,
    F64,
}

/// One WASM function import the link shim must satisfy.
///
/// `params`/`results` make the import's signature part of the manifest
/// itself, rather than requiring a second, separately-maintained match on
/// `(module, name)` wherever the type section is built — see
/// `docs/guides/thin-runtime-genericity.md` principle 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncImport {
    pub module: String,
    pub name: String,
    pub params: Vec<WasmValType>,
    pub results: Vec<WasmValType>,
    /// Guest-visible external symbol names (PLT/import-table entries) this
    /// slot intercepts, e.g. `exit`'s `["exit", "_exit"]`. Empty for
    /// internal-only slots a guest never calls by name (`__speet_hint`,
    /// `__speet_log_unreachable`). This is what lets a `HostApi::
    /// resolve_plt_redirect` implementation derive its whole hook set by
    /// walking `import_manifest().func_imports` instead of hand-maintaining
    /// a separate `match symbol { "x" => ..., ... }` table — see
    /// `docs/guides/thin-runtime-genericity.md` principle 1.
    pub intercepts: Vec<String>,
}

/// Imports and metadata needed to generate the C link shim.
#[derive(Debug, Clone, Default)]
pub struct ImportManifest {
    pub func_imports: Vec<FuncImport>,
}

impl ImportManifest {
    /// Standard native-syscall shim imports (`env.exit`, `env.write`, `env.__speet_hint`).
    pub fn native_syscall() -> Self {
        use WasmValType::I32;
        Self {
            func_imports: vec![
                FuncImport {
                    module: "env".into(),
                    name: "__speet_hint".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec![],
                },
                FuncImport {
                    module: "env".into(),
                    name: "exit".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec!["exit".into(), "_exit".into(), "_Exit".into()],
                },
                FuncImport {
                    module: "env".into(),
                    name: "write".into(),
                    params: vec![I32, I32, I32],
                    results: vec![I32],
                    intercepts: vec!["write".into(), "_write".into()],
                },
            ],
        }
    }

    /// Integrated thin-runtime manifest: syscall imports + unreachable logging + hooks.
    pub fn integrated_native() -> Self {
        use WasmValType::{I32, I64};
        let mut m = Self::native_syscall();
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "__speet_log_unreachable".into(),
            params: vec![I32],
            results: vec![],
            intercepts: vec![],
        });
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "__speet_execve".into(),
            params: vec![I64, I64, I64],
            results: vec![I32],
            intercepts: vec!["execve".into(), "_execve".into()],
        });
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "__speet_stub_for_pc".into(),
            params: vec![I64],
            results: vec![I64],
            intercepts: vec![],
        });
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "printf".into(),
            params: vec![I64, I64],
            results: vec![I32],
            intercepts: vec!["printf".into()],
        });
        m
    }

    /// Corpus harness manifest: syscall imports plus a wasmi-side unreachable
    /// trap hook (`env.__speet_unreachable_trap`) used by
    /// `speet-corpus-harness` instead of the integrated runtime's
    /// `__speet_log_unreachable`. Order matches historical corpus tests:
    /// hint, write, exit, trap.
    pub fn corpus_harness() -> Self {
        use WasmValType::I32;
        Self {
            func_imports: vec![
                FuncImport {
                    module: "env".into(),
                    name: "__speet_hint".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec![],
                },
                FuncImport {
                    module: "env".into(),
                    name: "write".into(),
                    params: vec![I32, I32, I32],
                    results: vec![I32],
                    intercepts: vec!["write".into(), "_write".into()],
                },
                FuncImport {
                    module: "env".into(),
                    name: "exit".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec!["exit".into(), "_exit".into(), "_Exit".into()],
                },
                FuncImport {
                    module: "env".into(),
                    name: "__speet_unreachable_trap".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec![],
                },
            ],
        }
    }

    /// RV64 native-syscall-lowering manifest: just `env.exit`/`env.write`, in
    /// that order (no `__speet_hint`) — a different, smaller import set from
    /// [`native_syscall`](Self::native_syscall), used by the RV64 `ecall`
    /// lowering path (`speet_syscall::WasmSyscallDispatcher`), which has its
    /// own dedicated 32-bit-memory module shape. Keep this a distinct
    /// manifest rather than reusing `native_syscall()` and ignoring the
    /// extra import — an unused import would still occupy a WASM index the
    /// RV64 shim never defines a matching symbol for.
    pub fn rv64_syscall() -> Self {
        use WasmValType::I32;
        Self {
            func_imports: vec![
                FuncImport {
                    module: "env".into(),
                    name: "exit".into(),
                    params: vec![I32],
                    results: vec![],
                    intercepts: vec!["exit".into(), "_exit".into(), "_Exit".into()],
                },
                FuncImport {
                    module: "env".into(),
                    name: "write".into(),
                    params: vec![I32, I32, I32],
                    results: vec![I32],
                    intercepts: vec!["write".into(), "_write".into()],
                },
            ],
        }
    }

    /// Find the func import whose `intercepts` list names `guest_symbol`
    /// (after stripping a leading Mach-O-style `_`), returning its
    /// `(module, name)` WASM-import identity. This is the single place that
    /// walks the manifest for redirect derivation — see
    /// `docs/guides/thin-runtime-genericity.md` principle 1 and
    /// [`HostApi::resolve_plt_redirect`](crate::HostApi::resolve_plt_redirect).
    pub fn resolve_intercept(&self, guest_symbol: &str) -> Option<(&str, &str)> {
        let bare = guest_symbol.strip_prefix('_').unwrap_or(guest_symbol);
        self.func_imports
            .iter()
            .find(|imp| {
                imp.intercepts.iter().any(|s| {
                    let s_bare = s.strip_prefix('_').unwrap_or(s.as_str());
                    s == guest_symbol || s_bare == bare
                })
            })
            .map(|imp| (imp.module.as_str(), imp.name.as_str()))
    }

    /// The WASM function-import index `(module, name)` would receive if
    /// this manifest's imports are registered in order (one host-capability
    /// slot each) — see `docs/guides/thin-runtime-genericity.md` principle 1.
    /// This is the position in `func_imports`, which callers must treat as
    /// the *only* source of truth for "which WASM import index is this,"
    /// never a separately hand-maintained constant.
    pub fn index_of(&self, module: &str, name: &str) -> Option<u32> {
        self.func_imports
            .iter()
            .position(|imp| imp.module == module && imp.name == name)
            .map(|i| i as u32)
    }

    /// Render a wasm-blitz external symbol name (`env__exit`).
    pub fn external_symbol(imp: &FuncImport) -> String {
        format!("{}__{}", imp.module, imp.name)
    }
}

/// Linker inputs derived from the host API backend.
#[derive(Debug, Clone)]
pub struct LinkRecipe {
    pub arch: BinArch,
    pub os: BinOs,
    /// `-lc`, `-lSystem`, etc.
    pub dylib_flags: Vec<String>,
    /// `(alias_emitted, real_host_symbol)` pairs for the linker.
    pub ambient_aliases: Vec<(String, String)>,
}

impl Default for LinkRecipe {
    fn default() -> Self {
        let (arch, os) = crate::host_link_target();
        Self {
            arch,
            os,
            dylib_flags: Vec::new(),
            ambient_aliases: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_of_matches_manifest_order() {
        let m = ImportManifest::integrated_native();
        assert_eq!(m.index_of("env", "__speet_hint"), Some(0));
        assert_eq!(m.index_of("env", "exit"), Some(1));
        assert_eq!(m.index_of("env", "write"), Some(2));
        assert_eq!(m.index_of("env", "__speet_log_unreachable"), Some(3));
        assert_eq!(m.index_of("env", "__speet_execve"), Some(4));
        assert_eq!(m.index_of("env", "__speet_stub_for_pc"), Some(5));
        assert_eq!(m.index_of("env", "printf"), Some(6));
        assert_eq!(m.index_of("env", "nonexistent"), None);
    }

    #[test]
    fn corpus_harness_trap_import_index() {
        let m = ImportManifest::corpus_harness();
        assert_eq!(m.index_of("env", "__speet_hint"), Some(0));
        assert_eq!(m.index_of("env", "write"), Some(1));
        assert_eq!(m.index_of("env", "exit"), Some(2));
        assert_eq!(m.index_of("env", "__speet_unreachable_trap"), Some(3));
        assert_eq!(m.func_imports.len(), 4);
    }

    #[test]
    fn rv64_syscall_is_exit_then_write() {
        let m = ImportManifest::rv64_syscall();
        assert_eq!(m.index_of("env", "exit"), Some(0));
        assert_eq!(m.index_of("env", "write"), Some(1));
        assert_eq!(m.func_imports.len(), 2);
    }

    #[test]
    fn resolve_intercept_normalizes_leading_underscore() {
        let m = ImportManifest::integrated_native();
        assert_eq!(m.resolve_intercept("exit"), Some(("env", "exit")));
        assert_eq!(m.resolve_intercept("_exit"), Some(("env", "exit")));
        assert_eq!(m.resolve_intercept("execve"), Some(("env", "__speet_execve")));
        assert_eq!(m.resolve_intercept("_execve"), Some(("env", "__speet_execve")));
        assert_eq!(m.resolve_intercept("printf"), Some(("env", "printf")));
        // Internal-only slots are never guest-symbol-addressable.
        assert_eq!(m.resolve_intercept("__speet_hint"), None);
    }
}
