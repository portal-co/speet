//! Import manifest and link recipe types.

use binary_io::{BinArch, BinOs};

/// One WASM function import the link shim must satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncImport {
    pub module: String,
    pub name: String,
}

/// Imports and metadata needed to generate the C link shim.
#[derive(Debug, Clone, Default)]
pub struct ImportManifest {
    pub func_imports: Vec<FuncImport>,
}

impl ImportManifest {
    /// Standard native-syscall shim imports (`env.exit`, `env.write`, `env.__speet_hint`).
    pub fn native_syscall() -> Self {
        Self {
            func_imports: vec![
                FuncImport {
                    module: "env".into(),
                    name: "__speet_hint".into(),
                },
                FuncImport {
                    module: "env".into(),
                    name: "exit".into(),
                },
                FuncImport {
                    module: "env".into(),
                    name: "write".into(),
                },
            ],
        }
    }

    /// Integrated thin-runtime manifest: syscall imports + unreachable logging + hooks.
    pub fn integrated_native() -> Self {
        let mut m = Self::native_syscall();
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "__speet_log_unreachable".into(),
        });
        m.func_imports.push(FuncImport {
            module: "env".into(),
            name: "__speet_execve".into(),
        });
        m
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
