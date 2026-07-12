//! Structured ABI model produced from BridgeSupport and future formats.

use speet_host_api::{FuncImport, ImportManifest, WasmValType};

/// High-level classification of an argument or return value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiValueKind {
    Void,
    Scalar,
    Pointer,
    FunctionPointer,
    /// Objective-C object / id-like pointer (BridgeSupport `type="@..."`).
    Object,
    /// Unrecognized or unsupported BridgeSupport encoding.
    Unknown(String),
}

/// One parameter or return slot in an ABI function signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiArg {
    pub kind: AbiValueKind,
    /// BridgeSupport `type` attribute when present (e.g. `i`, `*`, `^?`).
    pub bridgesupport_type: Option<String>,
    pub function_pointer: bool,
    pub pointer: bool,
}

impl AbiArg {
    pub fn void() -> Self {
        Self {
            kind: AbiValueKind::Void,
            bridgesupport_type: None,
            function_pointer: false,
            pointer: false,
        }
    }
}

/// One C/Objective-C callable symbol from an ABI description file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiFunction {
    pub name: String,
    pub args: Vec<AbiArg>,
    pub retval: AbiArg,
    pub variadic: bool,
}

/// Parsed ABI description for one framework or library surface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbiSpec {
    pub functions: Vec<AbiFunction>,
}

impl AbiSpec {
    pub fn lookup(&self, name: &str) -> Option<&AbiFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Whether `name` accepts at least one function-pointer argument.
    pub fn accepts_function_pointers(&self, name: &str) -> bool {
        self.lookup(name)
            .map(|f| {
                f.args
                    .iter()
                    .any(|a| a.function_pointer || a.kind == AbiValueKind::FunctionPointer)
            })
            .unwrap_or(false)
    }

    /// Symbols whose signatures include function-pointer parameters — the
    /// set `speet-runtime::suitability` will eventually move off the blanket
    /// deny path once codegen exists for them.
    pub fn fn_ptr_symbols(&self) -> Vec<&str> {
        self.functions
            .iter()
            .filter(|f| {
                f.args.iter().any(|a| {
                    a.function_pointer || a.kind == AbiValueKind::FunctionPointer
                })
            })
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Phase 1 helper: derive [`ImportManifest`] intercept metadata for
    /// symbols declared in this spec. Only symbols with simple scalar/pointer
    /// signatures and no function-pointer args become tunnel imports today;
    /// fn-ptr symbols are omitted until Phase 2 codegen exists.
    ///
    /// `base` supplies the always-present syscall/hint imports shared by every
    /// thin-runtime module; spec-derived imports are appended after it.
    pub fn extend_manifest(base: &ImportManifest, spec: &AbiSpec, symbols: &[&str]) -> ImportManifest {
        let mut m = base.clone();
        for &sym in symbols {
            let Some(f) = spec.lookup(sym) else {
                continue;
            };
            if spec.accepts_function_pointers(sym) && !speet_abi_stubs::has_stub(sym) {
                continue;
            }
            if m.func_imports.iter().any(|imp| {
                imp.intercepts.iter().any(|s| s == sym || s.strip_prefix('_') == Some(sym))
            }) {
                continue;
            }
            let (params, results) = bridgesupport_wasm_types(f);
            m.func_imports.push(FuncImport {
                module: "env".into(),
                name: sym.to_string(),
                params,
                results,
                intercepts: intercept_names(sym),
            });
        }
        m
    }
}

fn intercept_names(sym: &str) -> Vec<String> {
    let mut out = vec![sym.to_string()];
    if !sym.starts_with('_') {
        out.push(format!("_{sym}"));
    }
    out
}

/// Best-effort BridgeSupport → WASM type mapping for manifest construction.
/// Phase 2 codegen will replace this coarse mapping for redirect stubs.
fn bridgesupport_wasm_types(f: &AbiFunction) -> (Vec<WasmValType>, Vec<WasmValType>) {
    use WasmValType::{I32, I64};
    let params = f
        .args
        .iter()
        .map(|a| match a.bridgesupport_type.as_deref() {
            Some("q" | "Q" | "d" | "D") => I64,
            _ => I32,
        })
        .collect();
    let results = if f.retval.kind == AbiValueKind::Void {
        vec![]
    } else {
        vec![match f.retval.bridgesupport_type.as_deref() {
            Some("q" | "Q" | "d" | "D") => I64,
            _ => I32,
        }]
    };
    (params, results)
}
