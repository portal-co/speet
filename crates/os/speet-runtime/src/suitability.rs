//! Import-based suitability gate for the integrated thin runtime.

use binary_io::LoadedBinary;
use speet_host_api::HostApi;
use std::collections::HashSet;

/// Symbols allowed to be linked: tunnel-resolvable and known not to accept
/// function-pointer parameters.
fn fn_ptr_free_allowlist() -> HashSet<&'static str> {
    [
        "write",
        "_write",
        "read",
        "_read",
        "exit",
        "_exit",
        "close",
        "_close",
        "__stack_chk_fail",
        "__stack_chk_guard",
        "__libc_start_main",
        "_start",
        "main",
        "memset",
        "_memset",
        "memcpy",
        "_memcpy",
        "putchar",
        "_putchar",
        "strlen",
        "_strlen",
    ]
    .into_iter()
    .collect()
}

/// Normalize Mach-O underscore prefixes for allowlist lookup.
fn bare_import_name(name: &str) -> &str {
    name.strip_prefix('_').unwrap_or(name)
}

/// Linker/runtime imports that are not host tunnel symbols but are always
/// satisfied by the platform loader or our shim link.
fn is_linker_internal(name: &str) -> bool {
    matches!(
        name,
        "dyld_stub_binder" | "_dyld_stub_binder" | "__dso_handle" | "_dso_handle"
    )
}

/// Analyze a loaded binary's imports against the host tunnel and fn-ptr-free list.
pub fn analyze_imports(host: &dyn HostApi, bin: &LoadedBinary) -> (Vec<String>, Vec<String>) {
    let allow = fn_ptr_free_allowlist();
    let mut unresolved_deps = Vec::new();
    let mut fn_ptr_deps = Vec::new();

    for imp in &bin.imports {
        if is_linker_internal(&imp.name) {
            continue;
        }
        let bare = bare_import_name(&imp.name);
        if host.resolve_ambient(bare).is_none() && host.resolve_ambient(&imp.name).is_none() {
            unresolved_deps.push(imp.name.clone());
            continue;
        }
        let lookup = if host.resolve_ambient(bare).is_some() {
            bare
        } else {
            imp.name.as_str()
        };
        if !allow.contains(lookup) && !allow.contains(bare) && !allow.contains(imp.name.as_str()) {
            if speet_abi_stubs::has_stub(lookup)
                || speet_abi_stubs::has_stub(bare)
                || speet_abi_stubs::has_stub(imp.name.as_str())
            {
                continue;
            }
            fn_ptr_deps.push(imp.name.clone());
        }
    }

    unresolved_deps.sort();
    unresolved_deps.dedup();
    fn_ptr_deps.sort();
    fn_ptr_deps.dedup();
    (unresolved_deps, fn_ptr_deps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use binary_io::{BinArch, BinFormat, BinOs, ImportSym, LoadedBinary};
    use speet_host_api::TunneledHostApi;

    fn empty_bin(imports: Vec<ImportSym>) -> LoadedBinary {
        LoadedBinary {
            format: BinFormat::Elf,
            arch: BinArch::X86_64,
            os: BinOs::Linux,
            entry: 0,
            sections: vec![],
            symbols: vec![],
            relocs: vec![],
            dyn_deps: vec![],
            imports,
        }
    }

    #[test]
    fn write_and_exit_are_suitable() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![
            ImportSym {
                name: "write".into(),
                plt_addr: Some(0x1000),
            },
            ImportSym {
                name: "exit".into(),
                plt_addr: Some(0x1010),
            },
        ]);
        let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
        assert!(unresolved.is_empty());
        assert!(fn_ptr.is_empty());
    }

    #[test]
    fn write_is_suitable_via_abi_stub() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "write".into(),
            plt_addr: Some(0x1000),
        }]);
        let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
        assert!(unresolved.is_empty());
        assert!(fn_ptr.is_empty());
    }

    #[test]
    fn printf_is_suitable_via_abi_stub() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "printf".into(),
            plt_addr: Some(0x1000),
        }]);
        let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
        assert!(unresolved.is_empty());
        assert!(fn_ptr.is_empty());
    }

    #[test]
    fn unknown_sym_is_unresolved() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "not_a_real_symbol".into(),
            plt_addr: None,
        }]);
        let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
        assert_eq!(unresolved, vec!["not_a_real_symbol"]);
        assert!(fn_ptr.is_empty());
    }
}
