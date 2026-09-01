//! Import-based suitability gate for the integrated thin runtime.

use binary_io::LoadedBinary;
use speet_host_api::HostApi;
use speet_link_core::image_layout::{GuestImageLayout, MemoryModel, ZERO_OFFSET_MAX_DATA_END};
use std::collections::HashSet;

/// Symbols allowed to be linked: tunnel-resolvable, and every pointer
/// argument or return value they have is translated by hand-written glue in
/// `speet_rt::shim`'s `emit_import_stub` (`__wasm_mem`-relative input
/// pointers; host-owned return pointers copied into the reserved scratch
/// region rather than handed to the guest raw — see `getenv`'s stub and
/// `speet_rt::HOST_STR_SCRATCH_BYTES`'s doc comment for why a raw host
/// pointer would be unsafe). None of these ever take a function-pointer
/// argument (the original, narrower reason this list exists).
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
        "getenv",
        "_getenv",
    ]
    .into_iter()
    .collect()
}

/// Additional pointer-taking (but never function-pointer-taking) symbols
/// admitted only under [`MemoryModel::ZeroOffset`] — Phase 6 of the
/// aarch64-jit-parity/shared-memory/debug-seam plan.
///
/// Unlike [`fn_ptr_free_allowlist`]'s members, which need hand-written
/// `speet_rt::shim` glue (or an `os-abi-stubs` redirect) to stay safe under
/// *every* memory model, none of these have any translation glue at all:
/// under `ZeroOffset`, a guest pointer's raw numeric value is already valid
/// as a host address into the shared zero-offset memory region (see
/// `os_page::backends::SharedLinearHost`), so passing it straight through
/// to the real host libc implementation is safe by construction. Only
/// *function*-pointer arguments still need Phase 2's `CallRef`/`RefFunc`
/// machinery — none of these symbols take one.
///
/// Deliberately disjoint from `os-abi-stubs`' `STUB_SYMBOLS` (which are
/// already safe under *any* model via their generated redirect code, so
/// this list would never actually be reached for them — see
/// `analyze_imports_with_model`'s `has_wired_impl` check, which runs
/// unconditionally before this one). This list is for symbols with no
/// stub/glue of any kind, only the structural zero-offset guarantee.
fn zero_offset_pointer_allowlist() -> HashSet<&'static str> {
    [
        "strcmp", "_strcmp", "strncmp", "_strncmp", "memcmp", "_memcmp", "calloc", "_calloc",
        "realloc", "_realloc",
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
    analyze_imports_with_model(host, bin, MemoryModel::OwnedLinear)
}

/// Like [`analyze_imports`], but under [`MemoryModel::ZeroOffset`] also allows
/// BridgeSupport data-pointer stubs (`zero_offset_data_pointer_safe`) and
/// rejects images whose data span cannot fit the default host mirror.
pub fn analyze_imports_with_model(
    host: &dyn HostApi,
    bin: &LoadedBinary,
    model: MemoryModel,
) -> (Vec<String>, Vec<String>) {
    analyze_imports_with_model_and_wired(host, bin, model, &[])
}

fn extra_wires(name: &str, extra: &[String]) -> bool {
    let bare = bare_import_name(name);
    extra
        .iter()
        .any(|s| s == name || s == bare || bare_import_name(s) == bare)
}

/// Like [`analyze_imports_with_model`], but treats every name in `extra_wired`
/// as a hot-plugged stub (`has_wired_impl`) consulted before the statically
/// linked registry — used by `hot-recompiler` after loading `stubs.wasm`.
pub fn analyze_imports_with_model_and_wired(
    host: &dyn HostApi,
    bin: &LoadedBinary,
    model: MemoryModel,
    extra_wired: &[String],
) -> (Vec<String>, Vec<String>) {
    let allow = fn_ptr_free_allowlist();
    let mut unresolved_deps = Vec::new();
    let mut fn_ptr_deps = Vec::new();
    let zero = model.text_unmapped();

    if zero
        && bin
            .sections
            .iter()
            .any(|s| matches!(s.kind, binary_io::SectionKind::Text))
    {
        let layout = GuestImageLayout::from_loaded_binary(bin);
        if layout.data_end_max() > ZERO_OFFSET_MAX_DATA_END {
            unresolved_deps.push(format!(
                "zero-offset data span 0x{:x} exceeds mirror cap 0x{ZERO_OFFSET_MAX_DATA_END:x}",
                layout.data_end_max()
            ));
        }
    }

    for imp in &bin.imports {
        if is_linker_internal(&imp.name) {
            continue;
        }
        let bare = bare_import_name(&imp.name);
        // Hot-plugged stubs.wasm names are consulted first: a symbol the
        // agent rebuilt into the guest table is treated as wired even when
        // it is not yet in the statically linked registry or ambient set.
        if extra_wires(&imp.name, extra_wired) || extra_wires(bare, extra_wired) {
            continue;
        }
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
            if extra_wires(lookup, extra_wired)
                || extra_wires(bare, extra_wired)
                || extra_wires(&imp.name, extra_wired)
                || speet_abi_stubs::has_wired_impl(lookup)
                || speet_abi_stubs::has_wired_impl(bare)
                || speet_abi_stubs::has_wired_impl(imp.name.as_str())
                || os_shim_core::has_core_impl(lookup)
                || os_shim_core::has_core_impl(bare)
                || os_shim_core::has_core_impl(imp.name.as_str())
            {
                continue;
            }
            if zero
                && (speet_abi_stubs::zero_offset_data_pointer_safe(lookup)
                    || speet_abi_stubs::zero_offset_data_pointer_safe(bare)
                    || speet_abi_stubs::zero_offset_data_pointer_safe(imp.name.as_str()))
            {
                continue;
            }
            if zero {
                let extra = zero_offset_pointer_allowlist();
                if extra.contains(lookup)
                    || extra.contains(bare)
                    || extra.contains(imp.name.as_str())
                {
                    continue;
                }
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

    #[test]
    fn extra_wired_symbol_is_not_unresolved() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "not_a_real_symbol".into(),
            plt_addr: None,
        }]);
        let (unresolved, fn_ptr) = analyze_imports_with_model_and_wired(
            &host,
            &bin,
            MemoryModel::OwnedLinear,
            &["not_a_real_symbol".into()],
        );
        assert!(unresolved.is_empty(), "{unresolved:?}");
        assert!(fn_ptr.is_empty(), "{fn_ptr:?}");
    }

    #[test]
    fn zero_offset_admits_data_pointer_stub_metadata() {
        assert!(speet_abi_stubs::zero_offset_data_pointer_safe("write"));
        assert!(speet_abi_stubs::zero_offset_data_pointer_safe("exit"));
        assert!(!speet_abi_stubs::zero_offset_data_pointer_safe("printf"));
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "write".into(),
            plt_addr: Some(0x1000),
        }]);
        let (unresolved, fn_ptr) = analyze_imports_with_model(&host, &bin, MemoryModel::ZeroOffset);
        assert!(unresolved.is_empty());
        assert!(fn_ptr.is_empty());
    }

    /// Phase 6: `strcmp` has no `os-abi-stubs` entry and no `speet_rt::shim`
    /// glue -- it's admitted purely because it's on the new
    /// `zero_offset_pointer_allowlist`, and only under `ZeroOffset`.
    #[test]
    fn zero_offset_admits_plain_pointer_taking_symbol_with_no_stub_at_all() {
        let host = TunneledHostApi::for_host();
        let bin = empty_bin(vec![ImportSym {
            name: "strcmp".into(),
            plt_addr: Some(0x1000),
        }]);

        // Under the default (OwnedLinear) model, a raw guest pointer isn't
        // trustworthy as a host address, so this is correctly rejected.
        let (unresolved, fn_ptr) = analyze_imports(&host, &bin);
        assert!(
            unresolved.is_empty(),
            "strcmp should still resolve against the host ambient table"
        );
        assert_eq!(
            fn_ptr,
            vec!["strcmp"],
            "strcmp must be rejected under OwnedLinear -- no glue exists for it"
        );

        // Under ZeroOffset, the guest pointer's numeric value is already a
        // valid host address, so strcmp becomes suitable with zero
        // translation glue.
        let (unresolved, fn_ptr) = analyze_imports_with_model(&host, &bin, MemoryModel::ZeroOffset);
        assert!(unresolved.is_empty());
        assert!(
            fn_ptr.is_empty(),
            "strcmp should be admitted under ZeroOffset via zero_offset_pointer_allowlist"
        );
    }

    #[test]
    fn zero_offset_pointer_allowlist_does_not_leak_into_owned_linear() {
        let host = TunneledHostApi::for_host();
        for sym in ["strncmp", "memcmp", "calloc", "realloc"] {
            let bin = empty_bin(vec![ImportSym {
                name: sym.into(),
                plt_addr: Some(0x1000),
            }]);
            let (_, fn_ptr) = analyze_imports(&host, &bin);
            assert_eq!(
                fn_ptr,
                vec![sym.to_string()],
                "{sym} must stay rejected under OwnedLinear"
            );
            let (_, fn_ptr) = analyze_imports_with_model(&host, &bin, MemoryModel::ZeroOffset);
            assert!(fn_ptr.is_empty(), "{sym} must be admitted under ZeroOffset");
        }
    }
}
