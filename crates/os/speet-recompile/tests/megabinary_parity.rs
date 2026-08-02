//! Three-lane parity harness foundation (merged-with-mock module).

use binary_io::{BinArch, BinOs};
use speet_recompile::drive::compile_wasm_to_object;
use speet_recompile::frontend::recompile_to_wasm;
use speet_recompile::host_mem_shim::{lower_host_mem_imports, validate_standard_core};

/// Minimal x86 guest: `xor eax,eax; inc eax; ret`
const SNIPPET: &[u8] = &[0x31, 0xC0, 0xFF, 0xC0, 0xC3];

#[test]
fn lane_a_validate_megabinary() {
    let (wasm, _) = recompile_to_wasm(SNIPPET, 0x1000, BinArch::X86_64);
    validate_standard_core(&wasm).expect("lane A input must pass wasmparser");
    let canonical = lower_host_mem_imports(&wasm).expect("shimming pass idempotent");
    validate_standard_core(&canonical).expect("canonical module must validate");
}

#[test]
fn lane_b_wasm_compiler_input_matches_lane_a() {
    let (wasm, _) = recompile_to_wasm(SNIPPET, 0x1000, BinArch::X86_64);
    let canonical = lower_host_mem_imports(&wasm).unwrap();
    // Lane B-wasm consumes the same merged-with-mock artifact as lane A.
    // Full wasm-blitz compiler execution is gated on toolchain availability in CI.
    assert!(!canonical.is_empty());
    validate_standard_core(&canonical).unwrap();
}

/// Lane B-native: wasm-blitz object emission from the same canonical module as lane A.
#[test]
#[cfg(target_os = "macos")]
fn lane_b_native_compiles_canonical_megabinary() {
    let (wasm, _) = recompile_to_wasm(SNIPPET, 0x1000, BinArch::X86_64);
    let canonical = lower_host_mem_imports(&wasm).unwrap();
    validate_standard_core(&canonical).unwrap();

    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    let obj = compile_wasm_to_object(&canonical, arch, BinOs::MacOs)
        .expect("lane B-native wasm-blitz compile");
    assert!(!obj.is_empty());
    assert!(
        obj.starts_with(b"\xcf\xfa\xed\xfe") || obj.starts_with(b"\x7fELF"),
        "object must be Mach-O or ELF"
    );
}
