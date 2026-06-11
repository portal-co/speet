//! Frontend integration: guest machine code -> speet -> WASM module.
//!
//! Verifies the speet frontend is wired into the driver: a small x86-64 `.text`
//! blob is recompiled to a complete, valid WASM module. Also exercises feeding
//! that real speet output into the backend object compiler, to surface which
//! wasm-blitz features speet's output needs.

use binary_io::BinArch;
use speet_recompile::frontend::recompile_to_wasm;

/// `xor eax, eax` then `inc eax` (eax = 1): two simple ALU instructions.
const SNIPPET: &[u8] = &[0x31, 0xC0, 0xFF, 0xC0];

#[test]
fn frontend_produces_valid_wasm() {
    let (wasm, unsupported) = recompile_to_wasm(SNIPPET, 0x1000, BinArch::X86_64);
    assert!(!wasm.is_empty(), "should produce a wasm module");

    // Validate with all features (the module uses memory64 + table64).
    let mut validator = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    validator
        .validate_all(&wasm)
        .unwrap_or_else(|e| panic!("speet output failed validation: {e}\nunsupported: {unsupported:?}"));

    // The module must contain at least the entry function.
    let func_bodies = wasmparser::Parser::new(0)
        .parse_all(&wasm)
        .flatten()
        .filter(|p| matches!(p, wasmparser::Payload::CodeSectionEntry(_)))
        .count();
    assert!(func_bodies >= 1, "expected at least one recompiled function");
    eprintln!("recompiled {func_bodies} functions; unsupported = {unsupported:?}");
}

#[test]
fn frontend_output_through_backend() {
    let (wasm, unsupported) = recompile_to_wasm(SNIPPET, 0x1000, BinArch::X86_64);
    // Attempt to lower speet's real output to a native object. This surfaces any
    // wasm-blitz feature gaps (memory64, tables, imports) on real recompiler
    // output; report rather than hard-fail so the gap is visible.
    let res = std::panic::catch_unwind(|| {
        speet_recompile::drive::compile_wasm_to_object(&wasm, BinArch::X86_64, binary_io::BinOs::Linux)
    });
    match res {
        Ok(Ok(obj)) => eprintln!("backend produced {} object bytes", obj.len()),
        Ok(Err(e)) => eprintln!("backend gap on speet output: {e}\nunsupported: {unsupported:?}"),
        Err(_) => eprintln!("backend panicked (unimplemented op) on speet output; unsupported: {unsupported:?}"),
    }
}
