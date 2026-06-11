//! End-to-end backend test: WASM -> native object -> system link -> run.
//!
//! Validates the recompiler's backend half together with `binary-io` and the
//! C-ABI entry bridge (STATUS.md #2): a WASM function returning a constant is
//! compiled to a relocatable object, linked with a tiny C shim that calls
//! `__guest_entry`, and executed — its return value becomes the process exit
//! code. Gated to the host platform (v1 is same-platform).

#![cfg(target_os = "macos")]
//
// Note: we target x86_64 here (run via Rosetta on Apple silicon). The aarch64
// path currently faults on real hardware: blitz's aarch64 backend uses the
// hardware SP as the WASM operand stack with 8-byte pushes, which violates the
// arm64 16-byte SP-alignment check enforced on macOS (but not by Unicorn). See
// STATUS.md.

use binary_io::{BinArch, BinOs};
use portal_solutions_blitz_common::wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};
use speet_recompile::drive::compile_wasm_to_object;
use std::process::Command;

/// Build a module with one exported function `() -> i64` returning `val`.
fn const_return_module(val: i64) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], [ValType::I64]);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::I64Const(val));
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    module.section(&code);
    module.finish()
}

#[test]
fn recompiled_const_function_sets_exit_code() {
    let wasm = const_return_module(42);
    let obj = compile_wasm_to_object(&wasm, BinArch::X86_64, BinOs::MacOs)
        .expect("compile to object");

    let dir = std::env::temp_dir().join(format!("speet_be_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("guest.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("guest_exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "extern long __guest_entry(void);\nint main(void){ return (int)__guest_entry(); }\n",
    )
    .unwrap();

    let link = Command::new("clang")
        .arg("-arch")
        .arg("x86_64")
        .arg(&shim_path)
        .arg(&obj_path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .expect("run clang");
    assert!(
        link.status.success(),
        "link failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let run = Command::new(&exe_path).status().expect("run recompiled binary");
    assert_eq!(run.code(), Some(42), "recompiled guest should exit with its return value");

    let _ = std::fs::remove_dir_all(&dir);
}
