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

/// Two internal functions with a `return_call` that forwards a 1-param register
/// file, plus an `env.exit` import call. Isolates the `CallAbi::AllStack`
/// marshalling (internal call) + the C-ABI import call from speet's output.
/// f0 (entry): i64.const 42 ; return_call f1
/// f1(i64):    local.get 0 ; i32.wrap_i64 ; call env.exit ; unreachable
fn return_call_module() -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], []); // type 0: exit(i32)
    types.ty().function([ValType::I64], []); // type 1: internal (i64)->()
    module.section(&types);
    let mut imports = wasm_encoder::ImportSection::new();
    imports.import("env", "exit", wasm_encoder::EntityType::Function(0));
    module.section(&imports);
    let mut functions = FunctionSection::new();
    functions.function(1); // f0 (wasm idx 1)
    functions.function(1); // f1 (wasm idx 2)
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 1);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::I64Const(42));
    f0.instruction(&Instruction::ReturnCall(2)); // tail-call f1 with arg 42
    f0.instruction(&Instruction::End);
    code.function(&f0);
    let mut f1 = Function::new([]);
    f1.instruction(&Instruction::LocalGet(0));
    f1.instruction(&Instruction::I32WrapI64);
    f1.instruction(&Instruction::Call(0)); // env.exit
    f1.instruction(&Instruction::Unreachable);
    f1.instruction(&Instruction::End);
    code.function(&f1);
    module.section(&code);
    module.finish()
}

// Runs an x86_64 binary via Rosetta; the marshalling itself is verified by
// disassembly (STATUS.md). Ignored by default because executing requires a
// healthy Rosetta — re-enable to run on an x86 host or after a Rosetta reset.
#[test]
#[ignore = "executes x86_64 via Rosetta (run on an x86 host or after Rosetta reset)"]
fn allstack_return_call_marshalling() {
    let wasm = return_call_module();
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(&wasm).expect("validates");
    let obj = compile_wasm_to_object(&wasm, BinArch::X86_64, BinOs::MacOs).expect("object");

    let dir = std::env::temp_dir().join(format!("speet_rc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("g.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "#include <unistd.h>\nvoid env__exit(int c){_exit(c);}\nextern long __guest_entry();\nint main(void){__guest_entry();return 0;}\n",
    )
    .unwrap();
    let link = Command::new("clang")
        .args(["-arch", "x86_64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run");
    eprintln!("status={run:?}");
    assert_eq!(run.code(), Some(42), "return_call marshalling should deliver arg → exit(42)");
    let _ = std::fs::remove_dir_all(&dir);
}

/// AArch64 native: the `CallAbi::AllStack` tail-call marshalling + C-ABI import
/// call, run on this host. f0 `return_call`s f1(42); f1 calls `env.exit(42)`.
/// External symbols are loaded via ADRP+ADD (PAGE21/PAGEOFF12) so the object
/// links against the shim's `env__exit` on Mach-O.
#[cfg(target_arch = "aarch64")]
#[test]
fn allstack_return_call_marshalling_aarch64_native() {
    let wasm = return_call_module();
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(&wasm).expect("validates");
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("object");

    let dir = std::env::temp_dir().join(format!("speet_a64rc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("g.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "#include <unistd.h>\nvoid env__exit(int c){_exit(c);}\nextern long __guest_entry();\nint main(void){__guest_entry();return 0;}\n",
    )
    .unwrap();
    let link = Command::new("clang")
        .args(["-arch", "arm64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run");
    assert_eq!(run.code(), Some(42), "return_call marshalling should deliver arg → exit(42)");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "executes x86_64 via Rosetta (run on an x86 host or after Rosetta reset)"]
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

/// AArch64 native output, end-to-end on this host: compile -> link (native
/// arm64) -> run -> exit code. This is the same-platform M1 artifact for the
/// dev host (macOS/arm64). It exercises the 16-byte-slot operand stack that
/// keeps the hardware SP 16-byte aligned (macOS faults on a misaligned SP);
/// before that fix this faulted with SIGBUS.
#[cfg(target_arch = "aarch64")]
#[test]
fn recompiled_const_function_aarch64_native() {
    let wasm = const_return_module(42);
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs)
        .expect("compile to aarch64 object");

    let dir = std::env::temp_dir().join(format!("speet_a64_{}", std::process::id()));
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
        .args(["-arch", "arm64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("run clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run recompiled binary");
    assert_eq!(run.code(), Some(42), "recompiled aarch64 guest should exit with its return value");
    let _ = std::fs::remove_dir_all(&dir);
}
