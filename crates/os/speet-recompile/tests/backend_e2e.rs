//! End-to-end backend test: WASM -> native object -> system link -> run.
//!
//! Validates the recompiler's backend half together with `binary-io` and the
//! C-ABI entry bridge (STATUS.md #2): a WASM function returning a constant is
//! compiled to a relocatable object, linked with a tiny C shim that calls
//! `__guest_entry`, and executed — its return value becomes the process exit
//! code. Gated to the host platform (v1 is same-platform).

#![cfg(target_os = "macos")]
//
// Primary execute path on Apple Silicon: `*_aarch64_native` tests (blitz →
// asm-arch `AArch64Writer` → clang -arch arm64 → native spawn). x86_64-via-Rosetta
// execute tests remain `#[ignore]` as an optional secondary lane. See STATUS.md.

use binary_io::{BinArch, BinOs};
use portal_solutions_blitz_common::wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, Instruction, Module, RefType, TableSection, TableType, TypeSection, ValType,
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

/// A floating-point module exercising the scalar FP backend end-to-end. The
/// entry `() -> i64` computes, with no imports:
///   a = sqrt(i64_to_f64(7) / 2.0 + 0.5) = sqrt(4.0) = 2.0
///   b = promote(3.0f * 3.0f)            = 9.0
///   trunc_s(a + b) = trunc(11.0) = 11
/// covering i64->f64 convert, fdiv, fadd, fsqrt, f32 fmul, f64 promote, and
/// f64->i64 truncation.
fn fp_module() -> Vec<u8> {
    use portal_solutions_blitz_common::wasm_encoder::Ieee32;
    use portal_solutions_blitz_common::wasm_encoder::Ieee64;
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
    f0.instruction(&Instruction::I64Const(7));
    f0.instruction(&Instruction::F64ConvertI64S); // 7.0
    f0.instruction(&Instruction::F64Const(Ieee64::from(2.0f64)));
    f0.instruction(&Instruction::F64Div); // 3.5
    f0.instruction(&Instruction::F64Const(Ieee64::from(0.5f64)));
    f0.instruction(&Instruction::F64Add); // 4.0
    f0.instruction(&Instruction::F64Sqrt); // 2.0
    f0.instruction(&Instruction::F32Const(Ieee32::from(3.0f32)));
    f0.instruction(&Instruction::F32Const(Ieee32::from(3.0f32)));
    f0.instruction(&Instruction::F32Mul); // 9.0f
    f0.instruction(&Instruction::F64PromoteF32); // 9.0
    f0.instruction(&Instruction::F64Add); // 11.0
    f0.instruction(&Instruction::I64TruncF64S); // 11
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    module.section(&code);
    module.finish()
}

/// FP comparison module: `() -> i64` returning `(2<3) + (3>=3) + (5<1) = 2`,
/// exercising lt (true), ge (equal boundary, true) and lt (false) — i.e. the
/// hand-derived FP condition-code mappings.
fn fp_cmp_module() -> Vec<u8> {
    use portal_solutions_blitz_common::wasm_encoder::Ieee64;
    let c = |x: f64| Instruction::F64Const(Ieee64::from(x));
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
    f0.instruction(&c(2.0));
    f0.instruction(&c(3.0));
    f0.instruction(&Instruction::F64Lt); // 1
    f0.instruction(&c(3.0));
    f0.instruction(&c(3.0));
    f0.instruction(&Instruction::F64Ge); // 1 (equal boundary)
    f0.instruction(&Instruction::I32Add); // 2
    f0.instruction(&c(5.0));
    f0.instruction(&c(1.0));
    f0.instruction(&Instruction::F64Lt); // 0
    f0.instruction(&Instruction::I32Add); // 2
    f0.instruction(&Instruction::I64ExtendI32S);
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    module.section(&code);
    module.finish()
}

/// AArch64 native: FP comparisons evaluate to exit code 2 (see [`fp_cmp_module`]).
#[cfg(target_arch = "aarch64")]
#[test]
fn fp_cmp_aarch64_native() {
    let wasm = fp_cmp_module();
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(&wasm).expect("validates");
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("object");
    let dir = std::env::temp_dir().join(format!("speet_a64fc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("g.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "extern long __guest_entry(void);\nint main(void){ return (int)__guest_entry(); }\n",
    )
    .unwrap();
    let link = Command::new("clang")
        .args(["-arch", "arm64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run");
    assert_eq!(run.code(), Some(2), "FP comparisons should evaluate to 2");
    let _ = std::fs::remove_dir_all(&dir);
}

/// AArch64 native: the scalar FP pipeline, run on this host. Asserts the FP
/// expression in [`fp_module`] evaluates to exit code 11.
#[cfg(target_arch = "aarch64")]
#[test]
fn fp_arith_aarch64_native() {
    let wasm = fp_module();
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(&wasm).expect("validates");
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("object");

    let dir = std::env::temp_dir().join(format!("speet_a64fp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("g.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "extern long __guest_entry(void);\nint main(void){ return (int)__guest_entry(); }\n",
    )
    .unwrap();
    let link = Command::new("clang")
        .args(["-arch", "arm64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run");
    assert_eq!(run.code(), Some(11), "FP expression should evaluate to 11");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `call_indirect` module exercising the native `__wasm_table`. Three internal
/// functions of type `(i32)->i64` (`x+10`, `x+20`, `x+30`); the entry dispatches
/// through the table by index. With no imports, the identity table maps slot i to
/// `__wasm_func_i`, so calling index `which` runs `f{which}`.
/// f0 (entry, ()->i64): i32.const 5 ; i32.const `which` ; call_indirect t0 ; return
fn call_indirect_module(which: i32) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], [ValType::I64]); // type 0: (i32)->i64 (table fns)
    types.ty().function([], [ValType::I64]); // type 1: entry ()->i64
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(1); // f0 entry
    functions.function(0); // f1
    functions.function(0); // f2
    functions.function(0); // f3
    module.section(&functions);
    // A funcref table of 4 identity slots, initialised by an active element.
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        table64: false,
        minimum: 4,
        maximum: Some(4),
        shared: false,
    });
    module.section(&tables);
    let mut exports = ExportSection::new();
    exports.export("f0", ExportKind::Func, 0);
    module.section(&exports);
    let mut elems = ElementSection::new();
    elems.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions((&[0u32, 1, 2, 3][..]).into()));
    module.section(&elems);
    let mut code = CodeSection::new();
    let mut f0 = Function::new([]);
    f0.instruction(&Instruction::I32Const(5)); // arg
    f0.instruction(&Instruction::I32Const(which)); // table index (= func index)
    f0.instruction(&Instruction::CallIndirect { type_index: 0, table_index: 0 });
    f0.instruction(&Instruction::Return);
    f0.instruction(&Instruction::End);
    code.function(&f0);
    for add in [10i64, 20, 30] {
        let mut f = Function::new([]);
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::I64ExtendI32S);
        f.instruction(&Instruction::I64Const(add));
        f.instruction(&Instruction::I64Add);
        f.instruction(&Instruction::Return);
        f.instruction(&Instruction::End);
        code.function(&f);
    }
    module.section(&code);
    module.finish()
}

/// AArch64 native: dispatch through the recompiled `__wasm_table`. Func index 2
/// is `f2` (the second table function, `x + 20`), so `f2(5) = 25`. Validates C1
/// (data-section Abs64 relocs) + C2 (`__wasm_table`/`__wasm_func_N` emission) +
/// C3 (indirect codegen) end-to-end.
#[cfg(target_arch = "aarch64")]
#[test]
fn call_indirect_dispatch_aarch64_native() {
    let wasm = call_indirect_module(2);
    let mut v = wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all());
    v.validate_all(&wasm).expect("validates");
    let obj = compile_wasm_to_object(&wasm, BinArch::AArch64, BinOs::MacOs).expect("object");

    let dir = std::env::temp_dir().join(format!("speet_a64ci_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let obj_path = dir.join("g.o");
    let shim_path = dir.join("shim.c");
    let exe_path = dir.join("exe");
    std::fs::write(&obj_path, &obj).unwrap();
    std::fs::write(
        &shim_path,
        "extern long __guest_entry(void);\nint main(void){ return (int)__guest_entry(); }\n",
    )
    .unwrap();
    let link = Command::new("clang")
        .args(["-arch", "arm64"]).arg(&shim_path).arg(&obj_path).arg("-o").arg(&exe_path)
        .output().expect("clang");
    assert!(link.status.success(), "link:\n{}", String::from_utf8_lossy(&link.stderr));
    let run = Command::new(&exe_path).status().expect("run");
    assert_eq!(run.code(), Some(25), "call_indirect index 2 should run f2(5)=25");
    let _ = std::fs::remove_dir_all(&dir);
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
