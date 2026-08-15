//! Proves wasm-blitz's in-process codegen produces genuinely executable
//! native machine code: a WASM function is compiled to raw AArch64 SysV
//! bytes and called directly through a function pointer, with no object
//! file, external linker, or WASM engine involved anywhere in the loop.

use speet_native_jit::{ExecutableCode, compile_native_arm64};
use wasm_encoder::{CodeSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType};

/// `f(a, b) = a*a + b`.
fn build_module() -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function([ValType::I64, ValType::I64], [ValType::I64]);

    let mut functions = FunctionSection::new();
    functions.function(0);

    let mut f = Function::new([]);
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::I64Mul);
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::I64Add);
    f.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&f);

    let mut module = Module::new();
    module.section(&types).section(&functions).section(&code);
    module.finish()
}

#[test]
fn wasm_blitz_compiled_native_code_executes_correctly() {
    let wasm = build_module();
    let native = compile_native_arm64(&wasm);
    assert!(!native.is_empty(), "should produce nonempty native machine code");

    let code = ExecutableCode::new(&native);
    for (a, b) in [(3i64, 4i64), (0, 0), (-5, 100), (1000, -1)] {
        let got = unsafe { code.call_i64_i64_to_i64(a, b) };
        assert_eq!(got, a * a + b, "a={a} b={b}");
    }
}
