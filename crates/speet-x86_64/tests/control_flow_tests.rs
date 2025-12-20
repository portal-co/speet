//! Tests for control flow instructions in x86_64 recompiler

use speet_x86_64::X86Recompiler;

#[test]
fn test_test_instruction() {
    // Test: TEST rax, 0xFF - test lowest byte
    let bytes = vec![0x48, 0xA9, 0xFF, 0x00, 0x00, 0x00]; // TEST rax, 0xFF
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_cmp_instruction() {
    // Test: CMP rax, 0x10 - compare rax with 0x10
    let bytes = vec![0x48, 0x83, 0xF8, 0x10]; // CMP rax, 0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_jmp_instruction() {
    // Test: JMP +0x10 - unconditional jump forward
    let bytes = vec![0xEB, 0x10]; // JMP short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_je_instruction() {
    // Test: JE +0x10 - jump if equal (ZF set)
    let bytes = vec![0x74, 0x10]; // JE short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_jne_instruction() {
    // Test: JNE +0x10 - jump if not equal (ZF clear)
    let bytes = vec![0x75, 0x10]; // JNE short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_jl_instruction() {
    // Test: JL +0x10 - jump if less (SF != OF)
    let bytes = vec![0x7C, 0x10]; // JL short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_jg_instruction() {
    // Test: JG +0x10 - jump if greater (ZF=0 and SF=OF)
    let bytes = vec![0x7F, 0x10]; // JG short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_jb_instruction() {
    // Test: JB +0x10 - jump if below (CF set)
    let bytes = vec![0x72, 0x10]; // JB short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_ja_instruction() {
    // Test: JA +0x10 - jump if above (CF=0 and ZF=0)
    let bytes = vec![0x77, 0x10]; // JA short +0x10
    let mut recompiler: X86Recompiler<(), _> = X86Recompiler::new();
    let mut ctx = ();

    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            let mut func = wasm_encoder::Function::new(locals.collect::<Vec<_>>());
            func
        },
    );

    assert!(result.is_ok());
}