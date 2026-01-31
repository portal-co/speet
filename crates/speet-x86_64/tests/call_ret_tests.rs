//! Tests for call/ret instruction translations in x86_64 recompiler

use speet_x86_64::X86Recompiler;

#[test]
fn test_call_direct_translation() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    // This calls 5 bytes ahead from the end of the instruction
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, _> = X86Recompiler::new();

    let mut ctx = ();
    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            wasm_encoder::Function::new(locals.collect::<Vec<_>>())
        },
    );

    assert!(result.is_ok());
}

#[test] 
fn test_ret_translation() {
    // RET instruction: C3
    let bytes = vec![0xC3];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, _> = X86Recompiler::new();

    let mut ctx = ();
    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            wasm_encoder::Function::new(locals.collect::<Vec<_>>())
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_ret_with_immediate_translation() {
    // RET with immediate: C2 08 00 (ret 8) - return and clean up 8 bytes from stack
    let bytes = vec![0xC2, 0x08, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, _> = X86Recompiler::new();

    let mut ctx = ();
    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            wasm_encoder::Function::new(locals.collect::<Vec<_>>())
        },
    );

    assert!(result.is_ok());
}

#[test]
fn test_call_ret_sequence() {
    // Test a sequence: CALL followed by RET
    // CALL +5: E8 05 00 00 00
    // NOP: 90 (just to have something to call to)
    // RET: C3
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00, 0x90, 0xC3];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, _> = X86Recompiler::new();

    let mut ctx = ();
    let result = recompiler.translate_bytes(
        &mut ctx,
        &bytes,
        0x1000,
        &mut |locals| {
            wasm_encoder::Function::new(locals.collect::<Vec<_>>())
        },
    );

    assert!(result.is_ok());
}