//! Tests for speculative call/ret instruction translations in x86_64 recompiler

use speet_x86_64::X86Recompiler;
use yecta::{EscapeTag, TagIdx, TypeIdx};

#[test]
fn test_speculative_calls_disabled_by_default() {
    let recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_speculative_calls_toggle() {
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Initially disabled
    assert!(!recompiler.is_speculative_calls_enabled());

    // Enable
    recompiler.set_speculative_calls(true);
    assert!(recompiler.is_speculative_calls_enabled());

    // Disable
    recompiler.set_speculative_calls(false);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_escape_tag_configuration() {
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Initially None
    assert_eq!(recompiler.get_escape_tag(), None);

    // Set escape tag
    let tag = EscapeTag {
        tag: TagIdx(42),
        ty: TypeIdx(1),
    };
    recompiler.set_escape_tag(Some(tag));
    assert_eq!(recompiler.get_escape_tag(), Some(tag));

    // Clear escape tag
    recompiler.set_escape_tag(None);
    assert_eq!(recompiler.get_escape_tag(), None);
}

#[test]
fn test_call_with_speculative_calls_disabled() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Speculative calls disabled (default)
    assert!(!recompiler.is_speculative_calls_enabled());

    let mut ctx = ();
    let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_call_with_speculative_calls_enabled() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);
    recompiler.set_escape_tag(Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let mut ctx = ();
    let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_ret_with_speculative_calls_enabled() {
    // RET instruction: C3
    let bytes = vec![0xC3];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);
    recompiler.set_escape_tag(Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let mut ctx = ();
    let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_speculative_calls_requires_escape_tag() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Enable speculative calls but don't set escape tag
    recompiler.set_speculative_calls(true);
    // escape_tag remains None

    let mut ctx = ();
    let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    // Should still work but use non-speculative path
    assert!(result.is_ok());
}

#[test]
fn test_ret_with_immediate_speculative() {
    // RET with immediate: C2 08 00 (ret 8) - return and clean up 8 bytes from stack
    let bytes = vec![0xC2, 0x08, 0x00];
    let mut recompiler: X86Recompiler<(), core::convert::Infallible, wasm_encoder::Function> =
        X86Recompiler::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);
    recompiler.set_escape_tag(Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let mut ctx = ();
    let result = recompiler.translate_bytes(&mut ctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}
