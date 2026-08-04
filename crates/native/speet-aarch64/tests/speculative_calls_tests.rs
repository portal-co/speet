//! Tests for speculative BL/BLR/RET translations in the aarch64 recompiler.

use speet_aarch64::AArch64Recompiler;
use speet_link_core::ReactorAdapter;
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};

fn make_rctx(
    reactor: &mut Reactor<(), core::convert::Infallible, wasm_encoder::Function, LocalPool>,
) -> ReactorAdapter<'_, (), core::convert::Infallible, wasm_encoder::Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        injected_start: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool {
            handler: &T,
            ty: TypeIdx(0),
        },
        escape: yecta::CallEscape::Jump,
    }
}

#[test]
fn test_speculative_calls_disabled_by_default() {
    let recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_speculative_calls_toggle() {
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0);
    assert!(!recompiler.is_speculative_calls_enabled());
    recompiler.set_speculative_calls(true);
    assert!(recompiler.is_speculative_calls_enabled());
    recompiler.set_speculative_calls(false);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_escape_tag_configuration() {
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0);
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);

    assert_eq!(recompiler.get_escape_tag(&rctx), None);

    let tag = EscapeTag {
        tag: TagIdx(42),
        ty: TypeIdx(1),
    };
    recompiler.set_escape_tag(&mut rctx, Some(tag));
    assert_eq!(recompiler.get_escape_tag(&rctx), Some(tag));

    recompiler.set_escape_tag(&mut rctx, None);
    assert_eq!(recompiler.get_escape_tag(&rctx), None);
}

#[test]
fn test_bl_with_speculative_calls_disabled() {
    // BL +0 (self): 0x94000000
    let bytes = 0x94000000u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    assert!(!recompiler.is_speculative_calls_enabled());

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_bl_with_flag_escape() {
    let bytes = 0x94000000u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_bl_with_exception_escape() {
    let bytes = 0x94000000u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(
        &mut rctx,
        Some(EscapeTag {
            tag: TagIdx(0),
            ty: TypeIdx(0),
        }),
    );

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_ret_with_flag_escape() {
    // RET (alias of RET X30): 0xD65F03C0
    let bytes = 0xd65f03c0u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_ret_with_exception_escape() {
    let bytes = 0xd65f03c0u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(
        &mut rctx,
        Some(EscapeTag {
            tag: TagIdx(0),
            ty: TypeIdx(0),
        }),
    );

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_blr_with_flag_escape() {
    // BLR X0: 0xD63F0000
    let bytes = 0xd63f0000u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_speculative_calls_without_native_escape_uses_jump_path() {
    let bytes = 0x94000000u32.to_le_bytes().to_vec();
    let mut recompiler = AArch64Recompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);
    // escape remains Jump → non-speculative path

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_base_params_includes_expected_ra() {
    assert_eq!(
        AArch64Recompiler::<(), core::convert::Infallible>::BASE_PARAMS,
        73
    );
    assert_eq!(
        AArch64Recompiler::<(), core::convert::Infallible>::SP_PARAM_INDEX,
        72
    );
}
