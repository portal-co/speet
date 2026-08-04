//! Unit tests: decode + translate tiny A32 sequences without panic.

use speet_arm::ArmRecompiler;
use speet_link_core::ReactorAdapter;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

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
fn mov_r0_imm1_bx_lr_translates() {
    // MOV r0, #1 ; BX lr
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0xe3a0_0001u32.to_le_bytes());
    bytes.extend_from_slice(&0xe12f_ff1eu32.to_le_bytes());

    let mut rc = ArmRecompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    rc.setup_traps(&mut rctx, &mut ctx);
    let n = rc
        .translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
            wasm_encoder::Function::new(locals.collect::<Vec<_>>())
        })
        .expect("translate");
    assert_eq!(n, 8);
    assert!(rc.unsupported_insns().is_empty(), "{:?}", rc.unsupported_insns());
}

#[test]
fn bl_with_flag_escape_translates() {
    // BL +0 (self): imm24=0 → target = pc+8
    let bytes = 0xeb00_0000u32.to_le_bytes().to_vec();
    let mut rc = ArmRecompiler::<(), core::convert::Infallible>::new_with_base_pc(0x1000);
    rc.set_speculative_calls(true);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.set_escape(&mut rctx, yecta::CallEscape::Flag);
    let result = rc.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn expand_imm12_mov1() {
    assert_eq!(speet_arm::expand_imm12(0x001), 1);
}

#[test]
fn thumb_stub_names_only() {
    assert_eq!(
        ArmRecompiler::<(), ()>::thumb_unsupported_name(0x2001),
        "thumb16.mov/cmp/add/sub.imm"
    );
}
