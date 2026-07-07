//! Decode-coverage tests for the SCVTF/UCVTF/FCVTZS/FCVTZU width-combo fix.
//!
//! disarm64 merges all four `{s,d} <-> {w,x}` width combinations of each of
//! these int<->FP convert mnemonics into a single enum variant (distinguished
//! only by the raw `sf`/`ftype` instruction bits, not by separate variants).
//! The handler in `direct/fp.rs` used to hardcode the narrowest combo for all
//! four — this asserts every combo still decodes and translates (no fallback
//! to `Unreachable`/`unsupported_insns`) now that it branches on those bits.

use speet_aarch64::AArch64Recompiler;
use speet_link_core::ReactorAdapter;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn make_rctx(reactor: &mut Reactor<(), core::convert::Infallible, wasm_encoder::Function, LocalPool>)
    -> ReactorAdapter<'_, (), core::convert::Infallible, wasm_encoder::Function, LocalPool>
{
    static T: TableIdx = TableIdx(0);
    ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape_tag: None,
    }
}

/// Translate `bytes` and assert every instruction was recognized.
fn assert_all_supported(bytes: &[u8]) {
    let mut recompiler = AArch64Recompiler::new();
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok(), "translate_bytes failed: {:?}", result.err());
    assert!(
        recompiler.unsupported_insns().is_empty(),
        "unexpected unsupported instructions: {:?}",
        recompiler.unsupported_insns()
    );
}

#[test]
fn scvtf_all_width_combos() {
    assert_all_supported(&[
        0x00, 0x00, 0x22, 0x1E, // scvtf s0, w0
        0x00, 0x00, 0x62, 0x1E, // scvtf d0, w0
        0x00, 0x00, 0x22, 0x9E, // scvtf s0, x0
        0x00, 0x00, 0x62, 0x9E, // scvtf d0, x0
    ]);
}

#[test]
fn ucvtf_all_width_combos() {
    assert_all_supported(&[
        0x00, 0x00, 0x23, 0x1E, // ucvtf s0, w0
        0x00, 0x00, 0x63, 0x1E, // ucvtf d0, w0
        0x00, 0x00, 0x23, 0x9E, // ucvtf s0, x0
        0x00, 0x00, 0x63, 0x9E, // ucvtf d0, x0
    ]);
}

#[test]
fn fcvtzs_all_width_combos() {
    assert_all_supported(&[
        0x00, 0x00, 0x38, 0x1E, // fcvtzs w0, s0
        0x00, 0x00, 0x38, 0x9E, // fcvtzs x0, s0
        0x00, 0x00, 0x78, 0x1E, // fcvtzs w0, d0
        0x00, 0x00, 0x78, 0x9E, // fcvtzs x0, d0
    ]);
}

#[test]
fn fcvtzu_all_width_combos() {
    assert_all_supported(&[
        0x00, 0x00, 0x39, 0x1E, // fcvtzu w0, s0
        0x00, 0x00, 0x39, 0x9E, // fcvtzu x0, s0
        0x00, 0x00, 0x79, 0x1E, // fcvtzu w0, d0
        0x00, 0x00, 0x79, 0x9E, // fcvtzu x0, d0
    ]);
}
