//! Tests for the aarch64 FMIN/FMAX/FMINNM/FMAXNM instructions that close the
//! asm-arch sync gap (Phase A1; FADD/FSUB/FMUL/FDIV/FNMUL already worked).
//!
//! Each case decodes a real instruction word (assembled with `clang` on
//! arm64 macOS), translates to WASM, and asserts the recompiler recognized
//! every instruction — i.e. `unsupported_insns` is empty and nothing fell
//! through to the `Unreachable` catch-all.

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
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape: yecta::CallEscape::Jump,
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
fn fmin_fmax_f64() {
    assert_all_supported(&[
        0x00, 0x58, 0x61, 0x1E, // fmin   d0, d0, d1
        0x00, 0x48, 0x61, 0x1E, // fmax   d0, d0, d1
        0x00, 0x78, 0x61, 0x1E, // fminnm d0, d0, d1
        0x00, 0x68, 0x61, 0x1E, // fmaxnm d0, d0, d1
    ]);
}

#[test]
fn fmin_fmax_f32() {
    assert_all_supported(&[
        0x00, 0x58, 0x21, 0x1E, // fmin   s0, s0, s1
        0x00, 0x48, 0x21, 0x1E, // fmax   s0, s0, s1
        0x00, 0x78, 0x21, 0x1E, // fminnm s0, s0, s1
        0x00, 0x68, 0x21, 0x1E, // fmaxnm s0, s0, s1
    ]);
}
