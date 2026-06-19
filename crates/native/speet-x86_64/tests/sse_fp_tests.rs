//! Tests for SSE scalar floating-point translation in the x86_64 recompiler.
//!
//! Each case decodes real instruction bytes (assembled with `clang -arch
//! x86_64`), translates to WASM, and asserts the recompiler recognized every
//! instruction — i.e. `unsupported_insns` is empty and nothing fell through to
//! the `Unreachable` catch-all.

use speet_link_core::ReactorAdapter;
use speet_x86_64::X86Recompiler;
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
    let mut recompiler = X86Recompiler::new();
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
fn sse_arith_f64() {
    // addsd/subsd/mulsd/divsd/minsd/maxsd xmm0, xmm1 ; sqrtsd xmm0, xmm1
    assert_all_supported(&[
        0xF2, 0x0F, 0x58, 0xC1, // addsd  xmm0, xmm1
        0xF2, 0x0F, 0x5C, 0xC1, // subsd  xmm0, xmm1
        0xF2, 0x0F, 0x59, 0xC1, // mulsd  xmm0, xmm1
        0xF2, 0x0F, 0x5E, 0xC1, // divsd  xmm0, xmm1
        0xF2, 0x0F, 0x5D, 0xC1, // minsd  xmm0, xmm1
        0xF2, 0x0F, 0x5F, 0xC1, // maxsd  xmm0, xmm1
        0xF2, 0x0F, 0x51, 0xC1, // sqrtsd xmm0, xmm1
    ]);
}

#[test]
fn sse_arith_f32() {
    assert_all_supported(&[
        0xF3, 0x0F, 0x58, 0xC1, // addss xmm0, xmm1
        0xF3, 0x0F, 0x5C, 0xC1, // subss xmm0, xmm1
        0xF3, 0x0F, 0x59, 0xC1, // mulss xmm0, xmm1
        0xF3, 0x0F, 0x5E, 0xC1, // divss xmm0, xmm1
        0xF3, 0x0F, 0x51, 0xC1, // sqrtss xmm0, xmm1
    ]);
}

#[test]
fn sse_bitwise_abs_neg() {
    // andpd/xorpd/orpd + pxor — abs/neg/copysign idioms on raw bits
    assert_all_supported(&[
        0x66, 0x0F, 0x54, 0xC1, // andpd xmm0, xmm1
        0x66, 0x0F, 0x57, 0xC1, // xorpd xmm0, xmm1
        0x66, 0x0F, 0x56, 0xC1, // orpd  xmm0, xmm1
        0x66, 0x0F, 0xEF, 0xC1, // pxor  xmm0, xmm1
    ]);
}

#[test]
fn sse_moves() {
    // movsd/movss/movaps reg-reg ; movq/movd xmm<->gpr
    assert_all_supported(&[
        0xF2, 0x0F, 0x10, 0xC1, // movsd  xmm0, xmm1
        0xF3, 0x0F, 0x10, 0xC1, // movss  xmm0, xmm1
        0x0F, 0x28, 0xC1,       // movaps xmm0, xmm1
        0x66, 0x48, 0x0F, 0x6E, 0xC0, // movq xmm0, rax
        0x66, 0x48, 0x0F, 0x7E, 0xC0, // movq rax, xmm0
        0x66, 0x0F, 0x6E, 0xC0, // movd xmm0, eax
        0x66, 0x0F, 0x7E, 0xC0, // movd eax, xmm0
    ]);
}

#[test]
fn sse_compare() {
    assert_all_supported(&[
        0x66, 0x0F, 0x2E, 0xC1, // ucomisd xmm0, xmm1
        0x66, 0x0F, 0x2F, 0xC1, // comisd  xmm0, xmm1
        0x0F, 0x2E, 0xC1,       // ucomiss xmm0, xmm1
    ]);
}

#[test]
fn sse_convert() {
    assert_all_supported(&[
        0xF2, 0x48, 0x0F, 0x2A, 0xC0, // cvtsi2sd xmm0, rax
        0xF2, 0x0F, 0x2A, 0xC0,       // cvtsi2sd xmm0, eax
        0xF2, 0x48, 0x0F, 0x2C, 0xC0, // cvttsd2si rax, xmm0
        0xF2, 0x0F, 0x2C, 0xC0,       // cvttsd2si eax, xmm0
        0xF2, 0x0F, 0x5A, 0xC1,       // cvtsd2ss xmm0, xmm1
        0xF3, 0x0F, 0x5A, 0xC1,       // cvtss2sd xmm0, xmm1
    ]);
}

#[test]
fn sse_load_store() {
    // movsd xmm0, [rax] ; movsd [rax], xmm0 ; movss xmm0, [rax]
    assert_all_supported(&[
        0xF2, 0x0F, 0x10, 0x00, // movsd xmm0, [rax]
        0xF2, 0x0F, 0x11, 0x00, // movsd [rax], xmm0
        0xF3, 0x0F, 0x10, 0x00, // movss xmm0, [rax]
    ]);
}
