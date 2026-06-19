//! Tests for the x86_64 integer instructions that close the asm-arch sync
//! gap (NOT, CMOVcc, DIV/IDIV; MUL needed no new code — see direct.rs).
//!
//! Each case decodes real instruction bytes (assembled with `clang -target
//! x86_64-apple-macos`), translates to WASM, and asserts the recompiler
//! recognized every instruction — i.e. `unsupported_insns` is empty and
//! nothing fell through to the `Unreachable` catch-all.

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
fn not_all_widths() {
    assert_all_supported(&[
        0x48, 0xF7, 0xD0,       // not rax
        0xF7, 0xD0,             // not eax
        0x66, 0xF7, 0xD0,       // not ax
        0xF6, 0xD0,             // not al
        0x48, 0xF7, 0x10,       // not qword [rax]
    ]);
}

#[test]
fn cmovcc_widths() {
    assert_all_supported(&[
        0x48, 0x0F, 0x44, 0xC3, // cmove rax, rbx
        0x0F, 0x4C, 0xC3,       // cmovl eax, ebx
        0x66, 0x0F, 0x43, 0xC3, // cmovae ax, bx
    ]);
}

#[test]
fn div_idiv_widths() {
    assert_all_supported(&[
        0xF6, 0xF3,             // div bl
        0x66, 0xF7, 0xF3,       // div bx
        0xF7, 0xF3,             // div ebx
        0x48, 0xF7, 0xF3,       // div rbx
        0xF7, 0xFB,             // idiv ebx
        0x48, 0xF7, 0xFB,       // idiv rbx
    ]);
}

#[test]
fn hlt() {
    assert_all_supported(&[0xF4]); // hlt
}
