//! Tests for the x86_64 integer instructions that close the asm-arch sync
//! gap (NOT, CMOVcc, DIV/IDIV; MUL needed no new code — see direct.rs).
//!
//! Each case decodes real instruction bytes (assembled with `clang -target
//! x86_64-apple-macos`), translates to WASM, and asserts every real
//! instruction was recognized and none fell through to the `Unreachable`
//! catch-all (see `assert_all_supported` for why this isn't simply
//! `unsupported_insns().is_empty()`).

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
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape_tag: None,
    }
}

/// Translate `bytes` (a straight-line, non-branching real instruction
/// stream starting at offset 0, the actual entry point) and assert every one
/// of those real instructions was recognized.
///
/// The recompiler's decode-stride tries every byte offset as a potential
/// instruction slot (see docs/guides/yecta.md §1a), so misaligned offsets
/// routinely decode to real-but-unimplemented x86 mnemonics that legitimately
/// land in `unsupported_insns()` — comparing mnemonic names against that set
/// can't tell a real failure apart from this noise, since a misaligned
/// re-decode of a real instruction's own bytes can produce the very same
/// mnemonic name. Instead, this inspects the WASM produced for the offset-0
/// slot directly: every real instruction is preceded by a PC marker
/// (`i32.const <pc>; local.set 16`), and an unsupported instruction calls
/// `seal_fn` to terminate its function immediately rather than letting
/// translation continue into the next real instruction — so the chain
/// starting at offset 0 contains exactly as many PC markers as there are
/// real instructions only if every one of them was supported.
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

    let mut decoder = iced_x86::Decoder::with_ip(64, bytes, 0x1000, iced_x86::DecoderOptions::NONE);
    let mut inst = iced_x86::Instruction::default();
    let mut real_count = 0usize;
    while decoder.can_decode() {
        decoder.decode_out(&mut inst);
        real_count += 1;
    }

    rctx.reactor.seal_remaining(&mut ctx).unwrap();
    let fns = rctx.reactor.drain_fns();
    let raw = fns[0].clone().into_raw_body();
    let reader = wasmparser::BinaryReader::new(&raw, 0);
    let body = wasmparser::FunctionBody::new(reader);
    let mut ops_reader = body.get_operators_reader().expect("valid function body");
    let mut ops = Vec::new();
    while !ops_reader.eof() {
        ops.push(ops_reader.read().expect("valid operator"));
    }
    let pc_markers = ops.windows(2).filter(|w| {
        matches!(w[0], wasmparser::Operator::I32Const { .. })
            && matches!(w[1], wasmparser::Operator::LocalSet { local_index: 16 })
    }).count();

    assert_eq!(
        pc_markers, real_count,
        "expected all {real_count} real instructions to chain into the offset-0 \
         function, but only {pc_markers} PC markers were found — an unsupported \
         instruction broke the chain early (full unsupported set, including \
         decode-stride noise from misaligned offsets: {:?})",
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
