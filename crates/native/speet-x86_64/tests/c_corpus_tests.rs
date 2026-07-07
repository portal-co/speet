//! C compiler-output corpus tests (runtime unreachable-trap detection).

use core::convert::Infallible;
use speet_corpus_harness::{run_c_program_text, CorpusArch, N_CORPUS_IMPORTS};
use speet_link_core::{BaseContext, ReactorAdapter};
use speet_x86_64::X86Recompiler;
use std::path::Path;
use wasm_encoder::{Function, ValType};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus")
}

fn corpus_dir() -> std::path::PathBuf {
    corpus_root().join("x86_64-linux")
}

fn make_rctx(reactor: &mut Reactor<(), Infallible, Function, LocalPool>)
    -> ReactorAdapter<'_, (), Infallible, Function, LocalPool>
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

fn collect_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    rctx.layout
        .iter_before(&rctx.locals_mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

fn translate_x86(text: &[u8], base: u64) -> (Vec<Function>, Vec<ValType>) {
    let mut recompiler = X86Recompiler::new_with_base_rip(base);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    // Must match `assemble_corpus_module`'s import count — see the AArch64
    // corpus test's identical comment for why this is load-bearing.
    rctx.set_base_func_offset(N_CORPUS_IMPORTS);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = collect_params(&rctx);
    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, base, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .expect("translate_bytes");
    let _ = rctx.reactor.seal_remaining(&mut ctx);
    (rctx.reactor.drain_fns(), params)
}

#[test]
fn c_corpus_arith() {
    run_c_program_text(
        CorpusArch::X86_64,
        &corpus_root(),
        "arith",
        "x86_64-linux-gnu",
        &corpus_dir().join("arith.text.elf"),
        |text, base| translate_x86(text, base),
    );
}

#[test]
fn c_corpus_frame() {
    run_c_program_text(
        CorpusArch::X86_64,
        &corpus_root(),
        "frame",
        "x86_64-linux-gnu",
        &corpus_dir().join("frame.text.elf"),
        |text, base| translate_x86(text, base),
    );
}
