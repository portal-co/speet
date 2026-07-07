//! C compiler-output corpus tests (runtime unreachable-trap detection).

use core::convert::Infallible;
use speet_aarch64::AArch64Recompiler;
use speet_corpus_harness::{corpus_n_imports, run_c_program_text, CorpusArch};
use speet_link_core::{BaseContext, ReactorAdapter};
use std::path::Path;
use wasm_encoder::{Function, ValType};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus")
}

fn corpus_dir() -> std::path::PathBuf {
    corpus_root().join("aarch64-linux")
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

fn translate_aarch64(text: &[u8], base: u64) -> (Vec<Function>, Vec<ValType>) {
    let mut recompiler = AArch64Recompiler::<(), Infallible>::new_with_base_pc(base);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    // Must match corpus module assembly: table `elem` populates
    // `[n_imports, n_imports + n_fns)`, and every runtime-computed indirect-
    // call/return table index is `local_slot + base_func_offset`.
    rctx.set_base_func_offset(corpus_n_imports());
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
        CorpusArch::AArch64,
        &corpus_root(),
        "arith",
        "aarch64-linux-gnu",
        &corpus_dir().join("arith.text.elf"),
        |text, base| translate_aarch64(text, base),
    );
}

#[test]
fn c_corpus_frame() {
    run_c_program_text(
        CorpusArch::AArch64,
        &corpus_root(),
        "frame",
        "aarch64-linux-gnu",
        &corpus_dir().join("frame.text.elf"),
        |text, base| translate_aarch64(text, base),
    );
}

#[test]
fn c_corpus_pairs() {
    run_c_program_text(
        CorpusArch::AArch64,
        &corpus_root(),
        "pairs",
        "aarch64-linux-gnu",
        &corpus_dir().join("pairs.text.elf"),
        |text, base| translate_aarch64(text, base),
    );
}
