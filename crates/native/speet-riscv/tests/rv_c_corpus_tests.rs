//! RISC-V C compiler-output corpus tests.

use core::convert::Infallible;
use rv_asm::Xlen;
use speet_corpus_harness::{corpus_n_imports, run_c_program_text, CorpusArch};
use speet_link_core::{BaseContext, ReactorAdapter};
use speet_riscv::RiscVRecompiler;
use std::path::Path;
use wasm_encoder::{Function, ValType};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn corpus_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/rv-c-corpus")
}

fn c_corpus_programs_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/c-corpus")
}

fn make_rctx(reactor: &mut Reactor<(), Infallible, Function, LocalPool>)
    -> ReactorAdapter<'_, (), Infallible, Function, LocalPool>
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

fn collect_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    rctx.layout
        .iter_before(&rctx.locals_mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

fn translate_riscv(text: &[u8], base: u64, xlen: Xlen) -> (Vec<Function>, Vec<ValType>) {
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_base_pc(base);
    recompiler.set_rv64_support(matches!(xlen, Xlen::Rv64));
    recompiler.set_memory64(true);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    rctx.set_base_func_offset(corpus_n_imports());
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = collect_params(&rctx);
    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, base as u32, xlen, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .expect("translate_bytes");
    let _ = rctx.reactor.seal_remaining(&mut ctx);
    (rctx.reactor.drain_fns(), params)
}

#[test]
fn rv_c_corpus_arith32() {
    run_c_program_text(
        CorpusArch::Riscv,
        &c_corpus_programs_root(),
        "arith",
        "riscv32-unknown-linux-gnu",
        &corpus_root().join("rv32/arith.text.elf"),
        |text, base| translate_riscv(text, base, Xlen::Rv32),
    );
}

#[test]
fn rv_c_corpus_frame32() {
    run_c_program_text(
        CorpusArch::Riscv,
        &c_corpus_programs_root(),
        "frame",
        "riscv32-unknown-linux-gnu",
        &corpus_root().join("rv32/frame.text.elf"),
        |text, base| translate_riscv(text, base, Xlen::Rv32),
    );
}

#[test]
fn rv_c_corpus_arith64() {
    run_c_program_text(
        CorpusArch::Riscv,
        &c_corpus_programs_root(),
        "arith",
        "riscv64-unknown-linux-gnu",
        &corpus_root().join("rv64/arith.text.elf"),
        |text, base| translate_riscv(text, base, Xlen::Rv64),
    );
}

#[test]
fn rv_c_corpus_frame64() {
    run_c_program_text(
        CorpusArch::Riscv,
        &c_corpus_programs_root(),
        "frame",
        "riscv64-unknown-linux-gnu",
        &corpus_root().join("rv64/frame.text.elf"),
        |text, base| translate_riscv(text, base, Xlen::Rv64),
    );
}
