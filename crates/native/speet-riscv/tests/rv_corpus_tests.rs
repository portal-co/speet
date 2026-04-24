//! Integration corpus tests for the RISC-V recompiler.
//!
//! Each test reads a pre-assembled ELF object from `test-data/rv-corpus/`,
//! extracts the `.text` section, and feeds it through
//! [`RiscVRecompiler::translate_bytes`].
//!
//! Tests are *skipped* (not failed) when the submodule has not been initialised.

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_link_core::ReactorAdapter;
use speet_riscv::RiscVRecompiler;
use std::convert::Infallible;
use std::{fs, path::Path};
use wasm_encoder::Function;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

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

fn load_text_section(elf_path: &Path) -> Option<(Vec<u8>, u64)> {
    if !elf_path.exists() {
        eprintln!(
            "Skipping: ELF object not found at {elf_path:?}\n\
             Run 'git submodule update --init --recursive' to fetch rv-corpus."
        );
        return None;
    }
    let bytes = fs::read(elf_path).unwrap_or_else(|e| panic!("failed to read {}: {e}", elf_path.display()));
    let obj = object::File::parse(&*bytes).unwrap_or_else(|e| panic!("failed to parse ELF {}: {e}", elf_path.display()));
    let section = obj.section_by_name(".text").unwrap_or_else(|| panic!("no .text section in {}", elf_path.display()));
    let data = section.data().unwrap_or_else(|e| panic!("failed to read .text from {}: {e}", elf_path.display())).to_vec();
    let addr = section.address();
    Some((data, addr))
}

fn recompile_riscv_text(text: &[u8], start_addr: u64, xlen: Xlen) -> usize {
    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_base_pc(start_addr);
    let mut ctx = ();
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, start_addr as u32, xlen, &mut |a| Function::new(a.collect::<Vec<_>>()))
        .expect("translate_bytes failed")
}

fn test_rv32_corpus_file(elf_path: &Path) {
    let (text, addr) = match load_text_section(elf_path) { Some(v) => v, None => return };
    assert!(!text.is_empty(), "{}: .text section is empty", elf_path.display());
    let bytes = recompile_riscv_text(&text, addr, Xlen::Rv32);
    assert!(bytes > 0, "should have translated some bytes");
    println!("  ✓ {} — {bytes} bytes translated", elf_path.file_name().unwrap().to_string_lossy());
}

fn test_rv64_corpus_file(elf_path: &Path) {
    let (text, addr) = match load_text_section(elf_path) { Some(v) => v, None => return };
    assert!(!text.is_empty(), "{}: .text section is empty", elf_path.display());
    let bytes = recompile_riscv_text(&text, addr, Xlen::Rv64);
    assert!(bytes > 0, "should have translated some bytes");
    println!("  ✓ {} — {bytes} bytes translated", elf_path.file_name().unwrap().to_string_lossy());
}

macro_rules! rv32_test {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus").join($rel);
            test_rv32_corpus_file(&p);
        }
    };
}

macro_rules! rv64_test {
    ($name:ident, $rel:expr) => {
        #[test]
        fn $name() {
            let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus").join($rel);
            test_rv64_corpus_file(&p);
        }
    };
}

rv32_test!(rv32_multiply_divide,    "rv32im/01_multiply_divide");
rv32_test!(rv32_branches,           "rv32im/02_branches");
rv32_test!(rv32_loads_stores,       "rv32im/03_loads_stores");
rv32_test!(rv32_shifts,             "rv32im/04_shifts");
rv32_test!(rv32_immediate_ops,      "rv32im/05_immediate_ops");
rv64_test!(rv64_basic_arith,        "rv64im/01_basic_arith");
rv64_test!(rv64_word_ops,           "rv64im/02_word_ops");
rv64_test!(rv64_loads_stores,       "rv64im/03_loads_stores");
rv64_test!(rv64_shifts,             "rv64im/04_shifts");
rv64_test!(rv64_multiply_divide,    "rv64im/05_multiply_divide");
