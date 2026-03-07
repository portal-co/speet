//! Integration corpus tests for the MIPS recompiler.
//!
//! Each test reads a pre-assembled ELF object from `test-data/mips-corpus/`,
//! extracts the `.text` section word-by-word, and feeds it through
//! [`MipsRecompiler::translate_instruction`].
//!
//! ## Running
//!
//! ```
//! cargo test -p speet-mips --test mips_corpus_tests
//! ```
//!
//! ## Pre-built objects
//!
//! The `.elf` files sitting next to each `.s` source were produced by
//! `test-data/mips-corpus/compile_corpus.sh` and are committed to the
//! repository.  They contain only the `.text` section — no debug info, no
//! symbol table, no source-file metadata.  Re-run that script whenever a
//! `.s` source changes, then commit the updated `.elf`.

use object::{Object, ObjectSection};
use rabbitizer::{InstrCategory, Instruction};
use speet_mips::MipsRecompiler;
use std::convert::Infallible;
use std::{fs, path::Path};
use wasm_encoder::Function;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the `.text` section from a pre-built ELF object.
///
/// Panics on any I/O or parse error — the files are version-controlled and
/// must always be readable.
fn load_text_section(elf_path: &Path) -> (Vec<u8>, u64) {
    let bytes = fs::read(elf_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", elf_path.display()));
    let obj = object::File::parse(&*bytes)
        .unwrap_or_else(|e| panic!("failed to parse ELF {}: {e}", elf_path.display()));
    let section = obj
        .section_by_name(".text")
        .unwrap_or_else(|| panic!("no .text section in {}", elf_path.display()));
    let data = section
        .data()
        .unwrap_or_else(|e| panic!("failed to read .text from {}: {e}", elf_path.display()))
        .to_vec();
    let addr = section.address();
    (data, addr)
}

/// Parse `text` as a stream of big-endian MIPS32 words and translate each one
/// through the recompiler.  Returns the number of instructions processed.
fn recompile_mips_text(text: &[u8], base_pc: u32) -> usize {
    let mut recompiler: MipsRecompiler<'_, '_, (), Infallible, Function> =
        MipsRecompiler::new_with_base_pc(base_pc);
    let mut ctx = ();
    let mut count = 0usize;

    let words = text.len() / 4;
    for i in 0..words {
        let offset = i * 4;
        let word = u32::from_be_bytes([
            text[offset],
            text[offset + 1],
            text[offset + 2],
            text[offset + 3],
        ]);
        let pc = base_pc.wrapping_add(offset as u32);
        let inst = Instruction::new(word, pc, InstrCategory::CPU);
        recompiler
            .translate_instruction(&mut ctx, &inst, &mut |locals| {
                Function::new(locals.collect::<Vec<_>>())
            })
            .expect("translate_instruction failed");
        count += 1;
    }

    count
}

fn test_corpus_file(elf_path: &Path) {
    let (text, load_addr) = load_text_section(elf_path);

    assert!(
        !text.is_empty(),
        "{}: .text section is empty",
        elf_path.display()
    );
    assert_eq!(
        text.len() % 4,
        0,
        "{}: .text size {} is not a multiple of 4",
        elf_path.display(),
        text.len()
    );

    let count = recompile_mips_text(&text, load_addr as u32);
    assert!(count > 0, "{}: no instructions translated", elf_path.display());

    println!(
        "  ✓ {} — {count} instructions ({} bytes)",
        elf_path.file_name().unwrap().to_string_lossy(),
        text.len()
    );
}

// ---------------------------------------------------------------------------
// Individual test cases
// ---------------------------------------------------------------------------

/// Integer computational: ADD, ADDU, ADDI, ADDIU, SUB, SUBU, AND, OR, XOR,
/// NOR, ANDI, ORI, XORI, LUI, SLL, SRL, SRA, SLLV, SRLV, SRAV, SLT, SLTU,
/// SLTI, SLTIU.
#[test]
fn test_mips_integer_computational() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/mips-corpus/01_integer_computational.elf");
    test_corpus_file(&path);
}

/// Control transfer: J, JAL, JR, JALR, BEQ, BNE, BLEZ, BGTZ, BLTZ, BGEZ,
/// BLTZAL, BGEZAL, and delay-slot NOPs.
#[test]
fn test_mips_control_transfer() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/mips-corpus/02_control_transfer.elf");
    test_corpus_file(&path);
}

/// Load / store: LB, LBU, LH, LHU, LW, SB, SH, SW, all offset ranges.
#[test]
fn test_mips_load_store() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/mips-corpus/03_load_store.elf");
    test_corpus_file(&path);
}

/// Multiply / divide: MULT, MULTU, MUL, DIV, DIVU, MFHI, MFLO, MTHI, MTLO.
#[test]
fn test_mips_multiply_divide() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/mips-corpus/04_multiply_divide.elf");
    test_corpus_file(&path);
}

/// Edge cases: NOP, writes to $zero, MOVE/LI/LA pseudos, shift-amount
/// boundaries (sa=0, sa=31), MOVN/MOVZ.
#[test]
fn test_mips_edge_cases() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/mips-corpus/05_edge_cases.elf");
    test_corpus_file(&path);
}

/// Smoke test: recompile all corpus ELF objects in a single pass.
#[test]
fn test_mips_full_corpus() {
    let corpus_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/mips-corpus");

    let mut entries: Vec<_> = fs::read_dir(&corpus_dir)
        .expect("reading mips-corpus dir")
        .flatten()
        .filter(|e| e.path().extension().map_or(false, |x| x == "elf"))
        .collect();

    entries.sort_by_key(|e| e.path());

    assert!(
        !entries.is_empty(),
        "mips-corpus contains no .elf files — run compile_corpus.sh"
    );

    println!("\n  MIPS corpus ({} files):", entries.len());
    for entry in entries {
        test_corpus_file(&entry.path());
    }
}
