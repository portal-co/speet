//! Integration corpus tests for the x86_64 recompiler.
//!
//! Each test reads a pre-assembled ELF object from `test-data/x86_64-corpus/`,
//! extracts the `.text` section, and feeds it through
//! [`X86Recompiler::translate_bytes`].
//!
//! ## Running
//!
//! ```
//! cargo test -p speet-x86_64 --test x86_corpus_tests
//! ```
//!
//! ## Pre-built objects
//!
//! The `.elf` files sitting next to each `.s` source were produced by
//! `test-data/x86_64-corpus/compile_corpus.sh` and are committed to the
//! repository.  They contain only the `.text` section — no debug info, no
//! symbol table, no source-file metadata.  Re-run that script whenever a
//! `.s` source changes, then commit the updated `.elf`.

use object::{Object, ObjectSection};
use speet_x86_64::X86Recompiler;
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

fn test_corpus_file(elf_path: &Path) {
    let (text, load_addr) = load_text_section(elf_path);

    assert!(
        !text.is_empty(),
        "{}: .text section is empty",
        elf_path.display()
    );

    let mut recompiler: X86Recompiler<(), Infallible, Function> =
        X86Recompiler::new_with_base_rip(load_addr);
    let mut ctx = ();

    recompiler
        .translate_bytes(&mut ctx, &text, load_addr, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .expect("translate_bytes failed");

    println!(
        "  ✓ {} — {} bytes translated",
        elf_path.file_name().unwrap().to_string_lossy(),
        text.len()
    );
}

// ---------------------------------------------------------------------------
// Individual test cases
// ---------------------------------------------------------------------------

/// Integer computational instructions: MOV, ADD, SUB, IMUL, AND, OR, XOR,
/// NOT, NEG, INC, DEC, SHL, SHR, SAR, LEA.
#[test]
fn test_x86_integer_computational() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/x86_64-corpus/01_integer_computational.elf");
    test_corpus_file(&path);
}

/// Control transfer: JMP, Jcc (all 16 conditions), CALL/RET, LOOP, indirect
/// jump through register.
#[test]
fn test_x86_control_transfer() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/x86_64-corpus/02_control_transfer.elf");
    test_corpus_file(&path);
}

/// Load / store: MOV widths, MOVZX/MOVSX, PUSH/POP, XCHG, CMPXCHG,
/// stack-frame setup/teardown, all addressing modes.
#[test]
fn test_x86_load_store() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/x86_64-corpus/03_load_store.elf");
    test_corpus_file(&path);
}

/// Flags and conditional operations: CMP, TEST, SETcc, CMOVcc, LAHF/SAHF,
/// ADC, flag interactions after NEG/arithmetic.
#[test]
fn test_x86_flags_and_setcc() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/x86_64-corpus/04_flags_and_setcc.elf");
    test_corpus_file(&path);
}

/// Edge cases: REX prefixes, 32-bit zero-extension, NOP forms, CDQ/CQO,
/// BSWAP, BSF/BSR, MOVABS, RIP-relative, IDIV/DIV, wide IMUL.
#[test]
fn test_x86_edge_cases() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/x86_64-corpus/05_edge_cases.elf");
    test_corpus_file(&path);
}

/// Smoke test: recompile all corpus ELF objects in a single pass.
#[test]
fn test_x86_full_corpus() {
    let corpus_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/x86_64-corpus");

    let mut entries: Vec<_> = fs::read_dir(&corpus_dir)
        .expect("reading x86_64-corpus dir")
        .flatten()
        .filter(|e| e.path().extension().map_or(false, |x| x == "elf"))
        .collect();

    entries.sort_by_key(|e| e.path());

    assert!(
        !entries.is_empty(),
        "x86_64-corpus contains no .elf files — run compile_corpus.sh"
    );

    println!("\n  x86_64 corpus ({} files):", entries.len());
    for entry in entries {
        test_corpus_file(&entry.path());
    }
}
