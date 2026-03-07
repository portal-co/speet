//! Integration corpus tests for the RISC-V recompiler.
//!
//! Each test reads a pre-assembled ELF object from `test-data/rv-corpus/`,
//! extracts the `.text` section, and feeds it through
//! [`RiscVRecompiler::translate_bytes`].
//!
//! ## Running
//!
//! ```
//! cargo test -p speet-riscv --test rv_corpus_tests
//! ```
//!
//! ## Pre-built objects
//!
//! The ELF objects (extension-less files next to each `.s`) are committed to
//! the rv-corpus submodule and were produced by `compile_corpus.sh` in that
//! directory.  Tests are *skipped* (not failed) when the submodule has not
//! been initialised.

use object::{Object, ObjectSection};
use rv_asm::Xlen;
use speet_riscv::RiscVRecompiler;
use std::convert::Infallible;
use std::{fs, path::Path};
use wasm_encoder::Function;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the `.text` section from a pre-built ELF object.
///
/// Returns `None` when the file does not exist (submodule not initialised).
/// Panics on any other I/O or parse error so malformed committed objects are
/// caught immediately.
fn load_text_section(elf_path: &Path) -> Option<(Vec<u8>, u64)> {
    if !elf_path.exists() {
        eprintln!(
            "Skipping: ELF object not found at {elf_path:?}\n\
             Run 'git submodule update --init --recursive' to fetch rv-corpus."
        );
        return None;
    }
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
    Some((data, addr))
}

fn recompile_riscv_text(text: &[u8], start_addr: u64, xlen: Xlen) -> usize {
    let mut recompiler =
        RiscVRecompiler::<(), Infallible, Function>::new_with_base_pc(start_addr);
    let mut ctx = ();
    recompiler
        .translate_bytes(
            &mut ctx,
            text,
            start_addr as u32,
            xlen,
            &mut |a| Function::new(a.collect::<Vec<_>>()),
        )
        .expect("translate_bytes failed")
}

fn test_rv32_corpus_file(elf_path: &Path) {
    let (text, addr) = match load_text_section(elf_path) {
        Some(v) => v,
        None => return,
    };
    assert!(
        !text.is_empty(),
        "{}: .text section is empty",
        elf_path.display()
    );
    let bytes = recompile_riscv_text(&text, addr, Xlen::Rv32);
    assert!(bytes > 0, "should have translated some bytes");
    println!(
        "  ✓ {} — {bytes} bytes translated",
        elf_path.file_name().unwrap().to_string_lossy()
    );
}

fn test_rv64_corpus_file(elf_path: &Path) {
    let (text, addr) = match load_text_section(elf_path) {
        Some(v) => v,
        None => return,
    };
    assert!(
        !text.is_empty(),
        "{}: .text section is empty",
        elf_path.display()
    );
    let bytes = recompile_riscv_text(&text, addr, Xlen::Rv64);
    assert!(bytes > 0, "should have translated some bytes");
    println!(
        "  ✓ {} — {bytes} bytes translated",
        elf_path.file_name().unwrap().to_string_lossy()
    );
}

// ---------------------------------------------------------------------------
// Individual test cases
// ---------------------------------------------------------------------------

/// RV32I integer computational instructions.
#[test]
fn test_rv32i_integer_computational() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/01_integer_computational");
    test_rv32_corpus_file(&path);
}

/// RV32I control transfer instructions.
#[test]
fn test_rv32i_control_transfer() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/02_control_transfer");
    test_rv32_corpus_file(&path);
}

/// RV32I load / store instructions.
#[test]
fn test_rv32i_load_store() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/03_load_store");
    test_rv32_corpus_file(&path);
}

/// RV32I edge cases.
#[test]
fn test_rv32i_edge_cases() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/04_edge_cases");
    test_rv32_corpus_file(&path);
}

/// RV32I simple program.
#[test]
fn test_rv32i_simple_program() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/05_simple_program");
    test_rv32_corpus_file(&path);
}

/// RV32I NOP and hint encodings.
#[test]
fn test_rv32i_nop_and_hints() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/06_nop_and_hints");
    test_rv32_corpus_file(&path);
}

/// RV32I pseudo-instructions.
#[test]
fn test_rv32i_pseudo_instructions() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32i/07_pseudo_instructions");
    test_rv32_corpus_file(&path);
}

/// RV32IM multiply / divide instructions.
#[test]
fn test_rv32im_multiply_divide() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv32im/01_multiply_divide");
    test_rv32_corpus_file(&path);
}

/// RV64I basic 64-bit instructions.
#[test]
fn test_rv64i_basic_64bit() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-data/rv-corpus/rv64i/01_basic_64bit");
    test_rv64_corpus_file(&path);
}

/// Smoke test: run every pre-built ELF found in rv-corpus through the
/// recompiler in a single pass.  Skips gracefully when the submodule has not
/// been initialised.
#[test]
fn test_rv_full_corpus() {
    let corpus_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../test-data/rv-corpus");

    if !corpus_root.exists() {
        eprintln!("Skipping test_rv_full_corpus: rv-corpus submodule not initialised");
        return;
    }

    let elfs = collect_elfs(&corpus_root);
    if elfs.is_empty() {
        eprintln!("Skipping test_rv_full_corpus: no pre-built ELF objects found in rv-corpus");
        return;
    }

    println!("\n  rv corpus ({} files):", elfs.len());
    for (path, xlen) in &elfs {
        let (text, addr) = match load_text_section(path) {
            Some(v) => v,
            None => return,
        };
        if text.is_empty() {
            continue;
        }
        let bytes = recompile_riscv_text(&text, addr, *xlen);
        println!(
            "  ✓ {} — {bytes} bytes",
            path.file_name().unwrap().to_string_lossy()
        );
    }
}

// ---------------------------------------------------------------------------
// ELF discovery
// ---------------------------------------------------------------------------

/// Recursively collect all pre-built ELF objects (extension-less files whose
/// magic bytes confirm they are ELF) under `root`, paired with the [`Xlen`]
/// inferred from their top-level subdirectory name.
fn collect_elfs(root: &Path) -> Vec<(std::path::PathBuf, Xlen)> {
    let mut result = Vec::new();
    collect_elfs_rec(root, root, &mut result);
    result.sort_by(|(a, _), (b, _)| a.cmp(b));
    result
}

fn collect_elfs_rec(root: &Path, dir: &Path, out: &mut Vec<(std::path::PathBuf, Xlen)>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_elfs_rec(root, &path, out);
        } else if path.extension().is_none() && is_elf(&path) {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let top_dir = rel
                .components()
                .next()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_default();
            let xlen = if top_dir.starts_with("rv64") {
                Xlen::Rv64
            } else {
                Xlen::Rv32
            };
            out.push((path, xlen));
        }
    }
}

/// Return `true` when the first four bytes of the file are the ELF magic.
fn is_elf(path: &Path) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            f.read_exact(&mut buf)
        })
        .map(|_| buf == [0x7f, b'E', b'L', b'F'])
        .unwrap_or(false)
}
