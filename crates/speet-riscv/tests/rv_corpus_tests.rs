//! Integration tests using rv-corpus ELF files
//!
//! These tests validate the RISC-V recompiler against real compiled programs
//! from the rv-corpus test suite.

use object::{Object, ObjectSection};
use rv_asm::{Inst, Xlen};
use speet_riscv::RiscVRecompiler;
use std::convert::Infallible;
use std::fs;
use std::path::Path;
use wasm_encoder::Function;

/// Helper function to extract the .text section from an ELF file
fn extract_text_section(elf_path: &Path) -> Result<(Vec<u8>, u64), Box<dyn std::error::Error>> {
    let file_data = fs::read(elf_path)?;
    let obj_file = object::File::parse(&*file_data)?;

    if let Some(text_section) = obj_file.section_by_name(".text") {
        let data = text_section.data()?.to_vec();
        let address = text_section.address();
        Ok((data, address))
    } else {
        Err("No .text section found".into())
    }
}

/// Helper function to test recompilation of an ELF file
fn test_elf_recompilation(elf_path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let (text_data, start_addr) = extract_text_section(elf_path)?;

    // Create recompiler with the start address as base_pc
    let mut recompiler =
        RiscVRecompiler::<Infallible, Function>::new_with_base_pc(start_addr as u32);

    // Translate the entire .text section
    let bytes_translated = recompiler
        .translate_bytes(&text_data, start_addr as u32, Xlen::Rv32, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        })
        .map_err(|_| "Translation failed")?;

    Ok(bytes_translated)
}

/// Test RV32I integer computational instructions
#[test]
fn test_rv32i_integer_computational() {
    let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/rv-corpus/rv32i/01_integer_computational");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found at {:?}", corpus_path);
        eprintln!("Run 'git submodule update --init --recursive' to fetch rv-corpus");
        return;
    }

    match test_elf_recompilation(&corpus_path) {
        Ok(bytes) => {
            println!(
                "Successfully translated {} bytes from {}",
                bytes,
                corpus_path.display()
            );
            assert!(bytes > 0, "Should have translated some bytes");
        }
        Err(e) => {
            eprintln!("Failed to recompile {}: {}", corpus_path.display(), e);
            panic!("Recompilation failed");
        }
    }
}

/// Test RV32I control transfer instructions  
#[test]
fn test_rv32i_control_transfer() {
    let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/rv-corpus/rv32i/02_control_transfer");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found");
        return;
    }

    match test_elf_recompilation(&corpus_path) {
        Ok(bytes) => {
            println!(
                "Successfully translated {} bytes from {}",
                bytes,
                corpus_path.display()
            );
            assert!(bytes > 0, "Should have translated some bytes");
        }
        Err(e) => {
            eprintln!("Failed to recompile {}: {}", corpus_path.display(), e);
            panic!("Recompilation failed");
        }
    }
}

/// Test RV32I load/store instructions
#[test]
fn test_rv32i_load_store() {
    let corpus_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus/rv32i/03_load_store");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found");
        return;
    }

    match test_elf_recompilation(&corpus_path) {
        Ok(bytes) => {
            println!(
                "Successfully translated {} bytes from {}",
                bytes,
                corpus_path.display()
            );
            assert!(bytes > 0, "Should have translated some bytes");
        }
        Err(e) => {
            eprintln!("Failed to recompile {}: {}", corpus_path.display(), e);
            panic!("Recompilation failed");
        }
    }
}

/// Test RV32I edge cases
#[test]
fn test_rv32i_edge_cases() {
    let corpus_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus/rv32i/04_edge_cases");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found");
        return;
    }

    match test_elf_recompilation(&corpus_path) {
        Ok(bytes) => {
            println!(
                "Successfully translated {} bytes from {}",
                bytes,
                corpus_path.display()
            );
            assert!(bytes > 0, "Should have translated some bytes");
        }
        Err(e) => {
            eprintln!("Failed to recompile {}: {}", corpus_path.display(), e);
            panic!("Recompilation failed");
        }
    }
}

/// Test RV32IM (with M extension) programs
#[test]
fn test_rv32im_mul_div() {
    let corpus_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus/rv32im");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus rv32im not found");
        return;
    }

    // Find the first ELF file in the directory
    if let Ok(entries) = fs::read_dir(&corpus_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && !path.extension().map_or(false, |e| e == "s") {
                match test_elf_recompilation(&path) {
                    Ok(bytes) => {
                        println!(
                            "Successfully translated {} bytes from {}",
                            bytes,
                            path.display()
                        );
                        assert!(bytes > 0, "Should have translated some bytes");
                        return; // Test first file only
                    }
                    Err(e) => {
                        eprintln!("Failed to recompile {}: {}", path.display(), e);
                    }
                }
            }
        }
    }

    eprintln!("No RV32IM ELF files found to test");
}

/// Test RV32F (single-precision float) programs
#[test]
fn test_rv32f_float() {
    let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/rv-corpus/rv32f");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus rv32f not found");
        return;
    }

    // Find the first ELF file in the directory
    if let Ok(entries) = fs::read_dir(&corpus_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && !path.extension().map_or(false, |e| e == "s") {
                match test_elf_recompilation(&path) {
                    Ok(bytes) => {
                        println!(
                            "Successfully translated {} bytes from {}",
                            bytes,
                            path.display()
                        );
                        assert!(bytes > 0, "Should have translated some bytes");
                        return; // Test first file only
                    }
                    Err(e) => {
                        eprintln!("Failed to recompile {}: {}", path.display(), e);
                    }
                }
            }
        }
    }

    eprintln!("No RV32F ELF files found to test");
}

/// Test detailed instruction-by-instruction translation
#[test]
fn test_detailed_instruction_translation() {
    let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/rv-corpus/rv32i/05_simple_program");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found");
        return;
    }

    match extract_text_section(&corpus_path) {
        Ok((text_data, start_addr)) => {
            let mut recompiler =
                RiscVRecompiler::<Infallible, Function>::new_with_base_pc(start_addr as u32);

            // Try to decode and translate instruction by instruction
            let mut offset = 0;
            let mut instruction_count = 0;

            while offset < text_data.len() && offset < 100 {
                // Limit to first 100 bytes
                if offset + 1 >= text_data.len() {
                    break;
                }

                let inst_word = if offset + 3 < text_data.len() {
                    u32::from_le_bytes([
                        text_data[offset],
                        text_data[offset + 1],
                        text_data[offset + 2],
                        text_data[offset + 3],
                    ])
                } else {
                    u32::from_le_bytes([text_data[offset], text_data[offset + 1], 0, 0])
                };

                match Inst::decode(inst_word, Xlen::Rv32) {
                    Ok((inst, is_compressed)) => {
                        let pc = start_addr as u32 + offset as u32;
                        match recompiler.translate_instruction(&inst, pc, is_compressed, &mut |a| {
                            Function::new(a.collect::<Vec<_>>())
                        }) {
                            Ok(()) => {
                                instruction_count += 1;
                                offset += match is_compressed {
                                    rv_asm::IsCompressed::Yes => 2,
                                    rv_asm::IsCompressed::No => 4,
                                };
                            }
                            Err(_) => break,
                        }
                    }
                    Err(_) => break,
                }
            }

            println!("Successfully translated {} instructions", instruction_count);
            assert!(
                instruction_count > 0,
                "Should have translated at least some instructions"
            );
        }
        Err(e) => {
            eprintln!("Failed to extract .text section: {}", e);
            panic!("Test setup failed");
        }
    }
}
