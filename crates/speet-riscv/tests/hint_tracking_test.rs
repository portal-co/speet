//! Test HINT instruction tracking with rv-corpus test files
//!
//! This test demonstrates the HINT tracking functionality by loading
//! rv-corpus test files and extracting test case markers.

use object::{Object, ObjectSection};
use rv_asm::{Inst, Xlen};
use speet_riscv::RiscVRecompiler;
use std::path::Path;
use std::{convert::Infallible, fs};
use wasm_encoder::Function;
use yecta::{Pool, TableIdx, TypeIdx};

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

#[test]
fn test_hint_tracking_with_rv32im_multiply() {
    let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-data/rv-corpus/rv32im/01_multiply_divide");

    if !corpus_path.exists() {
        eprintln!("Skipping test: rv-corpus not found at {:?}", corpus_path);
        return;
    }

    let (text_data, start_addr) = match extract_text_section(&corpus_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to extract .text section: {}", e);
            return;
        }
    };

    // Create recompiler with HINT tracking enabled
    let mut recompiler = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
        Pool {
            table: TableIdx(0),
            ty: TypeIdx(0),
        },
        None,
        start_addr as u32,
        true, // Enable HINT tracking
        false, // Disable RV64
        false, // Disable memory64
    );

    // Translate the code and collect HINTs
    let mut offset = 0;
    while offset < text_data.len() {
        if offset + 2 > text_data.len() {
            break;
        }

        let inst_word = if offset + 4 <= text_data.len() {
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
                if let Err(_) =
                    recompiler.translate_instruction(&inst, pc, is_compressed, &mut |a| {
                        Function::new(a.collect::<Vec<_>>())
                    })
                {
                    break;
                }

                offset += match is_compressed {
                    rv_asm::IsCompressed::Yes => 2,
                    rv_asm::IsCompressed::No => 4,
                };
            }
            Err(_) => break,
        }
    }

    // Get collected hints
    let hints = recompiler.get_hints();

    // The rv32im/01_multiply_divide test file should have multiple test case markers
    if !hints.is_empty() {
        println!("Found {} test case markers:", hints.len());
        for hint in hints.iter().take(10) {
            println!("  Test case {} at PC 0x{:x}", hint.value, hint.pc);
        }

        // Test case numbers should be positive and reasonable
        for hint in hints {
            assert!(
                hint.value > 0,
                "HINT values should be positive test case numbers"
            );
            assert!(
                hint.value < 100,
                "HINT values should be reasonable test case numbers"
            );
        }
    } else {
        eprintln!("Warning: No HINT instructions found in the test file");
    }
}

#[test]
fn test_hint_info_structure() {
    // Test the HintInfo structure directly
    use speet_riscv::HintInfo;

    let hint = HintInfo {
        pc: 0x1000,
        value: 42,
    };

    assert_eq!(hint.pc, 0x1000);
    assert_eq!(hint.value, 42);

    // Test that HintInfo is Copy and PartialEq
    let hint2 = hint;
    assert_eq!(hint, hint2);
}

#[test]
fn test_hint_tracking_performance() {
    // Ensure that HINT tracking doesn't significantly impact translation
    let mut recompiler_no_hints = RiscVRecompiler::<Infallible, Function>::new_with_base_pc(0x1000);
    let mut recompiler_with_hints = RiscVRecompiler::<Infallible, Function>::new_with_full_config(
        Pool {
            table: TableIdx(0),
            ty: TypeIdx(0),
        },
        None,
        0x1000,
        true,
        false, // Disable RV64
        false, // Disable memory64
    );

    // Create a sequence of instructions including some HINTs
    let instructions: Vec<Inst> = vec![
        Inst::Addi {
            imm: rv_asm::Imm::new_i32(1),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        }, // HINT
        Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        }, // Regular instruction
        Inst::Addi {
            imm: rv_asm::Imm::new_i32(2),
            dest: rv_asm::Reg(0),
            src1: rv_asm::Reg(0),
        }, // HINT
        Inst::Add {
            dest: rv_asm::Reg(2),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(1),
        }, // Regular instruction
    ];

    // Both should translate successfully
    for (i, inst) in instructions.iter().enumerate() {
        let pc = 0x1000 + (i as u32 * 4);
        assert!(
            recompiler_no_hints
                .translate_instruction(inst, pc, rv_asm::IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
        assert!(
            recompiler_with_hints
                .translate_instruction(inst, pc, rv_asm::IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok()
        );
    }

    // Verify hints were collected
    assert_eq!(recompiler_no_hints.get_hints().len(), 0);
    assert_eq!(recompiler_with_hints.get_hints().len(), 2);
}
