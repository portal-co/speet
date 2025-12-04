//! Example demonstrating HINT instruction tracking
//!
//! This example shows how to enable and use HINT instruction tracking
//! to identify test case boundaries in RISC-V code (as used in rv-corpus).

use rv_asm::{Inst, Reg, Imm, IsCompressed};
use speet_riscv::RiscVRecompiler;
use yecta::{Pool, TableIdx, TypeIdx};

fn main() {
    println!("RISC-V HINT Instruction Tracking Demo");
    println!("======================================\n");

    // Create a recompiler with HINT tracking enabled
    let mut recompiler = RiscVRecompiler::new_with_full_config(
        Pool {
            table: TableIdx(0),
            ty: TypeIdx(0),
        },
        None,
        0x1000,  // base PC
        true,    // enable HINT tracking
    );

    println!("HINT tracking is enabled\n");

    // Simulate a test program with multiple test cases
    // Each test case is marked with a HINT: addi x0, x0, N
    let test_program = vec![
        // Test case 1: Basic arithmetic
        (Inst::Addi { imm: Imm::new_i32(1), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 1"),
        (Inst::Addi { imm: Imm::new_i32(5), dest: Reg(1), src1: Reg(0) }, "li x1, 5"),
        (Inst::Addi { imm: Imm::new_i32(3), dest: Reg(2), src1: Reg(0) }, "li x2, 3"),
        (Inst::Add { dest: Reg(3), src1: Reg(1), src2: Reg(2) }, "add x3, x1, x2"),
        
        // Test case 2: Multiplication
        (Inst::Addi { imm: Imm::new_i32(2), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 2"),
        (Inst::Addi { imm: Imm::new_i32(10), dest: Reg(4), src1: Reg(0) }, "li x4, 10"),
        (Inst::Addi { imm: Imm::new_i32(20), dest: Reg(5), src1: Reg(0) }, "li x5, 20"),
        (Inst::Mul { dest: Reg(6), src1: Reg(4), src2: Reg(5) }, "mul x6, x4, x5"),
        
        // Test case 3: Load/store
        (Inst::Addi { imm: Imm::new_i32(3), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 3"),
        (Inst::Lui { uimm: Imm::new_i32(0x1000), dest: Reg(7) }, "lui x7, 0x1000"),
        (Inst::Sw { offset: Imm::new_i32(0), src: Reg(3), base: Reg(7) }, "sw x3, 0(x7)"),
        (Inst::Lw { offset: Imm::new_i32(0), dest: Reg(8), base: Reg(7) }, "lw x8, 0(x7)"),
    ];

    // Translate the program
    println!("Translating program:");
    let mut pc = 0x1000_u32;
    for (inst, description) in &test_program {
        println!("  0x{:08x}: {}", pc, description);
        
        if let Err(e) = recompiler.translate_instruction(inst, pc, IsCompressed::No) {
            eprintln!("Error translating instruction: {:?}", e);
            break;
        }
        
        pc = pc.wrapping_add(4);
    }
    
    println!("\nTranslation complete!\n");

    // Retrieve and display collected HINT information
    let hints = recompiler.get_hints();
    
    println!("Detected {} test case boundaries:", hints.len());
    for hint in hints {
        println!("  Test case {} at PC 0x{:08x}", hint.value, hint.pc);
    }
    
    println!("\nNote: HINT instructions have no architectural effect.");
    println!("They are used purely as markers for debugging and test identification.");

    // Demonstrate clearing hints
    println!("\nClearing collected hints...");
    recompiler.clear_hints();
    println!("Hints cleared. Current count: {}", recompiler.get_hints().len());

    // Demonstrate toggling
    println!("\nDisabling HINT tracking...");
    recompiler.set_hint_tracking(false);
    
    // Translate another HINT - it won't be tracked
    let pc = 0x2000;
    let hint = Inst::Addi { imm: Imm::new_i32(4), dest: Reg(0), src1: Reg(0) };
    let _ = recompiler.translate_instruction(&hint, pc, IsCompressed::No);
    
    println!("Translated HINT at 0x{:08x}, but it wasn't tracked", pc);
    println!("Current hint count: {}", recompiler.get_hints().len());
    
    println!("\n=== Demo Complete ===");
}
