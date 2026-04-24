//! Example demonstrating HINT instruction tracking.

use std::convert::Infallible;

use rv_asm::{Imm, Inst, IsCompressed, Reg};
use speet_link_core::ReactorAdapter;
use speet_riscv::RiscVRecompiler;
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

fn main() {
    println!("RISC-V HINT Instruction Tracking Demo");
    println!("======================================\n");

    let mut recompiler = RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
        0x1000,
        true,  // enable HINT tracking
        false,
        false,
    );

    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    let mut ctx = ();

    println!("HINT tracking is enabled\n");

    let test_program = vec![
        (Inst::Addi { imm: Imm::new_i32(1), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 1"),
        (Inst::Addi { imm: Imm::new_i32(5), dest: Reg(1), src1: Reg(0) }, "li x1, 5"),
        (Inst::Addi { imm: Imm::new_i32(3), dest: Reg(2), src1: Reg(0) }, "li x2, 3"),
        (Inst::Add  { dest: Reg(3), src1: Reg(1), src2: Reg(2) },          "add x3, x1, x2"),
        (Inst::Addi { imm: Imm::new_i32(2), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 2"),
        (Inst::Addi { imm: Imm::new_i32(3), dest: Reg(0), src1: Reg(0) }, "HINT: Test case 3"),
    ];

    println!("Translating program:");
    let mut pc = 0x1000_u32;
    for (inst, description) in &test_program {
        println!("  0x{:08x}: {}", pc, description);
        if let Err(e) = recompiler.translate_instruction(
            &mut ctx, &mut rctx, inst, pc, IsCompressed::No,
            &mut |a| Function::new(a.collect::<Vec<_>>()),
        ) {
            eprintln!("Error: {:?}", e);
            break;
        }
        pc = pc.wrapping_add(4);
    }

    println!("\nTranslation complete!\n");
    let hints = recompiler.get_hints();
    println!("Detected {} test case boundaries:", hints.len());
    for hint in hints {
        println!("  Test case {} at PC 0x{:08x}", hint.value, hint.pc);
    }

    recompiler.clear_hints();
    println!("\nHints cleared. Count: {}", recompiler.get_hints().len());
    println!("\n=== Demo Complete ===");
}
