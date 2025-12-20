//! Integration tests for the Speet recompiler workspace
//!
//! These tests verify that the different crates work together correctly.

use wasm_encoder::{Function, Instruction, ValType};

#[test]
fn test_workspace_builds() {
    // Verify that all workspace members can be imported
    let _yecta_test = test_yecta_integration();
    let _riscv_test = test_riscv_integration();
}

fn test_yecta_integration() -> bool {
    use yecta::{FuncIdx, Reactor};
    
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    reactor.next([(1, ValType::I32)].into_iter(), 0);
    
    // Emit a simple instruction
    reactor.feed(&Instruction::LocalGet(0)).is_ok()
}

fn test_riscv_integration() -> bool {
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Test a simple ADDI instruction
    let inst = Inst::Addi {
        imm: rv_asm::Imm::new_i32(42),
        dest: rv_asm::Reg(1),
        src1: rv_asm::Reg(0),
    };
    
    recompiler
        .translate_instruction(&mut ctx, &inst, 0x1000, IsCompressed::No, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        })
        .is_ok()
}

#[test]
fn test_yecta_and_riscv_together() {
    // This tests that yecta reactor can be used by the RISC-V recompiler
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Translate multiple instructions
    let instructions = vec![
        Inst::Addi {
            imm: rv_asm::Imm::new_i32(5),
            dest: rv_asm::Reg(1),
            src1: rv_asm::Reg(0),
        },
        Inst::Addi {
            imm: rv_asm::Imm::new_i32(3),
            dest: rv_asm::Reg(2),
            src1: rv_asm::Reg(0),
        },
        Inst::Add {
            dest: rv_asm::Reg(3),
            src1: rv_asm::Reg(1),
            src2: rv_asm::Reg(2),
        },
    ];
    
    for (i, inst) in instructions.iter().enumerate() {
        let pc = 0x1000 + (i as u32 * 4);
        assert!(
            recompiler
                .translate_instruction(&mut ctx, inst, pc, IsCompressed::No, &mut |a| Function::new(
                    a.collect::<Vec<_>>()
                ))
                .is_ok(),
            "Failed to translate instruction at PC 0x{:x}",
            pc
        );
    }
}

#[test]
fn test_control_flow_with_yecta() {
    // Test that yecta's control flow management works correctly
    use yecta::{FuncIdx, JumpCallParams, Pool, Reactor, TableIdx, TypeIdx};
    
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(0),
    };
    
    // Create entry function
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    reactor.feed(&Instruction::LocalGet(0)).unwrap();
    reactor.feed(&Instruction::LocalGet(1)).unwrap();
    
    // Jump to another function
    let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    reactor.ji_with_params(params).unwrap();
    
    // Create target function
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    reactor.seal(&Instruction::Unreachable).unwrap();
}

#[test]
fn test_riscv_branch_translation() {
    // Test that RISC-V branch instructions work with yecta's control flow
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Test BEQ (branch if equal)
    let beq = Inst::Beq {
        offset: rv_asm::Imm::new_i32(8),
        src1: rv_asm::Reg(1),
        src2: rv_asm::Reg(2),
    };
    
    assert!(
        recompiler
            .translate_instruction(&mut ctx, &beq, 0x1000, IsCompressed::No, &mut |a| Function::new(
                a.collect::<Vec<_>>()
            ))
            .is_ok()
    );
}

#[test]
fn test_riscv_load_store_translation() {
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Test LW (load word)
    let lw = Inst::Lw {
        offset: rv_asm::Imm::new_i32(0),
        dest: rv_asm::Reg(1),
        base: rv_asm::Reg(2),
    };
    
    assert!(
        recompiler
            .translate_instruction(&mut ctx, &lw, 0x1000, IsCompressed::No, &mut |a| Function::new(
                a.collect::<Vec<_>>()
            ))
            .is_ok()
    );
    
    // Test SW (store word)
    let sw = Inst::Sw {
        offset: rv_asm::Imm::new_i32(4),
        src: rv_asm::Reg(1),
        base: rv_asm::Reg(2),
    };
    
    assert!(
        recompiler
            .translate_instruction(&mut ctx, &sw, 0x1004, IsCompressed::No, &mut |a| Function::new(
                a.collect::<Vec<_>>()
            ))
            .is_ok()
    );
}

#[test]
fn test_riscv_mul_extension() {
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Test MUL (multiply) from M extension
    let mul = Inst::Mul {
        dest: rv_asm::Reg(3),
        src1: rv_asm::Reg(1),
        src2: rv_asm::Reg(2),
    };
    
    assert!(
        recompiler
            .translate_instruction(&mut ctx, &mul, 0x1000, IsCompressed::No, &mut |a| Function::new(
                a.collect::<Vec<_>>()
            ))
            .is_ok()
    );
}

#[test]
fn test_riscv_float_extension() {
    use speet_riscv::RiscVRecompiler;
    use rv_asm::{Inst, IsCompressed, RoundingMode};
    
    let mut recompiler = RiscVRecompiler::<(), std::convert::Infallible, Function>::new_with_base_pc(0x1000);
    let mut ctx = ();

    // Test FADD.S (float add single) from F extension
    let fadd = Inst::FaddS {
        rm: RoundingMode::RoundToNearestTiesToEven,
        dest: rv_asm::FReg(1),
        src1: rv_asm::FReg(2),
        src2: rv_asm::FReg(3),
    };
    
    assert!(
        recompiler
            .translate_instruction(&mut ctx, &fadd, 0x1000, IsCompressed::No, &mut |a| Function::new(
                a.collect::<Vec<_>>()
            ))
            .is_ok()
    );
}
