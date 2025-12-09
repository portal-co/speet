use speet_mips::{MipsRecompiler, SyscallInfo, BreakInfo, CallbackContext};
use rabbitizer::{Instruction, InstrCategory};
use wasm_encoder::Function;

#[test]
fn test_basic_arithmetic() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test ADD instruction: add $t0, $t1, $t2
    let add_instruction = Instruction::new(0x012A4020, 0x1000, InstrCategory::CPU); // add $t0, $t1, $t2
    
    recompiler.translate_instruction(&add_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_immediate_instructions() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test ADDI instruction: addi $t0, $t1, 42
    let addi_instruction = Instruction::new(0x212A002A, 0x1000, InstrCategory::CPU); // addi $t0, $t1, 42
    
    recompiler.translate_instruction(&addi_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_load_store() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test LW instruction: lw $t0, 4($t1)
    let lw_instruction = Instruction::new(0x8D2A0004, 0x1000, InstrCategory::CPU); // lw $t0, 4($t1)
    
    recompiler.translate_instruction(&lw_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    // Test SW instruction: sw $t0, 4($t1)
    let sw_instruction = Instruction::new(0xAD2A0004, 0x1000, InstrCategory::CPU); // sw $t0, 4($t1)
    
    recompiler.translate_instruction(&sw_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_branch_instructions() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test BEQ instruction: beq $t0, $t1, target
    let beq_instruction = Instruction::new(0x11290004, 0x1000, InstrCategory::CPU); // beq $t0, $t1, 4
    
    recompiler.translate_instruction(&beq_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_jump_instructions() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test J instruction: j target
    let j_instruction = Instruction::new(0x08000010, 0x1000, InstrCategory::CPU); // j 0x40
    eprintln!("debug: j_instruction unique_id = {:?}", j_instruction.unique_id);
    
    recompiler.translate_instruction(&j_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_syscall_callback() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    let mut syscall_called = false;
    let mut syscall_callback = |_: &SyscallInfo, _: &mut CallbackContext<_, _>| {
        syscall_called = true;
    };
    
    recompiler.set_syscall_callback(&mut syscall_callback);
    
    // Test SYSCALL instruction
    let syscall_instruction = Instruction::new(0x0000000C, 0x1000, InstrCategory::CPU); // syscall
    
    recompiler.translate_instruction(&syscall_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    assert!(syscall_called);
}

#[test]
fn test_break_callback() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    let mut break_called = false;
    let mut break_callback = |_: &BreakInfo, _: &mut CallbackContext<_, _>| {
        break_called = true;
    };
    
    recompiler.set_break_callback(&mut break_callback);
    
    // Test BREAK instruction with code 0x123
    let break_instruction = Instruction::new(0x000048CD, 0x1000, InstrCategory::CPU); // break 0x123

    recompiler.translate_instruction(&break_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();

    assert!(break_called);
}

#[test]
fn test_logical_operations() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test AND instruction: and $t0, $t1, $t2
    let and_instruction = Instruction::new(0x012A4024, 0x1000, InstrCategory::CPU); // and $t0, $t1, $t2
    
    recompiler.translate_instruction(&and_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    // Test OR instruction: or $t0, $t1, $t2
    let or_instruction = Instruction::new(0x012A4025, 0x1000, InstrCategory::CPU); // or $t0, $t1, $t2
    
    recompiler.translate_instruction(&or_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_shift_operations() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test SLL instruction: sll $t0, $t1, 2
    let sll_instruction = Instruction::new(0x000A4080, 0x1000, InstrCategory::CPU); // sll $t0, $t1, 2
    
    recompiler.translate_instruction(&sll_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_multiplication_division() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test MULT instruction: mult $t0, $t1
    let mult_instruction = Instruction::new(0x01090018, 0x1000, InstrCategory::CPU); // mult $t0, $t1
    
    recompiler.translate_instruction(&mult_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    // Test DIV instruction: div $t0, $t1
    let div_instruction = Instruction::new(0x0109001A, 0x1000, InstrCategory::CPU); // div $t0, $t1
    
    recompiler.translate_instruction(&div_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_hi_lo_operations() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test MFHI instruction: mfhi $t0
    let mfhi_instruction = Instruction::new(0x00001010, 0x1000, InstrCategory::CPU); // mfhi $t0
    
    recompiler.translate_instruction(&mfhi_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    // Test MFLO instruction: mflo $t0
    let mflo_instruction = Instruction::new(0x00001012, 0x1000, InstrCategory::CPU); // mflo $t0
    
    recompiler.translate_instruction(&mflo_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}