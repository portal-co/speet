use speet_mips::MipsRecompiler;
use rabbitizer::{Instruction, InstrCategory};
use wasm_encoder::Function;

#[test]
fn test_simple_add() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test simple ADD instruction: add $t0, $t1, $t2 (0x012A4020)
    let add_instruction = Instruction::new(0x012A4020, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&add_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_simple_addi() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test simple ADDI instruction: addi $t0, $t1, 1 (0x21290001)
    let addi_instruction = Instruction::new(0x21290001, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&addi_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_simple_and() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test simple AND instruction: and $t0, $t1, $t2 (0x012A4024)
    let and_instruction = Instruction::new(0x012A4024, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&and_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_simple_or() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test simple OR instruction: or $t0, $t1, $t2 (0x012A4025)
    let or_instruction = Instruction::new(0x012A4025, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&or_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_simple_syscall() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    let mut syscall_called = false;
    let mut syscall_callback = |_: &speet_mips::SyscallInfo, _: &mut speet_mips::CallbackContext<_, _>| {
        syscall_called = true;
    };
    
    recompiler.set_syscall_callback(&mut syscall_callback);
    
    // Test SYSCALL instruction: syscall (0x0000000C)
    let syscall_instruction = Instruction::new(0x0000000C, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&syscall_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
    
    assert!(syscall_called);
}

#[test]
fn test_simple_jr() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test JR instruction: jr $ra (0x03E00008) - jump to return address
    let jr_instruction = Instruction::new(0x03E00008, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&jr_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}

#[test]
fn test_simple_jalr() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x1000);
    
    // Test JALR instruction: jalr $ra, $t0 (0x01800008) - jump and link
    let jalr_instruction = Instruction::new(0x01800008, 0x1000, InstrCategory::CPU);
    
    recompiler.translate_instruction(&jalr_instruction, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    }).unwrap();
}