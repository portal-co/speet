use speet_mips::MipsRecompiler;
use rabbitizer::{Instruction, InstrCategory};
use wasm_encoder::Function;

#[test]
fn test_load_store_variants() {
    let mut recompiler: MipsRecompiler<'_, '_, core::convert::Infallible, _> = MipsRecompiler::new_with_base_pc(0x2000);

    // LB: lb $t0, 1($t1) -> opcode 0x812A0001 (LB with rt=10, rs=9, imm=1)
    let lb = Instruction::new(0x812A0001, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&lb, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // LBU: lbu $t0, 1($t1) -> opcode 0x912A0001
    let lbu = Instruction::new(0x912A0001, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&lbu, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // LH: lh $t0, 2($t1) -> opcode 0x852A0002
    let lh = Instruction::new(0x852A0002, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&lh, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // LHU: lhu $t0, 2($t1) -> opcode 0x952A0002
    let lhu = Instruction::new(0x952A0002, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&lhu, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // SB: sb $t0, 3($t1) -> opcode 0xA12A0003
    let sb = Instruction::new(0xA12A0003, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&sb, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // SH: sh $t0, 4($t1) -> opcode 0xA52A0004
    let sh = Instruction::new(0xA52A0004, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&sh, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    // LW/SW already tested in basic_tests, but exercise again at new PC
    let lw = Instruction::new(0x8D2A0004, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&lw, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();

    let sw = Instruction::new(0xAD2A0004, 0x2000, InstrCategory::CPU);
    recompiler.translate_instruction(&sw, &mut |locals| Function::new(locals.collect::<Vec<_>>())).unwrap();
}
