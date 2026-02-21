use rabbitizer::{InstrCategory, Instruction};
use speet_mips::MipsRecompiler;
use wasm_encoder::Function;
use yecta::Pool;

// MIPS64 load/store translation smoke test
#[test]
fn test_mips64_load_store() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_full_config(
            Pool {
                table: yecta::TableIdx(0),
                ty: yecta::TypeIdx(0),
            },
            None,
            0,
            true, // enable mips64
        );
    let mut ctx = ();

    // Use existing LW/SW encodings to exercise MIPS64 paths (sign-extend / wrap behaviors)
    let lw_instruction = Instruction::new(0x8D2A0004, 0x1000, InstrCategory::CPU); // lw $t0, 4($t1)
    let sw_instruction = Instruction::new(0xAD2A0004, 0x1004, InstrCategory::CPU); // sw $t0, 4($t1)

    recompiler
        .translate_instruction(&mut ctx, &lw_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();

    recompiler
        .translate_instruction(&mut ctx, &sw_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();

    assert!(recompiler.is_mips64_enabled());
}
