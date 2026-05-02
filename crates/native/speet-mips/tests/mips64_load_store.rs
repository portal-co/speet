use rabbitizer::{InstrCategory, Instruction};
use speet_link_core::ReactorAdapter;
use speet_mips::MipsRecompiler;
use wasm_encoder::Function;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn make_rctx(reactor: &mut Reactor<(), core::convert::Infallible, Function, LocalPool>)
    -> ReactorAdapter<'_, (), core::convert::Infallible, Function, LocalPool>
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

// MIPS64 load/store translation smoke test
#[test]
fn test_mips64_load_store() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_full_config(
            0x1000,
            true, // enable mips64
        );
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);

    // Use existing LW/SW encodings to exercise MIPS64 paths (sign-extend / wrap behaviors)
    let lw_instruction = Instruction::new(0x8D2A0004, 0x1000, InstrCategory::CPU); // lw $t0, 4($t1)
    let sw_instruction = Instruction::new(0xAD2A0004, 0x1004, InstrCategory::CPU); // sw $t0, 4($t1)

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &lw_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &sw_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();

    assert!(recompiler.is_mips64_enabled());
}
