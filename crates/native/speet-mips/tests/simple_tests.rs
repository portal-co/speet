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
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape: yecta::CallEscape::Jump,
    }
}

#[test]
fn test_simple_add() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test simple ADD instruction: add $t0, $t1, $t2 (0x012A4020)
    let add_instruction = Instruction::new(0x012A4020, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &add_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}

#[test]
fn test_simple_addi() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test simple ADDI instruction: addi $t0, $t1, 1 (0x21290001)
    let addi_instruction = Instruction::new(0x21290001, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &addi_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}

#[test]
fn test_simple_and() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test simple AND instruction: and $t0, $t1, $t2 (0x012A4024)
    let and_instruction = Instruction::new(0x012A4024, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &and_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}

#[test]
fn test_simple_or() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test simple OR instruction: or $t0, $t1, $t2 (0x012A4025)
    let or_instruction = Instruction::new(0x012A4025, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &or_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}

#[test]
fn test_simple_syscall() {
    // The callback borrow lives in the recompiler's 'cb slot, so the
    // callback must be declared (and thus dropped) AFTER the recompiler.
    // The flag is shared via Cell so the borrow isn't two-way.
    let syscall_called = std::rc::Rc::new(core::cell::Cell::new(false));
    let flag = syscall_called.clone();
    let mut syscall_callback = move |_: &speet_mips::SyscallInfo,
                                _ctx: &mut (),
                                _: &mut speet_mips::CallbackContext<'_, (), core::convert::Infallible>| {
        flag.set(true);
    };

    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();

    recompiler.set_syscall_callback(&mut syscall_callback);

    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test SYSCALL instruction: syscall (0x0000000C)
    let syscall_instruction = Instruction::new(0x0000000C, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &syscall_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();

    assert!(syscall_called.get());
}

#[test]
fn test_simple_jr() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test JR instruction: jr $ra (0x03E00008) - jump to return address
    let jr_instruction = Instruction::new(0x03E00008, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &jr_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}

#[test]
fn test_simple_jalr() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, _> =
        MipsRecompiler::new_with_base_pc(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Test JALR instruction: jalr $ra, $t0 (0x01800008) - jump and link
    let jalr_instruction = Instruction::new(0x01800008, 0x1000, InstrCategory::CPU);

    recompiler
        .translate_instruction(&mut ctx, &mut rctx, &jalr_instruction, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .unwrap();
}
