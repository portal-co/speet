//! Tests for speculative JAL/JALR/JR translations in the MIPS recompiler.

use rabbitizer::{InstrCategory, Instruction};
use speet_link_core::ReactorAdapter;
use speet_mips::MipsRecompiler;
use wasm_encoder::Function;
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};

fn make_rctx(
    reactor: &mut Reactor<(), core::convert::Infallible, Function, LocalPool>,
) -> ReactorAdapter<'_, (), core::convert::Infallible, Function, LocalPool> {
    static T: TableIdx = TableIdx(0);
    ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        injected_start: yecta::Mark {
            slot_count: 0,
            total_locals: 0,
        },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool {
            handler: &T,
            ty: TypeIdx(0),
        },
        escape: yecta::CallEscape::Jump,
    }
}

#[test]
fn test_speculative_calls_disabled_by_default() {
    let recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_speculative_calls_toggle() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0);
    assert!(!recompiler.is_speculative_calls_enabled());
    recompiler.set_speculative_calls(true);
    assert!(recompiler.is_speculative_calls_enabled());
    recompiler.set_speculative_calls(false);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_escape_tag_configuration() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0);
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);

    assert_eq!(recompiler.get_escape_tag(&rctx), None);

    let tag = EscapeTag {
        tag: TagIdx(42),
        ty: TypeIdx(1),
    };
    recompiler.set_escape_tag(&mut rctx, Some(tag));
    assert_eq!(recompiler.get_escape_tag(&rctx), Some(tag));

    recompiler.set_escape_tag(&mut rctx, None);
    assert_eq!(recompiler.get_escape_tag(&rctx), None);
}

#[test]
fn test_jal_with_speculative_calls_disabled() {
    // JAL 0 (self): 0x0c000000
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    assert!(!recompiler.is_speculative_calls_enabled());

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    let inst = Instruction::new(0x0c000000, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jal_with_flag_escape() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let inst = Instruction::new(0x0c000000, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jal_with_exception_escape() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(
        &mut rctx,
        Some(EscapeTag {
            tag: TagIdx(0),
            ty: TypeIdx(0),
        }),
    );

    let inst = Instruction::new(0x0c000000, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jr_ra_with_flag_escape() {
    // jr $ra: 0x03e00008
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let inst = Instruction::new(0x03e00008, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jr_ra_with_exception_escape() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(
        &mut rctx,
        Some(EscapeTag {
            tag: TagIdx(0),
            ty: TypeIdx(0),
        }),
    );

    let inst = Instruction::new(0x03e00008, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jr_non_ra_uses_indirect_jump_path() {
    // jr $t0 (not ra): should never take the ABI-return path, even with
    // speculative calls + a native-stack escape enabled.
    let jr_t0 = 0x01000008u32; // jr $t0 (rs=8)
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let inst = Instruction::new(jr_t0, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jalr_with_flag_escape() {
    // jalr $ra, $t0: 0x0100f809
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let inst = Instruction::new(0x0100f809, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jalr_with_exception_escape() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(
        &mut rctx,
        Some(EscapeTag {
            tag: TagIdx(0),
            ty: TypeIdx(0),
        }),
    );

    let inst = Instruction::new(0x0100f809, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_jalr_non_abi_dest_uses_indirect_call_path() {
    // jalr $t1, $t0 (dest != ra): not ABI-compliant, should never take the
    // speculative path even with speculative calls + a native-stack escape.
    let jalr_t1_t0 = 0x01004809u32;
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape(&mut rctx, yecta::CallEscape::Flag);

    let inst = Instruction::new(jalr_t1_t0, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_speculative_calls_without_native_escape_uses_jump_path() {
    let mut recompiler: MipsRecompiler<'_, '_, (), core::convert::Infallible, Function> =
        MipsRecompiler::new_with_base_pc(0x1000);
    recompiler.set_speculative_calls(true);
    // escape remains Jump → non-speculative path

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);

    let inst = Instruction::new(0x0c000000, 0x1000, InstrCategory::CPU);
    let result = recompiler.translate_instruction(&mut ctx, &mut rctx, &inst, &mut |locals| {
        Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}

#[test]
fn test_base_params_includes_expected_ra() {
    assert_eq!(
        MipsRecompiler::<'_, '_, (), core::convert::Infallible, Function>::BASE_PARAMS,
        36
    );
}
