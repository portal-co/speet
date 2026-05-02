//! Tests for pushf/popf instruction translations in x86_64 recompiler

use speet_link_core::ReactorAdapter;
use speet_x86_64::X86Recompiler;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn make_rctx(reactor: &mut Reactor<(), core::convert::Infallible, wasm_encoder::Function, LocalPool>)
    -> ReactorAdapter<'_, (), core::convert::Infallible, wasm_encoder::Function, LocalPool>
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

#[test]
fn test_pushfq_translation() {
    // PUSHFQ (64-bit flags push)
    let bytes = vec![0x9C]; // PUSHF/Q
    let mut recompiler = X86Recompiler::new();

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_pushfd_translation() {
    // PUSHFD (32-bit flags push) - operand-size override prefix 0x66
    let bytes = vec![0x66, 0x9C]; // 0x66 prefix + PUSHF
    let mut recompiler = X86Recompiler::new();

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_popfq_translation() {
    // POPFQ (64-bit flags pop)
    let bytes = vec![0x9D]; // POPF/Q
    let mut recompiler = X86Recompiler::new();

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_popfd_translation() {
    // POPFD (32-bit flags pop) - operand-size override prefix 0x66
    let bytes = vec![0x66, 0x9D]; // 0x66 prefix + POPF
    let mut recompiler = X86Recompiler::new();

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}
