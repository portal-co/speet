//! Tests for push/pop instruction translations in x86_64 recompiler

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
fn test_push_register_translation() {
    // PUSH RAX: 50
    let bytes = vec![0x50];
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
fn test_push_immediate_translation() {
    // PUSH immediate: 6A 42 (push 0x42)
    let bytes = vec![0x6A, 0x42];
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
fn test_pop_register_translation() {
    // POP RAX: 58
    let bytes = vec![0x58];
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
fn test_push_pop_sequence() {
    // PUSH RAX, POP RBX: 50 5B
    let bytes = vec![0x50, 0x5B];
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
fn test_push_32bit_immediate() {
    // PUSH 32-bit immediate: 68 78 56 34 12 (push 0x12345678)
    let bytes = vec![0x68, 0x78, 0x56, 0x34, 0x12];
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
