//! Tests for memory destination support in x86_64 recompiler

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
fn test_add_memory_destination_immediate() {
    // Test: ADD [rax], 0x42 - add immediate to memory location
    let bytes = vec![0x48, 0x83, 0x00, 0x42]; // ADD [rax], 0x42
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
fn test_add_memory_destination_register() {
    // Test: ADD [rax], rbx - add register to memory location
    let bytes = vec![0x48, 0x01, 0x18]; // ADD [rax], rbx
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
fn test_sub_memory_destination_immediate() {
    // Test: SUB [rax], 0x10 - subtract immediate from memory location
    let bytes = vec![0x48, 0x83, 0x28, 0x10]; // SUB [rax], 0x10
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
fn test_mov_memory_destination_immediate() {
    // Test: MOV [rax], 0x12345678 - move immediate to memory location
    let bytes = vec![0x48, 0xC7, 0x00, 0x78, 0x56, 0x34, 0x12]; // MOV [rax], 0x12345678
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
fn test_mov_memory_destination_register() {
    // Test: MOV [rax], rbx - move register to memory location
    let bytes = vec![0x48, 0x89, 0x18]; // MOV [rax], rbx
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
