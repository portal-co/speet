//! Unit tests: decode + translate tiny i686 sequences without panic.

use speet_x86::X86_32Recompiler;
use speet_link_core::ReactorAdapter;
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

fn make_rctx(
    reactor: &mut Reactor<(), core::convert::Infallible, wasm_encoder::Function, LocalPool>,
) -> ReactorAdapter<'_, (), core::convert::Infallible, wasm_encoder::Function, LocalPool> {
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
fn mov_eax_1_ret_translates() {
    // mov eax, 1 ; ret
    let bytes = [0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3];
    let mut rc = X86_32Recompiler::<(), core::convert::Infallible>::new_with_base_eip(0x1000);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    })
    .expect("translate");
    // 1-byte slot stride may mark mid-instruction bytes unsupported; that is
    // expected (mirrors speet-x86_64). Real insn starts must still translate.
}

#[test]
fn call_with_flag_escape_translates() {
    // call +0 (E8 00 00 00 00) — relative call to next insn
    let bytes = [0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3];
    let mut rc = X86_32Recompiler::<(), core::convert::Infallible>::new_with_base_eip(0x1000);
    rc.set_speculative_calls(true);
    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.set_escape(&mut rctx, yecta::CallEscape::Flag);
    let result = rc.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());
}
