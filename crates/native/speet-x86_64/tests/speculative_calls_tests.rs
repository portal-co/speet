//! Tests for speculative call/ret instruction translations in x86_64 recompiler

use speet_link_core::ReactorAdapter;
use speet_x86_64::X86Recompiler;
use yecta::{EscapeTag, LocalPool, Reactor, TableIdx, TagIdx, TypeIdx};

fn make_rctx(reactor: &mut Reactor<(), core::convert::Infallible, wasm_encoder::Function, LocalPool>)
    -> ReactorAdapter<'_, (), core::convert::Infallible, wasm_encoder::Function, LocalPool>
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
fn test_speculative_calls_disabled_by_default() {
    let recompiler = X86Recompiler::<(), core::convert::Infallible>::new();
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_speculative_calls_toggle() {
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Initially disabled
    assert!(!recompiler.is_speculative_calls_enabled());

    // Enable
    recompiler.set_speculative_calls(true);
    assert!(recompiler.is_speculative_calls_enabled());

    // Disable
    recompiler.set_speculative_calls(false);
    assert!(!recompiler.is_speculative_calls_enabled());
}

#[test]
fn test_escape_tag_configuration() {
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);

    // Initially None
    assert_eq!(recompiler.get_escape_tag(&rctx), None);

    // Set escape tag
    let tag = EscapeTag {
        tag: TagIdx(42),
        ty: TypeIdx(1),
    };
    recompiler.set_escape_tag(&mut rctx, Some(tag));
    assert_eq!(recompiler.get_escape_tag(&rctx), Some(tag));

    // Clear escape tag
    recompiler.set_escape_tag(&mut rctx, None);
    assert_eq!(recompiler.get_escape_tag(&rctx), None);
}

#[test]
fn test_call_with_speculative_calls_disabled() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Speculative calls disabled (default)
    assert!(!recompiler.is_speculative_calls_enabled());

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_call_with_speculative_calls_enabled() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(&mut rctx, Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_ret_with_speculative_calls_enabled() {
    // RET instruction: C3
    let bytes = vec![0xC3];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(&mut rctx, Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

#[test]
fn test_speculative_calls_requires_escape_tag() {
    // Direct CALL instruction: E8 05 00 00 00 (call +5)
    let bytes = vec![0xE8, 0x05, 0x00, 0x00, 0x00];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Enable speculative calls but don't set escape tag
    recompiler.set_speculative_calls(true);
    // escape_tag remains None

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    // Should still work but use non-speculative path
    assert!(result.is_ok());
}

#[test]
fn test_ret_with_immediate_speculative() {
    // RET with immediate: C2 08 00 (ret 8) - return and clean up 8 bytes from stack
    let bytes = vec![0xC2, 0x08, 0x00];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();

    // Configure for speculative calls
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(&mut rctx, Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });

    assert!(result.is_ok());
}

/// Two back-to-back direct `call` instructions, both targeting an address
/// inside the translated range (so both go through the speculative-call
/// path), with speculative calls enabled. Counts `try_table`/`call` operators
/// per produced function via `wasmparser`.
///
/// Decode-stride means every byte offset in `bytes` is tried as its own
/// translation slot (see docs/guides/yecta.md §1a) — most of the produced
/// functions are re-decodes starting at a misaligned offset, not the "real"
/// instruction stream, and each such slot keeps its own `try_table`. The
/// slot starting at offset 0 *is* the real entry point: it decodes both
/// calls in one straight-line run (the first call's return falls straight
/// through into the second, with no intervening branch), so yecta's call-
/// region hoisting (item 4 of the yecta optimization plan) merges them into
/// the same `Entry` sharing a single `try_table`. This asserts that merge
/// actually fires: somewhere in the output there's a function with 2+ calls
/// sharing exactly one `try_table`, and merging strictly reduces the total
/// `try_table` count below the total call count.
#[test]
fn test_two_back_to_back_calls_share_one_try_table_when_hoisted() {
    // call +0 (target = next_ip, i.e. the second call's start address)
    // call -5 (target = the second call's own start address)
    let bytes = vec![
        0xE8, 0x00, 0x00, 0x00, 0x00, // call 0x1005
        0xE8, 0xFB, 0xFF, 0xFF, 0xFF, // call 0x1005
    ];
    // base_rip must match the `rip` passed to translate_bytes below so the
    // calls' targets (0x1005) fall inside rip_to_func_idx's translated range
    // and resolve to a real function instead of tripping the out-of-range
    // (oob_jump) trap path.
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new_with_base_rip(0x1000);
    recompiler.set_speculative_calls(true);

    let mut ctx = ();
    let mut reactor = Reactor::default();
    let mut rctx = make_rctx(&mut reactor);
    recompiler.setup_traps(&mut rctx, &mut ctx);
    recompiler.set_escape_tag(&mut rctx, Some(EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    }));

    let result = recompiler.translate_bytes(&mut ctx, &mut rctx, &bytes, 0x1000, &mut |locals| {
        wasm_encoder::Function::new(locals.collect::<Vec<_>>())
    });
    assert!(result.is_ok());

    rctx.reactor.seal_remaining(&mut ctx).unwrap();
    let fns = rctx.reactor.drain_fns();

    let mut per_fn_counts = Vec::new();
    for f in fns.iter() {
        let raw = f.clone().into_raw_body();
        let reader = wasmparser::BinaryReader::new(&raw, 0);
        let body = wasmparser::FunctionBody::new(reader);
        let mut ops = body.get_operators_reader().expect("valid function body");
        let (mut try_tables, mut calls) = (0usize, 0usize);
        while !ops.eof() {
            match ops.read().expect("valid operator") {
                wasmparser::Operator::TryTable { .. } => try_tables += 1,
                wasmparser::Operator::Call { .. } => calls += 1,
                _ => {}
            }
        }
        per_fn_counts.push((calls, try_tables));
    }

    let total_calls: usize = per_fn_counts.iter().map(|&(c, _)| c).sum();
    let total_try_tables: usize = per_fn_counts.iter().map(|&(_, t)| t).sum();

    assert!(total_calls >= 2, "both calls must be translated (saw {total_calls})");
    assert!(
        per_fn_counts.iter().any(|&(c, t)| c >= 2 && t == 1),
        "expected at least one function with 2+ calls sharing a single hoisted \
         try_table (the back-to-back calls at the real entry point's slot); \
         got per-function (calls, try_tables) = {per_fn_counts:?}"
    );
    assert!(
        total_try_tables < total_calls,
        "hoisting should merge at least one pair of calls into a shared \
         try_table, so total try_tables ({total_try_tables}) should be less \
         than total calls ({total_calls})"
    );
}
