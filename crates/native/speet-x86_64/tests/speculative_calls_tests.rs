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
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape_tag: None,
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
/// path), with speculative calls enabled. Counts `try_table` operators
/// across every produced function via `wasmparser`.
///
/// **Finding from implementing item 4**: yecta's hoisted call region (see
/// `crates/helper/yecta/tests/reactor_tests.rs::test_hoisted_speculative_calls_same_entry`,
/// which verifies the merge *does* happen when two calls are fed into the
/// same `Entry`) only merges calls that land in the same `Entry`. The x86-64
/// driver creates a fresh `Entry` per instruction slot (one-function-per-
/// possible-instruction-slot, see docs/guides/yecta.md §1a) and `handle_call`
/// never establishes a `jmp`-style predecessor edge from the call's slot to
/// "the next instruction's slot" the way an unconditional jump does — so two
/// consecutive native `call`s do **not** currently land in the same `Entry`,
/// and each keeps its own `try_table`. This asserts that *known*, current
/// per-call-site behavior (two calls, two try_tables) rather than a merge
/// that doesn't yet have anywhere to fire from this driver — realizing the
/// benefit here would need a follow-up change to `handle_call` to link
/// consecutive call sites the way `jmp` already links fall-through jumps,
/// which is out of scope for yecta's internal hoisting mechanism itself.
#[test]
fn test_two_back_to_back_calls_each_keep_their_own_try_table() {
    // call +0 (target = next_ip, i.e. the second call's start address)
    // call -5 (target = the second call's own start address)
    let bytes = vec![
        0xE8, 0x00, 0x00, 0x00, 0x00, // call 0x1005
        0xE8, 0xFB, 0xFF, 0xFF, 0xFF, // call 0x1005
    ];
    let mut recompiler = X86Recompiler::<(), core::convert::Infallible>::new();
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

    let mut try_table_count = 0usize;
    let mut call_count = 0usize;
    for f in &fns {
        let raw = f.clone().into_raw_body();
        let reader = wasmparser::BinaryReader::new(&raw, 0);
        let body = wasmparser::FunctionBody::new(reader);
        let mut ops = body.get_operators_reader().expect("valid function body");
        while !ops.eof() {
            match ops.read().expect("valid operator") {
                wasmparser::Operator::TryTable { .. } => try_table_count += 1,
                wasmparser::Operator::Call { .. } => call_count += 1,
                _ => {}
            }
        }
    }

    assert!(call_count >= 2, "both calls must be translated (saw {call_count})");
    assert_eq!(
        try_table_count, call_count,
        "each call currently lands in its own Entry (no jmp-style predecessor \
         edge links consecutive call sites), so each keeps its own try_table; \
         see this test's doc comment for the hoisting-doesn't-fire-here finding"
    );
}
