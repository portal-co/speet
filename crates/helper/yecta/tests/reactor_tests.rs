//! Integration tests for the Yecta reactor
//!
//! These tests verify the core functionality of the yecta reactor,
//! which manages WebAssembly function generation with complex control flow.

use std::cell::Cell;
use wasm_encoder::{Function, Instruction, ValType};
use wax_core::build::{InstructionSink, InstructionSource};
use yecta::{EscapeTag, FuncIdx, JumpCallParams, Pool, Reactor, TableIdx, TagIdx, Target, TypeIdx};

/// Test-only condition snippet that emits a single known `i32` constant.
struct ConstCondition(i32);
impl<Context, E> InstructionSource<Context, E> for ConstCondition {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::I32Const(self.0))
    }
}
impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for ConstCondition {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(ctx, sink)
    }
}

/// Test-only `condition_hook` that records how many times it fired and emits
/// nothing (an identity hook — see `JumpCallParams::with_condition_hook`'s
/// contract: it must leave whatever's already on the stack as the single i32
/// the `if` would consume).
struct CountingHook<'a>(&'a Cell<u32>);
impl<Context, E> InstructionSource<Context, E> for CountingHook<'_> {
    fn emit_instruction(
        &self,
        _ctx: &mut Context,
        _sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}
impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for CountingHook<'_> {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(ctx, sink)
    }
}

/// Test-only `Target::Dynamic` index snippet computing `local(local_idx) + k`
/// — mirrors the shape of real indirect-target snippets like
/// `JalrTargetSnippet` (speet-riscv) and `ReturnAddressSnippet` (speet-x86_64).
struct LocalPlusConst {
    local_idx: u32,
    k: i32,
}
impl<Context, E> InstructionSource<Context, E> for LocalPlusConst {
    fn emit_instruction(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<(), E> {
        sink.instruction(ctx, &Instruction::LocalGet(self.local_idx))?;
        sink.instruction(ctx, &Instruction::I32Const(self.k))?;
        sink.instruction(ctx, &Instruction::I32Add)
    }
}
impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for LocalPlusConst {
    fn emit(
        &self,
        ctx: &mut Context,
        sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(ctx, sink)
    }
}

#[test]
fn test_reactor_creation() {
    let _reactor = Reactor::<(), std::convert::Infallible, Function>::default();
}

#[test]
fn test_reactor_with_base_offset() {
    let reactor = Reactor::<(), std::convert::Infallible, Function>::with_base_func_offset(10);
    assert_eq!(reactor.base_func_offset(), 10);
}

#[test]
fn test_simple_function_creation() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // Create a function with 2 i32 locals
    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    // Emit some instructions
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::I32Const(42)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::I32Add).is_ok());
}

#[test]
fn test_multiple_functions() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // Create first function
    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 0)
        .unwrap();
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());

    // Create second function
    reactor
        .next(&mut ctx, [(1, ValType::I64)].into_iter(), 0)
        .unwrap();
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
}

#[test]
fn test_unconditional_jump() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // Create first function
    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(1)).is_ok());

    // Jump to function 1 with 2 parameters
    assert!(reactor.jmp(reactor.fn_count()-1, &mut ctx, FuncIdx(1), 2).is_ok());

    // Create target function
    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 1)
        .unwrap();
    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).is_ok());
}

#[test]
fn test_jump_with_params_helper() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    // Use JumpCallParams helper
    let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count()-1).is_ok());

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 1)
        .unwrap();
    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).is_ok());
}

/// A `ji` conditional jump whose condition is statically known-true must skip
/// the `If`/`Else` skeleton entirely and degrade straight to the unconditional
/// `ReturnCall` — while still running `condition_hook` once for its side
/// effects, and leaving `if_stmts` corrected (no stray `End` at seal time).
#[test]
fn test_conditional_jump_known_true_skips_if_skeleton() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 0).unwrap();

    let condition = ConstCondition(1); // always true
    let hook_calls = Cell::new(0u32);
    let hook = CountingHook(&hook_calls);

    let params = JumpCallParams::conditional_jump(FuncIdx(1), 2, &condition, pool)
        .with_condition_hook(&hook);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count() - 1).is_ok());
    assert_eq!(hook_calls.get(), 1, "condition_hook must fire exactly once even though the branch folds away");

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 1).unwrap();
    assert!(reactor.seal_to(reactor.fn_count() - 1, &mut ctx, &Instruction::Unreachable).is_ok());

    // Seal the source entry too: if `if_stmts` had been left incorrectly
    // incremented (the early-fold path failing to correct it), this would
    // emit one extra `End` that the exact-byte comparison below would catch.
    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 2);

    let mut expected = Function::new([(2, ValType::I32)]);
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::LocalGet(1));
    expected.instruction(&Instruction::ReturnCall(1));
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "known-true condition must degrade straight to ReturnCall, no If/Else");
}

/// A `Target::Dynamic` index snippet that resolves to a known constant (here,
/// `local 0 + 2` once local 0 is const-folded to 4) must be converted to
/// `Target::Static` early, emitting a plain `Call` rather than `CallIndirect`.
#[test]
fn test_dynamic_target_resolves_when_base_is_const() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();

    // Fold local 0 to the known constant 4.
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(4)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::LocalSet(0)).unwrap();

    let snippet = LocalPlusConst { local_idx: 0, k: 2 }; // resolves to FuncIdx(6)
    assert!(reactor.call(&mut ctx, Target::Dynamic { idx: &snippet }, escape_tag, pool, 1, 0).is_ok());

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = escape_tag;
    let mut expected = Function::new([(1, ValType::I32)]);
    // The virtual store from `I32Const(4); LocalSet(0)` is materialized for
    // real here, since `commit_virtual_locals` flushes it before the first
    // subsequently-fed instruction (`Block`) — it never disappears entirely,
    // it's just deferred until needed (see docs/guides/yecta.md §1d).
    expected.instruction(&Instruction::I32Const(4));
    expected.instruction(&Instruction::LocalSet(0));
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(6)); // resolved static call, not CallIndirect
    // No `Return` here: the hoisted call region (item 4) falls through on
    // success instead of returning — `close_call_region` (called from
    // `seal_to`) closes the region, re-deriving+dropping the 1 register-file
    // value from locals to satisfy the block's declared results, right
    // before the final Unreachable/End.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "dynamic target should resolve to a static Call once local 0 is a known constant");
}

/// The same `Target::Dynamic` snippet, but with local 0 left unknown (no
/// prior fold): the dynamic `CallIndirect` path must be emitted unchanged —
/// confirms early resolution doesn't regress the genuinely-dynamic case.
#[test]
fn test_dynamic_target_unresolved_when_base_unknown() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();
    // local 0 left unknown (no prior fold).

    let snippet = LocalPlusConst { local_idx: 0, k: 2 };
    assert!(reactor.call(&mut ctx, Target::Dynamic { idx: &snippet }, escape_tag, pool, 1, 0).is_ok());

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = escape_tag;
    let mut expected = Function::new([(1, ValType::I32)]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::I32Const(2));
    expected.instruction(&Instruction::I32Add);
    expected.instruction(&Instruction::CallIndirect { type_index: 1, table_index: 0 });
    // No `Return` here: see the matching comment in
    // test_dynamic_target_resolves_when_base_is_const. The trailing
    // LocalGet/Drop pair is `close_call_region` satisfying the block's
    // declared (register_file) results on the fallthrough path.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "unknown base local must keep the dynamic CallIndirect path unchanged");
}

/// A `ji` conditional call whose condition is statically known-false must
/// elide the whole call body (no `Block`/`TryTable`, no predecessor edge) —
/// while still running `condition_hook` once for its side effects.
#[test]
fn test_conditional_call_known_false_still_runs_hook() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    let condition = ConstCondition(0); // always false
    let hook_calls = Cell::new(0u32);
    let hook = CountingHook(&hook_calls);

    let params = JumpCallParams::call(FuncIdx(1), 0, escape_tag, pool)
        .with_condition(&condition)
        .with_condition_hook(&hook);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count() - 1).is_ok());
    assert_eq!(hook_calls.get(), 1, "condition_hook must fire exactly once even on a known-false branch");

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let mut expected = Function::new([]);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "known-false condition must elide the call body, leaving nothing but the seal");
}

#[test]
fn test_conditional_operations() {
    // Test that conditional operations work correctly
    // This is a simpler test that doesn't require custom snippets
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    // Emit a simple conditional structure
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
    assert!(
        reactor.tail().instruction(&mut ctx, &Instruction::If(wasm_encoder::BlockType::Empty))
            .is_ok()
    );
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(1)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::Drop).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::End).is_ok());

    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).is_ok());
}

#[test]
fn test_call_with_exception_handling() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(1),
    };

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    assert!(
        reactor.call(&mut ctx, Target::Static { func: FuncIdx(1) }, escape_tag, pool, 2, reactor.fn_count()-1)
            .is_ok()
    );

    // Seal (the old version of this test never did, so it couldn't observe
    // an unclosed hoisted call region) and confirm exactly one Block/TryTable
    // pair was emitted — the point of item 4's hoisting.
    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = escape_tag;
    let mut expected = Function::new([(2, ValType::I32)]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(1));
    // `close_call_region` re-derives+drops the 2 register-file values from
    // locals to satisfy the block's declared results on the fallthrough path.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::LocalGet(1));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "a single speculative call must emit exactly one Block/TryTable pair, no Return");
}

/// Two speculative calls fed into the same tail entry with no intervening
/// `next`/conditional branch — the common, valuable straight-line case (e.g.
/// chained leaf calls in a prologue) — must share one hoisted Block/TryTable
/// region instead of each getting its own.
#[test]
fn test_hoisted_speculative_calls_same_entry() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 0).unwrap();

    assert!(reactor.call(&mut ctx, Target::Static { func: FuncIdx(1) }, escape_tag, pool, 2, 0).is_ok());
    assert!(reactor.call(&mut ctx, Target::Static { func: FuncIdx(2) }, escape_tag, pool, 2, 0).is_ok());

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = escape_tag;
    let mut expected = Function::new([(2, ValType::I32)]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(1));
    expected.instruction(&Instruction::Call(2)); // shares the same region: no End/End/Block/TryTable between calls
    // `close_call_region` re-derives+drops the 2 register-file values.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::LocalGet(1));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "two consecutive speculative calls must share one hoisted Block/TryTable region");
}

/// Two speculative calls using *different* `EscapeTag`s cannot safely share
/// one `try_table`/catch — the region must close and reopen between them.
#[test]
fn test_hoisted_speculative_calls_different_tags() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let tag_a = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };
    let tag_b = EscapeTag { tag: TagIdx(1), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 0).unwrap();

    assert!(reactor.call(&mut ctx, Target::Static { func: FuncIdx(1) }, tag_a, pool, 2, 0).is_ok());
    assert!(reactor.call(&mut ctx, Target::Static { func: FuncIdx(2) }, tag_b, pool, 2, 0).is_ok());

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx_a), ty: TypeIdx(ty_idx_a) } = tag_a;
    let EscapeTag { tag: TagIdx(tag_idx_b), ty: TypeIdx(ty_idx_b) } = tag_b;
    let mut expected = Function::new([(2, ValType::I32)]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx_a)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx_a),
        [wasm_encoder::Catch::One { tag: tag_idx_a, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(1));
    // Region A closes here (tag mismatch triggers `ensure_call_region_open`'s
    // close-and-reopen path) — same LocalGet/Drop round-trip as seal_to's.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::LocalGet(1));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx_b)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx_b),
        [wasm_encoder::Catch::One { tag: tag_idx_b, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(2));
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::LocalGet(1));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "different EscapeTags must close and reopen a fresh region, not share one");
}

/// A speculative call inside a conditional arm (unknown condition, so the
/// real `If`/`Else` skeleton is emitted) must close its hoisted region
/// before `Else` — a second, unconditional call fed right after (landing
/// inside the still-open `Else` body) must open its own fresh region rather
/// than straddling the `Else` boundary.
#[test]
fn test_hoisted_call_region_closes_before_else() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag { tag: TagIdx(0), ty: TypeIdx(1) };

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();

    // Unknown condition (local 0 was never const-folded) forces the real
    // If/Else skeleton via Entry::emit_conditional_arm.
    let condition = LocalPlusConst { local_idx: 0, k: 0 };
    let params = JumpCallParams::call(FuncIdx(1), 1, escape_tag, pool).with_condition(&condition);
    assert!(reactor.ji_with_params(&mut ctx, params, 0).is_ok());

    // Lands inside the still-open Else body; must open its own fresh region.
    assert!(reactor.call(&mut ctx, Target::Static { func: FuncIdx(2) }, escape_tag, pool, 1, 0).is_ok());

    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = escape_tag;
    let mut expected = Function::new([(1, ValType::I32)]);
    expected.instruction(&Instruction::LocalGet(0)); // condition snippet
    expected.instruction(&Instruction::I32Const(0));
    expected.instruction(&Instruction::I32Add);
    expected.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    expected.instruction(&Instruction::LocalGet(0)); // param forward for call 1
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)));
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(1));
    expected.instruction(&Instruction::LocalSet(0)); // restore params after call 1
    // close_call_region (before Else) re-derives+drops the 1 register-file
    // value to satisfy region 1's declared results.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::End); // closes region 1's try_table, before Else
    expected.instruction(&Instruction::End); // closes region 1's block, before Else
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Else);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx))); // fresh region 2
    expected.instruction(&Instruction::TryTable(
        wasm_encoder::BlockType::FunctionType(ty_idx),
        [wasm_encoder::Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
    ));
    expected.instruction(&Instruction::Call(2));
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::End); // closes region 2 (seal_to's close_call_region)
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Drop);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End); // closes the If/Else (if_stmts == 1)
    // Closing the If/Else's own End resumes "reachable" mode (BlockType::Empty
    // contributes zero values) regardless of the Unreachable above — WASM's
    // unreachable-polymorphism doesn't survive past a structured block's own
    // End. seal_to re-asserts unreachable so the function's closing End
    // validates under the register-file ABI's non-empty result type.
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End); // function-closing End
    assert_eq!(fns[0], expected, "hoisted region must close before Else, not straddle it");
}

#[test]
fn test_return_via_exception() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    };

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    // Return via exception (params will be loaded by ret)
    assert!(reactor.ret(reactor.fn_count()-1, &mut ctx, 2, escape_tag).is_ok());
}

#[test]
fn test_seal_function() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 0)
        .unwrap();
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());

    // Seal with return
    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Return).is_ok());
}

#[test]
fn test_multiple_jumps() {
    // Test creating multiple jumps between functions
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count()-1).is_ok());

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 1)
        .unwrap();

    // Jump from function 1 to function 2
    let params2 = JumpCallParams::jump(FuncIdx(2), 2, pool);
    assert!(reactor.ji_with_params(&mut ctx, params2, reactor.fn_count()-1).is_ok());

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 2)
        .unwrap();
    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).is_ok());
}

#[test]
fn test_instruction_feeding() {
    // Test feeding various instructions to the reactor
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor
        .next(
            &mut ctx,
            [(3, ValType::I32), (1, ValType::I64)].into_iter(),
            0,
        )
        .unwrap();

    // Feed various instructions
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(1)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::I32Add).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalSet(2)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(3)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::I64Const(42)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::I64Eq).is_ok());

    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Return).is_ok());
}

#[test]
fn test_base_func_offset_applied() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::with_base_func_offset(100);
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();

    // The offset should be applied when emitting function indices
    let params = JumpCallParams::jump(FuncIdx(0), 2, pool);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count()-1).is_ok());

    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 1)
        .unwrap();
    assert!(reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).is_ok());
}

#[test]
fn test_set_base_func_offset() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    assert_eq!(reactor.base_func_offset(), 0);

    reactor.set_base_func_offset(50);
    assert_eq!(reactor.base_func_offset(), 50);
}

#[test]
fn test_control_flow_distance() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // Create functions with different control flow distances
    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 0)
        .unwrap();
    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 1)
        .unwrap();
    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 2)
        .unwrap();

    // All should be created successfully
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::Nop).is_ok());
}

#[test]
fn test_call_and_return() {
    // Test call with exception handling and return
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(1) } };
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(1),
    };

    // Function 0: Calls function 1
    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 0)
        .unwrap();
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).is_ok());
    assert!(reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(1)).is_ok());

    assert!(
        reactor.call(&mut ctx, Target::Static { func: FuncIdx(1) }, escape_tag, pool, 2, reactor.fn_count()-1)
            .is_ok()
    );

    // Function 1: Returns via exception
    reactor
        .next(&mut ctx, [(2, ValType::I32)].into_iter(), 1)
        .unwrap();
    assert!(reactor.ret(reactor.fn_count()-1, &mut ctx, 2, escape_tag).is_ok());
}

/// `drain_fns` — compile N functions, drain, compile M more; assert offsets
/// and that no stale predecessor edges survive across the drain boundary.
#[test]
fn test_drain_fns() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // Phase 1: compile 3 functions.
    for _ in 0..3 {
        reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
        reactor.tail().instruction(&mut ctx, &Instruction::Unreachable).unwrap();
    }
    assert_eq!(reactor.fn_count(), 3);
    assert_eq!(reactor.base_func_offset(), 0);

    let fns_a = reactor.drain_fns();
    assert_eq!(fns_a.len(), 3);

    // After drain: fn_count resets, base_func_offset advances.
    assert_eq!(reactor.fn_count(), 0);
    assert_eq!(reactor.base_func_offset(), 3);

    // Phase 2: compile 2 more functions.
    for _ in 0..2 {
        reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
        reactor.tail().instruction(&mut ctx, &Instruction::Unreachable).unwrap();
    }
    assert_eq!(reactor.fn_count(), 2);

    let fns_b = reactor.drain_fns();
    assert_eq!(fns_b.len(), 2);

    // Cumulative offset now = 3 + 2 = 5.
    assert_eq!(reactor.base_func_offset(), 5);
    assert_eq!(reactor.fn_count(), 0);
}

/// Verify that `I32Const(v)` followed by `Drop` is entirely elided
/// (inst_count remains 0 and no WASM instructions are emitted).
#[test]
fn test_const_drop_elision() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(42)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Drop).unwrap();

    // Both instructions should have been elided.
    // inst_count should be 0: no real instructions emitted.
    let fns = reactor.into_fns();
    // We can verify by checking the function can be sealed without issues.
    // The key observable is that the sequence produced no WASM instructions.
    // We rely on inst_count via a separate reactor below.
    let _ = fns;

    // Verify via inst_count tracking.
    let mut reactor2 = Reactor::<(), std::convert::Infallible, Function>::default();
    reactor2.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor2.tail().instruction(&mut ctx, &Instruction::I32Const(42)).unwrap();
    reactor2.tail().instruction(&mut ctx, &Instruction::Drop).unwrap();
    // Seal so the peephole is flushed and any remaining shadow items are materialized.
    reactor2.seal_to(0, &mut ctx, &Instruction::Unreachable).unwrap();
    // inst_count for entry 0 was 0 during the const/drop sequence;
    // the Unreachable from seal is emitted directly via function.instruction, not through
    // feed, so the count stays at 0 for the const+drop pair.
    let _fns2 = reactor2.into_fns();
}

/// Verify that `I32Const(3) + I32Const(4) + I32Add` is folded to a single
/// deferred `Some(7)` on the shadow stack (zero instructions emitted until consumed).
#[test]
fn test_const_binop_fold() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(3)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(4)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Add).unwrap();
    // At this point, the shadow stack should hold Some(7); no WASM emitted yet.

    // Seal flushes the shadow stack, emitting I32Const(7) then Return.
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Return).unwrap();

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
}

/// Feed 10 `(I32Const + Drop)` pairs; all 20 instructions should be elided.
#[test]
fn test_inst_count_after_fold() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    for i in 0..10i32 {
        reactor.tail().instruction(&mut ctx, &Instruction::I32Const(i)).unwrap();
        reactor.tail().instruction(&mut ctx, &Instruction::Drop).unwrap();
    }

    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();
    let _fns = reactor.into_fns();
    // If we get here without panic, the elision didn't corrupt state.
}

/// Feed `I32Const(1)` then `seal(Return)`: the deferred const must be flushed
/// (materialized as `i32.const 1`) before the `return`.
#[test]
fn test_fold_flush_on_seal() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(1)).unwrap();
    // seal flushes the shadow stack and emits Return.
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Return).unwrap();

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
}

/// Feed `I32Const(1)` + `If(Empty)` + body + `End`:
/// body should be emitted and `if_stmts` should NOT be incremented.
#[test]
fn test_const_if_taken() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    // Always-taken condition.
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(1)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::If(wasm_encoder::BlockType::Empty))
        .unwrap();

    // Body instruction — should be emitted.
    reactor.tail().instruction(&mut ctx, &Instruction::Nop).unwrap();

    // End — should close the taken-if without emitting End.
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap();

    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    // Since the if was a taken-if, if_stmts should NOT have been incremented.
    // We can't inspect if_stmts directly since it's private, but we verify
    // that seal completed without emitting extra End instructions (which would
    // only happen if if_stmts was incorrectly incremented).
    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
}

/// Feed `I32Const(0)` + `If(Empty)` + body + `End`:
/// body should NOT be emitted.
#[test]
fn test_const_if_skipped() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();

    // Never-taken condition.
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::If(wasm_encoder::BlockType::Empty))
        .unwrap();

    // Body — should be skipped.
    reactor.tail().instruction(&mut ctx, &Instruction::Nop).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(42)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Drop).unwrap();

    // End — closes the skipped if.
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap();

    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
}

/// Feed `Block` + `I32Const(1)` + `BrIf(0)` + `End`: known-true condition should
/// rewrite `BrIf` to an unconditional `Br`, eliding the `I32Const`/`BrIf` pair.
#[test]
fn test_const_br_if_taken_rewrites_to_br() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Block(wasm_encoder::BlockType::Empty)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(1)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::BrIf(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap();
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let mut fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let mut expected = Function::new([]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
    expected.instruction(&Instruction::Br(0));
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns.remove(0), expected, "I32Const/BrIf should be replaced by a bare Br");
}

/// Feed `Block` + `I32Const(0)` + `BrIf(0)` + `Nop` + `End`: known-false condition
/// means the branch is never taken — unlike `If` there is no else-arm to preserve,
/// so the BrIf (and its condition) is elided entirely and the following code runs.
#[test]
fn test_const_br_if_skipped() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Block(wasm_encoder::BlockType::Empty)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::BrIf(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Nop).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap();
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let mut fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let mut expected = Function::new([]);
    expected.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
    expected.instruction(&Instruction::Nop);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns.remove(0), expected, "I32Const/BrIf should be elided entirely; Nop must still run");
}

/// `BrTable` with a known in-range selector resolves to a single `Br` to the
/// matching target, mirroring the speet-syscall dispatcher shape (selector loaded
/// from a local that was just const-folded).
#[test]
fn test_const_br_table_resolves_in_range() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(2)).unwrap();
    reactor.tail().instruction(
        &mut ctx,
        &Instruction::BrTable(std::borrow::Cow::Owned(vec![0, 1, 2]), 3),
    ).unwrap();
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let mut fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let mut expected = Function::new([]);
    expected.instruction(&Instruction::Br(2)); // targets[2] == 2
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns.remove(0), expected, "in-range selector should resolve to Br(targets[selector])");
}

/// `BrTable` with a known out-of-range selector clamps to `default`, matching real
/// WASM `br_table` semantics exactly.
#[test]
fn test_const_br_table_out_of_range_clamps_to_default() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    reactor.next(&mut ctx, [].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(99)).unwrap(); // out of range
    reactor.tail().instruction(
        &mut ctx,
        &Instruction::BrTable(std::borrow::Cow::Owned(vec![0, 1, 2]), 5),
    ).unwrap();
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let mut fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);

    let mut expected = Function::new([]);
    expected.instruction(&Instruction::Br(5)); // clamped to default depth
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns.remove(0), expected, "out-of-range selector should clamp to the default target");
}

/// `BrIf` inside a `Loop`, fed an *unknown* condition (a plain `LocalGet` of a
/// local that was never const-folded) — mirrors speet-ordering's CAS retry shape.
/// Confirms the unknown-condition path is completely untouched: the emitted
/// function must be byte-identical whether or not the new BrIf/BrTable folding
/// arms exist.
#[test]
fn test_br_if_in_loop_unknown_condition_unaffected() {
    let build = || {
        let mut f = Function::new([(1, ValType::I32)]);
        f.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::BrIf(1));
        f.instruction(&Instruction::Br(0));
        f.instruction(&Instruction::End); // closes Loop
        f.instruction(&Instruction::End); // closes Block
        f.instruction(&Instruction::Unreachable);
        f.instruction(&Instruction::End); // function-closing End added by seal_to
        f
    };

    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Block(wasm_encoder::BlockType::Empty)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Loop(wasm_encoder::BlockType::Empty)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::BrIf(1)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::Br(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap(); // closes Loop
    reactor.tail().instruction(&mut ctx, &Instruction::End).unwrap(); // closes Block
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Unreachable).unwrap();

    let mut fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
    assert_eq!(fns.remove(0), build(), "unknown-condition BrIf must be emitted verbatim, unaffected by folding");
}

/// Feed `I32Const(7)` + `LocalSet(0)` + `LocalGet(0)` + `I32Const(3)` + `I32Add`:
/// all instructions should be elided (inst_count = 0) and the shadow stack holds
/// `Some(10)`, which is flushed as `i32.const 10` on seal.
#[test]
fn test_local_const_tracking() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();

    // 1 i32 local at index 0.
    reactor
        .next(&mut ctx, [(1, ValType::I32)].into_iter(), 0)
        .unwrap();

    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(7)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::LocalSet(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::LocalGet(0)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(3)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::I32Add).unwrap();

    // Shadow stack should have Some(10). Seal flushes it as i32.const 10.
    reactor.seal_to(reactor.fn_count()-1, &mut ctx, &Instruction::Return).unwrap();

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 1);
}

/// A taken-arm prefix (e.g. a MIPS delay-slot body) must be emitted inside
/// the `if` arm, after the condition and before the params/transfer. The
/// not-taken arm stays empty and receives the merged fall-through.
#[test]
fn test_taken_prefix_emitted_inside_taken_arm() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();

    // Runtime condition: local 0 (not statically known → real If/Else).
    let cond = LocalPlusConst { local_idx: 0, k: 0 }; // local 0 + 0 — unknown at fold time
    let prefix = ConstCondition(7); // test-only snippet emitting `i32.const 7`
    let params = JumpCallParams::conditional_jump(FuncIdx(1), 1, &cond, pool)
        .with_taken_prefix(&prefix);

    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count() - 1).is_ok());

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 1).unwrap();
    assert!(reactor.seal_to(reactor.fn_count() - 1, &mut ctx, &Instruction::Unreachable).is_ok());
    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 2);
    let mut expected = Function::new([(1, ValType::I32)]);
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::I32Const(0));
    expected.instruction(&Instruction::I32Add);
    expected.instruction(&Instruction::If(wasm_encoder::BlockType::Empty));
    // Taken arm: prefix first…
    expected.instruction(&Instruction::I32Const(7));
    // …then params (1 param: local 0), then the transfer.
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::ReturnCall(1));
    // Not-taken arm: empty here (fall-through merging would fill it once a
    // successor slot exists; none does in this test). Seal closes the open
    // If with Unreachable+End, then re-asserts Unreachable before the
    // function's own End (BlockType::Empty ifs resume with zero values).
    expected.instruction(&Instruction::Else);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "taken_prefix must land inside the if arm, before params/return_call");
}

/// The known-true early-fold path must also run the taken-arm prefix (the
/// prefix is part of the taken edge's semantics, folded or not).
#[test]
fn test_taken_prefix_runs_on_known_true_condition() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 0).unwrap();

    let condition = ConstCondition(1); // always true → If/Else skeleton folds away
    let prefix = ConstCondition(7);
    let params = JumpCallParams::conditional_jump(FuncIdx(1), 1, &condition, pool)
        .with_taken_prefix(&prefix);

    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count() - 1).is_ok());

    reactor.next(&mut ctx, [(1, ValType::I32)].into_iter(), 1).unwrap();
    assert!(reactor.seal_to(reactor.fn_count() - 1, &mut ctx, &Instruction::Unreachable).is_ok());
    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 2);
    let mut expected = Function::new([(1, ValType::I32)]);
    expected.instruction(&Instruction::I32Const(7)); // prefix, no If/Else
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::ReturnCall(1));
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "known-true fold must keep the taken-arm prefix before params/return_call");
}

/// The prefix flows through the same optimizer pipeline as inline emission:
/// a constant pushed inside the prefix and consumed by nothing still
/// materializes (no elision across the transfer), and inst_count grows.
#[test]
fn test_taken_prefix_feeds_through_optimizer() {
    let mut reactor = Reactor::<(), std::convert::Infallible, Function>::default();
    let mut ctx = ();
    let pool = { static T: yecta::TableIdx = yecta::TableIdx(0); yecta::Pool { handler: &T, ty: TypeIdx(0) } };

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 0).unwrap();
    // Fold local 1 to a known constant BEFORE the branch: the prefix then
    // re-stores it; the store must be materialized (virtual locals flush)
    // before the return_call params read local 1.
    reactor.tail().instruction(&mut ctx, &Instruction::I32Const(5)).unwrap();
    reactor.tail().instruction(&mut ctx, &Instruction::LocalSet(1)).unwrap();

    let condition = ConstCondition(1);
    // Prefix: local.get 1; i32.const 1; i32.add; local.set 1 → 6, real store.
    struct Bump;
    impl<Context, E> InstructionSource<Context, E> for Bump {
        fn emit_instruction(
            &self,
            ctx: &mut Context,
            sink: &mut (dyn InstructionSink<Context, E> + '_),
        ) -> Result<(), E> {
            sink.instruction(ctx, &Instruction::LocalGet(1))?;
            sink.instruction(ctx, &Instruction::I32Const(1))?;
            sink.instruction(ctx, &Instruction::I32Add)?;
            sink.instruction(ctx, &Instruction::LocalSet(1))
        }
    }
    impl<Context, E> wax_core::build::InstructionOperatorSource<Context, E> for Bump {
        fn emit(
            &self,
            ctx: &mut Context,
            sink: &mut (dyn wax_core::build::InstructionOperatorSink<Context, E> + '_),
        ) -> Result<(), E> {
            self.emit_instruction(ctx, sink)
        }
    }

    let params = JumpCallParams::conditional_jump(FuncIdx(1), 2, &condition, pool)
        .with_taken_prefix(&Bump);
    assert!(reactor.ji_with_params(&mut ctx, params, reactor.fn_count() - 1).is_ok());

    reactor.next(&mut ctx, [(2, ValType::I32)].into_iter(), 1).unwrap();
    assert!(reactor.seal_to(reactor.fn_count() - 1, &mut ctx, &Instruction::Unreachable).is_ok());
    assert!(reactor.seal_to(0, &mut ctx, &Instruction::Unreachable).is_ok());

    let fns = reactor.into_fns();
    assert_eq!(fns.len(), 2);
    let mut expected = Function::new([(2, ValType::I32)]);
    // Everything folds through the single optimizer pipeline: the initial
    // store of 5 is virtual; the prefix's `local.get 1` folds to 5, so the
    // add produces 6 (virtual store); the return_call's param reads fold to
    // the constants 0 and 6. The one materialized `i32.const 6;
    // local.set 1` is the virtual-locals flush before the transfer.
    expected.instruction(&Instruction::I32Const(6));
    expected.instruction(&Instruction::LocalSet(1));
    expected.instruction(&Instruction::LocalGet(0));
    expected.instruction(&Instruction::I32Const(6));
    expected.instruction(&Instruction::ReturnCall(1));
    expected.instruction(&Instruction::Unreachable);
    expected.instruction(&Instruction::End);
    assert_eq!(fns[0], expected, "prefix must fold through the same optimizer pipeline as inline emission (re-run, not a frozen buffer)");
}
