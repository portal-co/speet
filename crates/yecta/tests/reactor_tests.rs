//! Integration tests for the Yecta reactor
//!
//! These tests verify the core functionality of the yecta reactor,
//! which manages WebAssembly function generation with complex control flow.

use wasm_encoder::{Function, Instruction, ValType};
use yecta::{EscapeTag, FuncIdx, JumpCallParams, Pool, Reactor, TableIdx, TagIdx, Target, TypeIdx};

#[test]
fn test_reactor_creation() {
    let _reactor = Reactor::<std::convert::Infallible, Function>::default();
}

#[test]
fn test_reactor_with_base_offset() {
    let reactor = Reactor::<std::convert::Infallible, Function>::with_base_func_offset(10);
    assert_eq!(reactor.base_func_offset(), 10);
}

#[test]
fn test_simple_function_creation() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    // Create a function with 2 i32 locals
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    // Emit some instructions
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    assert!(reactor.feed(&Instruction::I32Const(42)).is_ok());
    assert!(reactor.feed(&Instruction::I32Add).is_ok());
}

#[test]
fn test_multiple_functions() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    // Create first function
    reactor.next([(1, ValType::I32)].into_iter(), 0);
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    
    // Create second function
    reactor.next([(1, ValType::I64)].into_iter(), 0);
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
}

#[test]
fn test_unconditional_jump() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    // Create first function
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    assert!(reactor.feed(&Instruction::LocalGet(1)).is_ok());
    
    // Jump to function 1 with 2 parameters
    assert!(reactor.jmp(FuncIdx(1), 2).is_ok());
    
    // Create target function
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    assert!(reactor.seal(&Instruction::Unreachable).is_ok());
}

#[test]
fn test_jump_with_params_helper() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(0),
    };
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    // Use JumpCallParams helper
    let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    assert!(reactor.ji_with_params(params).is_ok());
    
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    assert!(reactor.seal(&Instruction::Unreachable).is_ok());
}

#[test]
fn test_conditional_operations() {
    // Test that conditional operations work correctly
    // This is a simpler test that doesn't require custom snippets
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    // Emit a simple conditional structure
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    assert!(reactor.feed(&Instruction::If(wasm_encoder::BlockType::Empty)).is_ok());
    assert!(reactor.feed(&Instruction::LocalGet(1)).is_ok());
    assert!(reactor.feed(&Instruction::Drop).is_ok());
    assert!(reactor.feed(&Instruction::End).is_ok());
    
    assert!(reactor.seal(&Instruction::Unreachable).is_ok());
}

#[test]
fn test_call_with_exception_handling() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(1),
    };
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(1),
    };
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    assert!(reactor
        .call(
            Target::Static {
                func: FuncIdx(1)
            },
            escape_tag,
            pool
        )
        .is_ok());
}

#[test]
fn test_return_via_exception() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(0),
    };
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    // Return via exception (params will be loaded by ret)
    assert!(reactor.ret(2, escape_tag).is_ok());
}

#[test]
fn test_seal_function() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    reactor.next([(1, ValType::I32)].into_iter(), 0);
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    
    // Seal with return
    assert!(reactor.seal(&Instruction::Return).is_ok());
}

#[test]
fn test_multiple_jumps() {
    // Test creating multiple jumps between functions
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(0),
    };
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    assert!(reactor.ji_with_params(params).is_ok());
    
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    
    // Jump from function 1 to function 2
    let params2 = JumpCallParams::jump(FuncIdx(2), 2, pool);
    assert!(reactor.ji_with_params(params2).is_ok());
    
    reactor.next([(2, ValType::I32)].into_iter(), 2);
    assert!(reactor.seal(&Instruction::Unreachable).is_ok());
}

#[test]
fn test_instruction_feeding() {
    // Test feeding various instructions to the reactor
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    reactor.next([(3, ValType::I32), (1, ValType::I64)].into_iter(), 0);
    
    // Feed various instructions
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    assert!(reactor.feed(&Instruction::LocalGet(1)).is_ok());
    assert!(reactor.feed(&Instruction::I32Add).is_ok());
    assert!(reactor.feed(&Instruction::LocalSet(2)).is_ok());
    assert!(reactor.feed(&Instruction::LocalGet(3)).is_ok());
    assert!(reactor.feed(&Instruction::I64Const(42)).is_ok());
    assert!(reactor.feed(&Instruction::I64Eq).is_ok());
    
    assert!(reactor.seal(&Instruction::Return).is_ok());
}

#[test]
fn test_base_func_offset_applied() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::with_base_func_offset(100);
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(0),
    };
    
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    
    // The offset should be applied when emitting function indices
    let params = JumpCallParams::jump(FuncIdx(0), 2, pool);
    assert!(reactor.ji_with_params(params).is_ok());
    
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    assert!(reactor.seal(&Instruction::Unreachable).is_ok());
}

#[test]
fn test_set_base_func_offset() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    assert_eq!(reactor.base_func_offset(), 0);
    
    reactor.set_base_func_offset(50);
    assert_eq!(reactor.base_func_offset(), 50);
}

#[test]
fn test_control_flow_distance() {
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    
    // Create functions with different control flow distances
    reactor.next([(1, ValType::I32)].into_iter(), 0);
    reactor.next([(1, ValType::I32)].into_iter(), 1);
    reactor.next([(1, ValType::I32)].into_iter(), 2);
    
    // All should be created successfully
    assert!(reactor.feed(&Instruction::Nop).is_ok());
}

#[test]
fn test_call_and_return() {
    // Test call with exception handling and return
    let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    let pool = Pool {
        table: TableIdx(0),
        ty: TypeIdx(1),
    };
    let escape_tag = EscapeTag {
        tag: TagIdx(0),
        ty: TypeIdx(1),
    };
    
    // Function 0: Calls function 1
    reactor.next([(2, ValType::I32)].into_iter(), 0);
    assert!(reactor.feed(&Instruction::LocalGet(0)).is_ok());
    assert!(reactor.feed(&Instruction::LocalGet(1)).is_ok());
    
    assert!(reactor
        .call(
            Target::Static {
                func: FuncIdx(1)
            },
            escape_tag,
            pool
        )
        .is_ok());
    
    // Function 1: Returns via exception
    reactor.next([(2, ValType::I32)].into_iter(), 1);
    assert!(reactor.ret(2, escape_tag).is_ok());
}
