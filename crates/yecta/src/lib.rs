//! # Yecta - WebAssembly Control Flow Reactor
//!
//! Yecta is a library for generating WebAssembly functions with complex control flow.
//! It provides a "reactor" system that manages function generation with:
//!
//! - Control flow graph management (predecessors, successors)
//! - Exception-based escapes for non-local control flow
//! - Conditional and unconditional jumps
//! - Direct and indirect function calls
//! - Nested if statements with automatic closure
//!
//! ## Core Concepts
//!
//! ### Reactor
//! The [`Reactor`] is the main interface for generating WebAssembly functions.
//! It maintains a collection of functions being generated and manages control flow edges between them.
//!
//! ### Control Flow
//! Functions can jump to each other, creating a control flow graph. The reactor tracks
//! predecessor relationships and can detect cycles, converting them to tail calls when necessary.
//!
//! ### Exception-Based Escapes
//! The library uses WebAssembly exceptions (via `EscapeTag`) to implement non-local returns
//! and other control flow patterns.
//!
//! ### Code Snippets
//! The [`Snippet`] trait allows for dynamic code generation, where small pieces of code
//! can be composed and emitted as part of larger control flow patterns.
//!
//! ## Example Usage
//!
//! ```no_run
//! use yecta::{Reactor, Target, EscapeTag, Pool, FuncIdx, TagIdx, TypeIdx, TableIdx};
//! use wasm_encoder::ValType;
//!
//! let mut reactor = Reactor::default();
//! 
//! // Create a new function with 2 i32 locals
//! reactor.next([(2, ValType::I32)].into_iter(), 0);
//!
//! // Add instructions, jumps, calls, etc.
//! ```

#![no_std]

use core::mem::take;

use alloc::{
    collections::{btree_map::BTreeMap, btree_set::BTreeSet, vec_deque::VecDeque},
    vec::Vec,
};
use wasm_encoder::{Catch, Function, Instruction, ValType};

extern crate alloc;

/// Index of a WebAssembly function in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct FuncIdx(pub u32);

/// Index of a WebAssembly tag (exception tag) in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TagIdx(pub u32);

/// Index of a WebAssembly table in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TableIdx(pub u32);

/// Index of a WebAssembly type (function signature) in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TypeIdx(pub u32);

/// Index of a local variable within a function.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct LocalIdx(pub u32);

/// Escape tag configuration for exception handling in WebAssembly.
/// Contains a tag index and a type index that defines the exception signature.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct EscapeTag {
    pub tag: TagIdx,
    pub ty: TypeIdx,
}

/// Pool configuration for indirect function calls.
/// Contains a table index and a type index for call_indirect operations.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Pool {
    pub table: TableIdx,
    pub ty: TypeIdx,
}
/// Target for a jump or call operation.
/// Can be either a static function reference or a dynamic indirect call.
#[derive(Clone, Copy)]
pub enum Target<'a> {
    /// Static call to a known function index.
    Static { func: FuncIdx },
    /// Dynamic call through a table, with the index computed by a snippet.
    Dynamic { idx: &'a (dyn Snippet + 'a) },
}
/// Trait for code snippets that can emit WebAssembly instructions.
/// Used for dynamic code generation within the reactor system.
pub trait Snippet {
    /// Emit WebAssembly instructions by calling the provided function for each instruction.
    fn emit(&self, go: &mut (dyn FnMut(&Instruction<'_>) + '_));
}
/// A reactor manages the generation of WebAssembly functions with control flow.
/// It handles function generation, control flow edges (predecessors), and nested if statements.
#[derive(Default)]
pub struct Reactor {
    fns: Vec<Entry>,
    lens: VecDeque<BTreeSet<FuncIdx>>,
}

/// Internal entry representing a function being generated.
struct Entry {
    function: Function,
    preds: BTreeSet<FuncIdx>,
    if_stmts: usize,
}
impl Reactor {
    /// Create a new function with the given locals and control flow distance.
    /// 
    /// # Arguments
    /// * `locals` - Iterator of (count, type) pairs defining local variables
    /// * `len` - Control flow distance/depth for this function
    pub fn next(
        &mut self,
        locals: impl IntoIterator<Item = (u32, ValType), IntoIter: ExactSizeIterator>,
        len: u32,
    ) {
        while self.lens.len() != len as usize + 1 {
            self.lens.push_back(Default::default());
        }
        self.lens
            .iter_mut()
            .nth(len as usize)
            .unwrap()
            .insert(FuncIdx(self.fns.len() as u32));
        self.fns.push(Entry {
            function: Function::new(locals),
            preds: self.lens.pop_front().into_iter().flatten().collect(),
            if_stmts: 0,
        });
    }
    /// Add a predecessor edge from pred to succ in the control flow graph.
    fn add_pred(&mut self, succ: FuncIdx, pred: FuncIdx) {
        match self.fns.get_mut(succ.0 as usize) {
            Some(a) => {
                a.preds.insert(pred);
            }
            None => {
                let len = (self.fns.len() - succ.0 as usize) as u32;
                while self.lens.len() != len as usize + 1 {
                    self.lens.push_back(Default::default());
                }
                self.lens.iter_mut().nth(len as usize).unwrap().insert(pred);
            }
        }
    }
    /// Add a predecessor edge with cycle detection.
    /// If a cycle is detected, converts to return calls instead.
    fn add_pred_checked(&mut self, succ: FuncIdx, pred: FuncIdx, params: u32) {
        let ifs = self.total_ifs(pred);
        let mut queue = VecDeque::new();
        queue.push_back(pred);
        let mut cache = BTreeSet::new();

        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            // self.fns[q.0 as usize].function.instruction(instruction);
            for p in self.fns[q.0 as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
        if cache.contains(&succ) {
            for k in cache {
                let f = &mut self.fns[k.0 as usize];
                _ = take(&mut f.preds);
                for p in 0..params {
                    f.function.instruction(&Instruction::LocalGet(p));
                }
                f.function.instruction(&Instruction::ReturnCall(succ.0));
                for _ in 0..ifs {
                    f.function.instruction(&Instruction::End);
                }
            }
        } else {
            self.add_pred(succ, pred);
        }
    }
    /// Emit a call instruction with exception handling.
    /// The call is wrapped in a try-catch block to handle escapes via the specified tag.
    /// 
    /// # Arguments
    /// * `target` - The function to call (static or dynamic)
    /// * `tag` - Exception tag configuration for escape handling
    /// * `pool` - Pool configuration for indirect calls
    pub fn call(&mut self, target: Target, tag: EscapeTag, pool: Pool) {
        let EscapeTag { tag, ty } = tag;
        self.feed(&Instruction::Block(wasm_encoder::BlockType::FunctionType(
            ty.0,
        )));
        self.feed(&Instruction::TryTable(
            wasm_encoder::BlockType::FunctionType(ty.0),
            [Catch::One { tag: tag.0, label: 0 }].into_iter().collect(),
        ));
        match target {
            Target::Static { func } => {
                self.feed(&Instruction::Call(func.0));
            }
            Target::Dynamic { idx } => {
                idx.emit(&mut |a| self.feed(a));
                self.feed(&Instruction::CallIndirect {
                    type_index: pool.ty.0,
                    table_index: pool.table.0,
                });
            }
        }
        self.feed(&Instruction::Return);
        self.feed(&Instruction::End);
        self.feed(&Instruction::End);
    }
    /// Emit a return via exception throw.
    /// Loads the specified number of parameter locals and throws them with the tag.
    /// 
    /// # Arguments
    /// * `params` - Number of parameters to pass through the exception
    /// * `tag` - Exception tag to throw
    pub fn ret(&mut self, params: u32, tag: EscapeTag) {
        let EscapeTag { tag, ty: _ } = tag;
        for p in 0..params {
            self.feed(&Instruction::LocalGet(p));
        }
        self.feed(&Instruction::Throw(tag.0));
    }
    /// Emit a jump or call instruction, optionally conditional.
    /// 
    /// This is the main control flow primitive that can emit:
    /// - Unconditional jumps
    /// - Conditional jumps
    /// - Calls (with exception handling)
    /// - With parameter fixups (modifications to parameters before the jump/call)
    /// 
    /// # Arguments
    /// * `params` - Number of parameters to pass
    /// * `fixups` - Map of parameter indices to snippets that compute new values
    /// * `target` - The target function (static or dynamic)
    /// * `call` - If Some, emit a call with exception handling; if None, emit a jump
    /// * `pool` - Pool configuration for indirect calls
    /// * `condition` - If Some, make the jump/call conditional on this snippet
    pub fn ji(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        target: Target,
        call: Option<EscapeTag>,
        pool: Pool,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        if let Some(_c) = condition.as_deref() {
            let mut queue = VecDeque::new();
            queue.push_back(FuncIdx((self.fns.len() - 1) as u32));
            let mut cache = BTreeSet::new();
            while let Some(q) = queue.pop_front() {
                if cache.contains(&q) {
                    continue;
                };
                cache.insert(q);
                // self.fns[q.0 as usize].function.instruction(instruction);
                for p in self.fns[q.0 as usize].preds.iter().cloned() {
                    queue.push_back(p);
                }
            }
            for c in cache {
                self.fns[c.0 as usize].if_stmts += 1;
            }
        }
        match call {
            Some(tag) => {
                if let Some(s) = condition.as_deref() {
                    s.emit(&mut |a| self.feed(a));
                    self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                }
                for p in 0..params {
                    if let Some(f) = fixups.get(&p) {
                        f.emit(&mut |a| self.feed(a));
                    } else {
                        self.feed(&Instruction::LocalGet(p));
                    }
                }
                self.call(target, tag, pool);
                for p in (0..params).rev() {
                    if let Some(_f) = fixups.get(&p) {
                        self.feed(&Instruction::Drop);
                    } else {
                        self.feed(&Instruction::LocalSet(p));
                    }
                }
                if let Some(_s) = condition.as_deref() {
                    self.feed(&Instruction::Else);
                }
            }
            None => match target {
                Target::Static { func } => {
                    if let Some(_s) = condition.as_deref() {
                        _s.emit(&mut |a| self.feed(a));
                        self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                    }

                    if let Some(_s) = condition.as_deref() {
                        for p in 0..params {
                            if let Some(f) = fixups.get(&p) {
                                f.emit(&mut |a| self.feed(a));
                            } else {
                                self.feed(&Instruction::LocalGet(p));
                            }
                        }
                        self.feed(&Instruction::ReturnCall(func.0));
                        self.feed(&Instruction::Else);
                    } else {
                        for (i, f) in fixups.iter() {
                            f.emit(&mut |a| self.feed(a));
                            self.feed(&Instruction::LocalSet(*i));
                        }
                        self.jmp(func, params)
                    }
                }
                Target::Dynamic { idx } => {
                    if let Some(_s) = condition.as_deref() {
                        _s.emit(&mut |a| self.feed(a));
                        self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
                    }
                    for p in 0..params {
                        if let Some(f) = fixups.get(&p) {
                            f.emit(&mut |a| self.feed(a));
                        } else {
                            self.feed(&Instruction::LocalGet(p));
                        }
                    }
                    idx.emit(&mut |a| self.feed(a));
                    if let Some(_s) = condition.as_deref() {
                        self.feed(&Instruction::ReturnCallIndirect {
                            type_index: pool.ty.0,
                            table_index: pool.table.0,
                        });
                        self.feed(&Instruction::Else);
                    } else {
                        self.seal(&Instruction::ReturnCallIndirect {
                            type_index: pool.ty.0,
                            table_index: pool.table.0,
                        });
                    }
                }
            },
        }
    }
    /// Emit an unconditional jump to the target function.
    /// This creates control flow edges from all active functions to the target.
    /// 
    /// # Arguments
    /// * `target` - The function index to jump to
    /// * `params` - Number of parameters to pass
    pub fn jmp(&mut self, target: FuncIdx, params: u32) {
        let mut queue = VecDeque::new();
        queue.push_back(FuncIdx((self.fns.len() - 1) as u32));
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            // self.fns[q.0 as usize].function.instruction(instruction);
            for p in self.fns[q.0 as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
        for x in cache {
            self.add_pred_checked(target, x, params);
        }
    }
    /// Feed an instruction to all active functions.
    /// The instruction is added to the current function and all its predecessors.
    pub fn feed(&mut self, instruction: &Instruction<'_>) {
        let mut queue = VecDeque::new();
        queue.push_back(FuncIdx((self.fns.len() - 1) as u32));
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            self.fns[q.0 as usize].function.instruction(instruction);
            for p in self.fns[q.0 as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
    }
    /// Seal the current function by emitting a final instruction and closing all if statements.
    /// This terminates the function and removes all predecessor edges.
    pub fn seal(&mut self, instruction: &Instruction<'_>) {
        let ifs = self.total_ifs(FuncIdx((self.fns.len() - 1) as u32));
        let mut queue = VecDeque::new();
        queue.push_back(FuncIdx((self.fns.len() - 1) as u32));
        let mut cache = BTreeSet::new();
        while let Some(q) = queue.pop_front() {
            if cache.contains(&q) {
                continue;
            };
            cache.insert(q);
            self.fns[q.0 as usize].function.instruction(instruction);
            for _ in 0..ifs {
                self.fns[q.0 as usize].function.instruction(&Instruction::End);
            }
            for p in take(&mut self.fns[q.0 as usize].preds) {
                queue.push_back(p);
            }
        }
    }
    
    /// Get the total number of nested if statements for a function.
    fn total_ifs(&self, p: FuncIdx) -> usize {
        return self.fns[p.0 as usize].if_stmts;
    }
}
