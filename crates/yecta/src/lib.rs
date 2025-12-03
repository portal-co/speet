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

/// Parameters for the `ji` (jump/call instruction) operation.
/// 
/// This struct encapsulates all the parameters needed for emitting jumps or calls,
/// providing helper constructors for common use cases.
#[derive(Clone)]
pub struct JumpCallParams<'a> {
    /// Number of parameters to pass to the target function.
    pub params: u32,
    /// Map of parameter indices to snippets that compute modified values.
    pub fixups: BTreeMap<u32, &'a (dyn Snippet + 'a)>,
    /// The target function (static or dynamic).
    pub target: Target<'a>,
    /// If Some, emit a call with exception handling; if None, emit a jump.
    pub call: Option<EscapeTag>,
    /// Pool configuration for indirect calls.
    pub pool: Pool,
    /// If Some, make the jump/call conditional on this snippet.
    pub condition: Option<&'a (dyn Snippet + 'a)>,
}

impl<'a> JumpCallParams<'a> {
    /// Create parameters for an unconditional jump to a static function.
    /// 
    /// # Arguments
    /// * `func` - The target function index
    /// * `params` - Number of parameters to pass
    /// * `pool` - Pool configuration (can use default values if not doing indirect calls)
    pub fn jump(func: FuncIdx, params: u32, pool: Pool) -> Self {
        Self {
            params,
            fixups: BTreeMap::new(),
            target: Target::Static { func },
            call: None,
            pool,
            condition: None,
        }
    }

    /// Create parameters for an unconditional call to a static function with exception handling.
    /// 
    /// # Arguments
    /// * `func` - The target function index
    /// * `params` - Number of parameters to pass
    /// * `escape_tag` - Exception tag for non-local returns
    /// * `pool` - Pool configuration (can use default values if not doing indirect calls)
    pub fn call(func: FuncIdx, params: u32, escape_tag: EscapeTag, pool: Pool) -> Self {
        Self {
            params,
            fixups: BTreeMap::new(),
            target: Target::Static { func },
            call: Some(escape_tag),
            pool,
            condition: None,
        }
    }

    /// Create parameters for a conditional jump to a static function.
    /// 
    /// # Arguments
    /// * `func` - The target function index
    /// * `params` - Number of parameters to pass
    /// * `condition` - Snippet that computes the condition (non-zero = true)
    /// * `pool` - Pool configuration
    pub fn conditional_jump(
        func: FuncIdx,
        params: u32,
        condition: &'a (dyn Snippet + 'a),
        pool: Pool,
    ) -> Self {
        Self {
            params,
            fixups: BTreeMap::new(),
            target: Target::Static { func },
            call: None,
            pool,
            condition: Some(condition),
        }
    }

    /// Create parameters for an unconditional jump to a dynamic (indirect) function.
    /// 
    /// # Arguments
    /// * `idx` - Snippet that computes the function table index
    /// * `params` - Number of parameters to pass
    /// * `pool` - Pool configuration for the indirect call
    pub fn indirect_jump(idx: &'a (dyn Snippet + 'a), params: u32, pool: Pool) -> Self {
        Self {
            params,
            fixups: BTreeMap::new(),
            target: Target::Dynamic { idx },
            call: None,
            pool,
            condition: None,
        }
    }

    /// Add a parameter fixup that modifies a parameter value before the jump/call.
    /// 
    /// # Arguments
    /// * `param_idx` - Index of the parameter to modify
    /// * `fixup` - Snippet that computes the new value
    pub fn with_fixup(mut self, param_idx: u32, fixup: &'a (dyn Snippet + 'a)) -> Self {
        self.fixups.insert(param_idx, fixup);
        self
    }

    /// Set the condition for this jump/call.
    pub fn with_condition(mut self, condition: &'a (dyn Snippet + 'a)) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Convert this jump to a call with exception handling.
    pub fn with_call(mut self, escape_tag: EscapeTag) -> Self {
        self.call = Some(escape_tag);
        self
    }
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
        let FuncIdx(succ_idx) = succ;
        match self.fns.get_mut(succ_idx as usize) {
            Some(a) => {
                a.preds.insert(pred);
            }
            None => {
                let len = (self.fns.len() - succ_idx as usize) as u32;
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
            let FuncIdx(q_idx) = q;
            // self.fns[q_idx as usize].function.instruction(instruction);
            for p in self.fns[q_idx as usize].preds.iter().cloned() {
                queue.push_back(p);
            }
        }
        if cache.contains(&succ) {
            let FuncIdx(succ_idx) = succ;
            for k in cache {
                let FuncIdx(k_idx) = k;
                let f = &mut self.fns[k_idx as usize];
                _ = take(&mut f.preds);
                for p in 0..params {
                    f.function.instruction(&Instruction::LocalGet(p));
                }
                f.function.instruction(&Instruction::ReturnCall(succ_idx));
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
        let EscapeTag { tag: TagIdx(tag_idx), ty: TypeIdx(ty_idx) } = tag;
        self.feed(&Instruction::Block(wasm_encoder::BlockType::FunctionType(
            ty_idx,
        )));
        self.feed(&Instruction::TryTable(
            wasm_encoder::BlockType::FunctionType(ty_idx),
            [Catch::One { tag: tag_idx, label: 0 }].into_iter().collect(),
        ));
        match target {
            Target::Static { func: FuncIdx(func_idx) } => {
                self.feed(&Instruction::Call(func_idx));
            }
            Target::Dynamic { idx } => {
                idx.emit(&mut |a| self.feed(a));
                let Pool { ty: TypeIdx(pool_ty), table: TableIdx(pool_table) } = pool;
                self.feed(&Instruction::CallIndirect {
                    type_index: pool_ty,
                    table_index: pool_table,
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
        let EscapeTag { tag: TagIdx(tag_idx), ty: _ } = tag;
        for p in 0..params {
            self.feed(&Instruction::LocalGet(p));
        }
        self.feed(&Instruction::Throw(tag_idx));
    }
    /// Emit a jump or call instruction using a parameter struct.
    /// 
    /// This is the main control flow primitive that can emit:
    /// - Unconditional jumps
    /// - Conditional jumps
    /// - Calls (with exception handling)
    /// - With parameter fixups (modifications to parameters before the jump/call)
    /// 
    /// # Example
    /// ```no_run
    /// # use yecta::{Reactor, JumpCallParams, FuncIdx, Pool, TableIdx, TypeIdx};
    /// # use wasm_encoder::ValType;
    /// # let mut reactor = Reactor::default();
    /// # let pool = Pool { table: TableIdx(0), ty: TypeIdx(0) };
    /// // Simple unconditional jump
    /// let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    /// reactor.ji_with_params(params);
    /// ```
    pub fn ji_with_params(&mut self, params: JumpCallParams) {
        let JumpCallParams {
            params: param_count,
            fixups,
            target,
            call,
            pool,
            condition,
        } = params;
        
        self.ji(param_count, &fixups, target, call, pool, condition);
    }

    /// Emit a jump or call instruction, optionally conditional.
    /// 
    /// This is the main control flow primitive that can emit:
    /// - Unconditional jumps
    /// - Conditional jumps
    /// - Calls (with exception handling)
    /// - With parameter fixups (modifications to parameters before the jump/call)
    /// 
    /// For simpler cases, consider using [`JumpCallParams`] helper constructors
    /// and calling [`ji_with_params`](Self::ji_with_params) instead.
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
        // Track if statements for conditional branches
        if condition.is_some() {
            self.increment_if_stmts_for_predecessors();
        }
        
        match call {
            Some(escape_tag) => {
                self.emit_conditional_call(params, fixups, target, escape_tag, pool, condition);
            }
            None => {
                self.emit_conditional_jump(params, fixups, target, pool, condition);
            }
        }
    }

    /// Increment if statement counter for all predecessor functions.
    fn increment_if_stmts_for_predecessors(&mut self) {
        let reachable_funcs = self.collect_reachable_predecessors();
        for func_idx in reachable_funcs {
            let FuncIdx(idx) = func_idx;
            self.fns[idx as usize].if_stmts += 1;
        }
    }

    /// Collect all functions reachable by following predecessor edges.
    fn collect_reachable_predecessors(&self) -> BTreeSet<FuncIdx> {
        let mut queue = VecDeque::new();
        queue.push_back(FuncIdx((self.fns.len() - 1) as u32));
        let mut visited = BTreeSet::new();
        
        while let Some(current_func) = queue.pop_front() {
            if visited.contains(&current_func) {
                continue;
            }
            visited.insert(current_func);
            
            let FuncIdx(func_idx) = current_func;
            for predecessor in self.fns[func_idx as usize].preds.iter().cloned() {
                queue.push_back(predecessor);
            }
        }
        
        visited
    }

    /// Emit parameters with fixups applied.
    fn emit_params_with_fixups(&mut self, params: u32, fixups: &BTreeMap<u32, &(dyn Snippet + '_)>) {
        for param_idx in 0..params {
            if let Some(fixup) = fixups.get(&param_idx) {
                fixup.emit(&mut |instr| self.feed(instr));
            } else {
                self.feed(&Instruction::LocalGet(param_idx));
            }
        }
    }

    /// Restore parameters after a call, dropping fixed-up values and restoring original locals.
    fn restore_params_after_call(&mut self, params: u32, fixups: &BTreeMap<u32, &(dyn Snippet + '_)>) {
        for param_idx in (0..params).rev() {
            if fixups.contains_key(&param_idx) {
                self.feed(&Instruction::Drop);
            } else {
                self.feed(&Instruction::LocalSet(param_idx));
            }
        }
    }

    /// Emit a conditional call with exception handling.
    fn emit_conditional_call(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        target: Target,
        escape_tag: EscapeTag,
        pool: Pool,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit(&mut |instr| self.feed(instr));
            self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
        }
        
        self.emit_params_with_fixups(params, fixups);
        self.call(target, escape_tag, pool);
        self.restore_params_after_call(params, fixups);
        
        if condition.is_some() {
            self.feed(&Instruction::Else);
        }
    }

    /// Emit a conditional jump (no exception handling).
    fn emit_conditional_jump(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        target: Target,
        pool: Pool,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        match target {
            Target::Static { func } => {
                self.emit_static_jump(params, fixups, func, condition);
            }
            Target::Dynamic { idx } => {
                self.emit_dynamic_jump(params, fixups, idx, pool, condition);
            }
        }
    }

    /// Emit a static (direct) jump to a known function.
    fn emit_static_jump(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        func: FuncIdx,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit(&mut |instr| self.feed(instr));
            self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
            
            let FuncIdx(func_idx) = func;
            self.emit_params_with_fixups(params, fixups);
            self.feed(&Instruction::ReturnCall(func_idx));
            self.feed(&Instruction::Else);
        } else {
            // Unconditional jump: apply fixups to locals, then jump
            for (local_idx, fixup) in fixups.iter() {
                fixup.emit(&mut |instr| self.feed(instr));
                self.feed(&Instruction::LocalSet(*local_idx));
            }
            self.jmp(func, params);
        }
    }

    /// Emit a dynamic (indirect) jump through a table.
    fn emit_dynamic_jump(
        &mut self,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet + '_)>,
        idx: &(dyn Snippet + '_),
        pool: Pool,
        condition: Option<&(dyn Snippet + '_)>,
    ) {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit(&mut |instr| self.feed(instr));
            self.feed(&Instruction::If(wasm_encoder::BlockType::Empty));
        }
        
        self.emit_params_with_fixups(params, fixups);
        idx.emit(&mut |instr| self.feed(instr));
        
        let Pool { ty: TypeIdx(pool_ty), table: TableIdx(pool_table) } = pool;
        if condition.is_some() {
            self.feed(&Instruction::ReturnCallIndirect {
                type_index: pool_ty,
                table_index: pool_table,
            });
            self.feed(&Instruction::Else);
        } else {
            self.seal(&Instruction::ReturnCallIndirect {
                type_index: pool_ty,
                table_index: pool_table,
            });
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
            let FuncIdx(q_idx) = q;
            // self.fns[q_idx as usize].function.instruction(instruction);
            for p in self.fns[q_idx as usize].preds.iter().cloned() {
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
            let FuncIdx(q_idx) = q;
            self.fns[q_idx as usize].function.instruction(instruction);
            for p in self.fns[q_idx as usize].preds.iter().cloned() {
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
            let FuncIdx(q_idx) = q;
            self.fns[q_idx as usize].function.instruction(instruction);
            for _ in 0..ifs {
                self.fns[q_idx as usize].function.instruction(&Instruction::End);
            }
            for p in take(&mut self.fns[q_idx as usize].preds) {
                queue.push_back(p);
            }
        }
    }
    
    /// Get the total number of nested if statements for a function.
    fn total_ifs(&self, FuncIdx(p_idx): FuncIdx) -> usize {
        return self.fns[p_idx as usize].if_stmts;
    }
}
