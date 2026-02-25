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
//! ```ignore
//! use yecta::{Reactor, Target, EscapeTag, Pool, FuncIdx, TagIdx, TypeIdx, TableIdx};
//! use wasm_encoder::{ValType, Function};
//!
//! let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
//!
//! // Create a new function with 2 i32 locals
//! reactor.next([(2, ValType::I32)].into_iter(), 0);
//!
//! // Add instructions, jumps, calls, etc.
//! ```

//! ## Static Speculative Call Lowering
//!
//! In contexts where guest binaries use standard ABI call instructions (for
//! example x86_64 `call` or RISC-V `jal`/`jalr`), the recompiler may lower
//! those ABI-compliant calls to native WebAssembly `call` instructions. The
//! implementation in this crate emits calls wrapped in a small validation
//! scaffold: the call is placed inside a `Block`/`TryTable` region tied to an
//! `EscapeTag`. When the callee returns, the generated code validates the
//! guest program counter (or other return metadata). If the return location
//! differs from the statically-expected value the code throws the configured
//! `EscapeTag` carrying the unexpected return payload. The outer `Reactor`
//! catch handler receives the exception and performs a safe, validated
//! transfer (for example via `return_call` or a dispatcher jump) to resume
//! execution at the correct guest location.
//!
//! This crate's `Reactor::call` and `ret` implementations reflect that
//! strategy: `call` emits a `Block(FunctionType(ty_idx))` and a `TryTable`
//! region; `ret` pushes the configured parameters onto the exception payload
//! and emits a `Throw(tag_idx)`. The exception-based escape ensures any
//! deviation from the expected control flow is intercepted and re-validated
//! by the vkernel-backed dispatcher.

#![no_std]

use core::{
    convert::Infallible,
    marker::PhantomData,
    mem::{take, transmute},
};

use alloc::{
    collections::{btree_map::BTreeMap, btree_set::BTreeSet, vec_deque::VecDeque},
    vec::Vec,
};
use wasm_encoder::{Catch, Function, Instruction, ValType};
use wax_core::build::InstructionSink;

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
pub struct JumpCallParams<'a, Context, E> {
    /// Number of parameters to pass to the target function.
    pub params: u32,
    /// Map of parameter indices to snippets that compute modified values.
    pub fixups: BTreeMap<u32, &'a (dyn Snippet<Context, E> + 'a)>,
    /// The target function (static or dynamic).
    pub target: Target<'a, Context, E>,
    /// If Some, emit a call with exception handling; if None, emit a jump.
    pub call: Option<EscapeTag>,
    /// Pool configuration for indirect calls.
    pub pool: Pool,
    /// If Some, make the jump/call conditional on this snippet.
    pub condition: Option<&'a (dyn Snippet<Context, E> + 'a)>,
}

impl<'a, Context, E> JumpCallParams<'a, Context, E> {
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
        condition: &'a (dyn Snippet<Context, E> + 'a),
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
    pub fn indirect_jump(idx: &'a (dyn Snippet<Context, E> + 'a), params: u32, pool: Pool) -> Self {
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
    pub fn with_fixup(mut self, param_idx: u32, fixup: &'a (dyn Snippet<Context, E> + 'a)) -> Self {
        self.fixups.insert(param_idx, fixup);
        self
    }

    /// Set the condition for this jump/call.
    ///
    /// The condition snippet should compute an i32 value. If the value is zero,
    /// the jump/call is skipped; otherwise it executes.
    ///
    /// # Arguments
    /// * `condition` - Snippet that evaluates the condition (non-zero = true)
    pub fn with_condition(mut self, condition: &'a (dyn Snippet<Context, E> + 'a)) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Convert this jump to a call with exception handling.
    ///
    /// When set, the target is called with a try-catch wrapper that handles
    /// non-local returns via the specified exception tag.
    ///
    /// # Arguments
    /// * `escape_tag` - The exception tag used for non-local returns
    pub fn with_call(mut self, escape_tag: EscapeTag) -> Self {
        self.call = Some(escape_tag);
        self
    }
}
/// Target for a jump or call operation.
/// Can be either a static function reference or a dynamic indirect call.
#[derive(Clone, Copy)]
pub enum Target<'a, Context, E> {
    /// Static call to a known function index.
    Static { func: FuncIdx },
    /// Dynamic call through a table, with the index computed by a snippet.
    Dynamic {
        idx: &'a (dyn Snippet<Context, E> + 'a),
    },
}
/// Trait for code snippets that can emit WebAssembly instructions.
/// Used for dynamic code generation within the reactor system.
pub trait Snippet<Context, E = Infallible>: wax_core::build::InstructionSource<Context, E> {
    /// Emit WebAssembly instructions by calling the provided function for each instruction.
    fn emit_snippet(
        &self,
        ctx: &mut Context,
        go: &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + '_),
    ) -> Result<(), E>;
}
impl<Context, E, T: wax_core::build::InstructionSource<Context, E> + ?Sized> Snippet<Context, E>
    for T
{
    fn emit_snippet(
        &self,
        ctx: &mut Context,
        go: &mut (dyn FnMut(&mut Context, &Instruction<'_>) -> Result<(), E> + '_),
    ) -> Result<(), E> {
        self.emit_instruction(
            ctx,
            &mut wax_core::build::FromFn::instruction_sink(|ctx, a| go(ctx, a)),
        )
    }
}
/// Default maximum number of WebAssembly instructions emitted into a single
/// generated function before [`Reactor::next_with`] forces a split.
pub const DEFAULT_MAX_INSTS_PER_FN: usize = 4096;

/// Default maximum number of open `if` blocks that may accumulate in a single
/// generated function before a conditional branch forces a split.
pub const DEFAULT_MAX_IFS_PER_FN: usize = 32;

/// A reactor manages the generation of WebAssembly functions with control flow.
/// It handles function generation, control flow edges (predecessors), and nested if statements.
pub struct Reactor<Context, E = Infallible, F: InstructionSink<Context, E> = Function> {
    fns: Vec<Entry<F>>,
    lens: VecDeque<BTreeSet<FuncIdx>>,
    phantom: PhantomData<(Context, E)>,
    /// Base offset added to all emitted function indices.
    /// Used when imports or helper functions precede the generated functions in the module.
    base_func_offset: u32,
    /// Maximum instructions per generated function before a forced split.
    max_insts_per_fn: usize,
    /// Maximum open `if` blocks per generated function before a forced split.
    max_ifs_per_fn: usize,
}
impl<Context, E, F: InstructionSink<Context, E>> Default for Reactor<Context, E, F> {
    fn default() -> Self {
        Self {
            fns: Default::default(),
            lens: Default::default(),
            phantom: Default::default(),
            base_func_offset: 0,
            max_insts_per_fn: DEFAULT_MAX_INSTS_PER_FN,
            max_ifs_per_fn: DEFAULT_MAX_IFS_PER_FN,
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>> Reactor<Context, E, F> {
    /// Create a new reactor with a base function offset.
    ///
    /// The offset is added to all emitted function indices. This is useful when
    /// the WebAssembly module has imported functions or helper functions that
    /// precede the generated functions.
    ///
    /// # Arguments
    /// * `base_func_offset` - Offset added to function indices in emitted instructions
    ///
    /// # Example
    /// If the module has 10 imports and 5 helper functions, use `base_func_offset = 15`
    /// so that generated function 0 emits as WebAssembly function 15.
    pub fn with_base_func_offset(base_func_offset: u32) -> Self {
        Self {
            fns: Default::default(),
            lens: Default::default(),
            phantom: Default::default(),
            base_func_offset,
            max_insts_per_fn: DEFAULT_MAX_INSTS_PER_FN,
            max_ifs_per_fn: DEFAULT_MAX_IFS_PER_FN,
        }
    }

    /// Get the current base function offset.
    pub fn base_func_offset(&self) -> u32 {
        self.base_func_offset
    }

    /// Set the base function offset.
    ///
    /// The offset is added to all emitted function indices. This is useful when
    /// the WebAssembly module has imported functions or helper functions that
    /// precede the generated functions.
    pub fn set_base_func_offset(&mut self, offset: u32) {
        self.base_func_offset = offset;
    }

    /// Return the maximum number of instructions per generated function.
    pub fn max_insts_per_fn(&self) -> usize {
        self.max_insts_per_fn
    }

    /// Set the maximum number of instructions per generated function.
    ///
    /// When the tail function's instruction count reaches this limit,
    /// the next call to [`next_with`](Self::next_with) will seal the current
    /// function group before starting a new one.
    pub fn set_max_insts_per_fn(&mut self, limit: usize) {
        self.max_insts_per_fn = limit;
    }

    /// Return the maximum number of open `if` blocks per generated function.
    pub fn max_ifs_per_fn(&self) -> usize {
        self.max_ifs_per_fn
    }

    /// Set the maximum number of open `if` blocks per generated function.
    ///
    /// When adding a conditional branch would push the tail function's open
    /// `if` count to this limit, the current function group is sealed first
    /// and the conditional branch is emitted into a fresh function.
    pub fn set_max_ifs_per_fn(&mut self, limit: usize) {
        self.max_ifs_per_fn = limit;
    }
}

/// Internal entry representing a function being generated.
struct Entry<F> {
    function: F,
    bundles: Vec<Instruction<'static>>,
    preds: BTreeSet<FuncIdx>,
    if_stmts: usize,
    /// Running count of instructions emitted into this function via `feed`.
    inst_count: usize,
    /// Lazily-computed cache of all functions reachable from this entry by
    /// following predecessor edges transitively, including this entry itself.
    ///
    /// `None` means the cache is stale and must be recomputed by BFS before
    /// use.  Invalidated (set to `None`) whenever any entry's `preds` set
    /// changes.  Only recomputed when actually needed (in `feed`, `seal`,
    /// saturation checks, etc.).
    transitive_preds: Option<BTreeSet<FuncIdx>>,
}
impl<Context, E> Reactor<Context, E> {
    /// Create a new function with the given locals and control flow distance.
    ///
    /// # Arguments
    /// * `ctx` - Mutable context passed through to the instruction sink on split
    /// * `locals` - Iterator of (count, type) pairs defining local variables
    /// * `len` - Control flow distance/depth for this function
    pub fn next(
        &mut self,
        ctx: &mut Context,
        locals: impl IntoIterator<Item = (u32, ValType), IntoIter: ExactSizeIterator>,
        len: u32,
    ) -> Result<(), E> {
        self.next_with(ctx, Function::new(locals), len)
    }
}
impl<Context, E, F: InstructionSink<Context, E>> InstructionSink<Context, E>
    for Reactor<Context, E, F>
{
    fn instruction(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.feed(ctx, instruction)
    }
}
impl<Context, E, F: InstructionSink<Context, E>> Reactor<Context, E, F> {
    /// Create a new function with the given locals and control flow distance.
    ///
    /// If any function in the current reachable predecessor set has reached the
    /// instruction limit or the `if`-nesting limit, the current function group
    /// is sealed first: every open `if` block is closed, and all predecessor
    /// edges are severed so the new function starts with a clean slate.
    ///
    /// # Arguments
    /// * `ctx` - Mutable context passed through to the instruction sink on split
    /// * `f` - The function (instruction sink) to add
    /// * `len` - Control flow distance/depth for this function
    pub fn next_with(&mut self, ctx: &mut Context, f: F, len: u32) -> Result<(), E> {
        // Check every function that would receive instructions from the new
        // tail (i.e. the current tail's transitive predecessor set).  If any
        // has hit a limit, seal the whole group before opening a new function.
        if !self.fns.is_empty() {
            let tail_idx = self.fns.len() - 1;
            let reachable = self.transitive_preds_of(tail_idx).clone();
            let max_insts = self.max_insts_per_fn;
            let max_ifs = self.max_ifs_per_fn;
            let saturated = reachable.iter().any(|&FuncIdx(i)| {
                self.fns[i as usize].inst_count >= max_insts
                    || self.fns[i as usize].if_stmts >= max_ifs
            });
            if saturated {
                self.seal_for_split(ctx)?;
            }
        }

        while self.lens.len() < len as usize + 1 {
            self.lens.push_back(Default::default());
        }
        self.lens
            .iter_mut()
            .nth(len as usize)
            .unwrap()
            .insert(FuncIdx(self.fns.len() as u32));

        // Build the direct predecessor set for the new entry from the lens queue.
        // Filter out self-references (the new entry's own index in lens[0] for len=0).
        let new_idx = FuncIdx(self.fns.len() as u32);
        let direct_preds: BTreeSet<FuncIdx> = self
            .lens
            .pop_front()
            .into_iter()
            .flatten()
            .filter(|&p| p != new_idx)
            .collect();

        self.fns.push(Entry {
            function: f,
            preds: direct_preds,
            if_stmts: 0,
            inst_count: 0,
            bundles: Vec::new(),
            transitive_preds: None, // computed lazily on first use
        });
        Ok(())
    }

    /// Seal all functions in the current reachable group due to a saturation
    /// split triggered from [`next_with`] or [`increment_if_stmts_for_predecessors`].
    ///
    /// Emits `Unreachable` followed by enough `End` instructions to close all
    /// open `if` blocks on every reachable function, then drains all
    /// predecessor edges and clears every entry's `transitive_preds` to just
    /// itself so the next function starts with a clean slate.
    /// Seal all functions in the current reachable group due to a saturation
    /// split triggered from [`next_with`] or [`increment_if_stmts_for_predecessors`].
    ///
    /// Emits `Unreachable` followed by enough `End` instructions to close all
    /// open `if` blocks on every reachable function, then drains all
    /// predecessor edges and invalidates every affected entry's transitive
    /// cache so the next function starts with a clean slate.
    fn seal_for_split(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.fns.is_empty() {
            return Ok(());
        }
        let tail_idx = self.fns.len() - 1;
        let ifs = self.fns[tail_idx].if_stmts;
        let reachable: BTreeSet<FuncIdx> = self.transitive_preds_of(tail_idx).clone();
        for &FuncIdx(idx) in &reachable {
            self.fns[idx as usize]
                .function
                .instruction(ctx, &Instruction::Unreachable)?;
            for _ in 0..ifs {
                self.fns[idx as usize]
                    .function
                    .instruction(ctx, &Instruction::End)?;
            }
            _ = take(&mut self.fns[idx as usize].preds);
            // Invalidate transitive cache after severing preds.
            self.fns[idx as usize].transitive_preds = None;
        }
        // Clear any pending future-function predecessor assignments in the
        // lens queue — they reference entries from before the split and are
        // no longer valid.
        self.lens.clear();
        Ok(())
    }

    /// Return the transitive predecessor set for the entry at `idx`,
    /// computing and caching it via BFS if the cache is stale.
    ///
    /// The returned reference borrows `self` immutably after the cache is
    /// populated, so callers that need to mutate `self` afterward must clone
    /// the result first.
    fn transitive_preds_of(&mut self, idx: usize) -> &BTreeSet<FuncIdx> {
        // If already cached, return immediately.
        if self.fns[idx].transitive_preds.is_some() {
            return self.fns[idx].transitive_preds.as_ref().unwrap();
        }

        // BFS over direct preds to build the transitive set.
        let mut visited: BTreeSet<FuncIdx> = BTreeSet::new();
        visited.insert(FuncIdx(idx as u32));

        let mut stack: Vec<FuncIdx> = self.fns[idx].preds.iter().cloned().collect();
        while let Some(p) = stack.pop() {
            if visited.contains(&p) {
                continue;
            }
            visited.insert(p);
            let FuncIdx(pi) = p;
            for &q in &self.fns[pi as usize].preds {
                if !visited.contains(&q) {
                    stack.push(q);
                }
            }
        }

        self.fns[idx].transitive_preds = Some(visited);
        self.fns[idx].transitive_preds.as_ref().unwrap()
    }
    /// Add a predecessor edge from pred to succ in the control flow graph.
    ///
    /// Invalidates the transitive predecessor cache for `succ` and for every
    /// live entry that has `succ` in its cached transitive set (since those
    /// entries can now reach `pred` transitively too).
    fn add_pred(&mut self, succ: FuncIdx, pred: FuncIdx) {
        let FuncIdx(succ_idx) = succ;
        match self.fns.get_mut(succ_idx as usize) {
            Some(a) => {
                a.preds.insert(pred);
                // Invalidate this entry's cache — it has a new predecessor.
                a.transitive_preds = None;
            }
            None => {
                // succ_idx is beyond the current fns vector — store in lens
                // for when that entry is created by next_with.
                let len = (succ_idx as usize)
                    .checked_sub(self.fns.len())
                    .expect("add_pred: succ_idx should be >= fns.len() in None branch")
                    as u32;
                while self.lens.len() < len as usize + 1 {
                    self.lens.push_back(Default::default());
                }
                self.lens.iter_mut().nth(len as usize).unwrap().insert(pred);
                // No live entry to invalidate; done.
                return;
            }
        }
        // Invalidate the cache of every live entry whose cached transitive set
        // included succ — those sets are now stale because they're missing the
        // new predecessors reachable through succ.
        for i in 0..self.fns.len() {
            if let Some(ref tp) = self.fns[i].transitive_preds {
                if tp.contains(&succ) {
                    self.fns[i].transitive_preds = None;
                }
            }
        }
    }
    /// Add a predecessor edge with cycle detection.
    /// If a cycle is detected, converts to return calls instead.
    fn add_pred_checked(
        &mut self,
        ctx: &mut Context,
        succ: FuncIdx,
        pred: FuncIdx,
        params: u32,
    ) -> Result<(), E> {
        let ifs = self.total_ifs(pred);
        let FuncIdx(pred_idx) = pred;
        // Use the per-entry transitive predecessor cache for the cycle check.
        let cycle = self.transitive_preds_of(pred_idx as usize).contains(&succ);
        if cycle {
            // Clone the set so we can mutate fns while iterating.
            let pred_transitive: BTreeSet<FuncIdx> = self.fns[pred_idx as usize]
                .transitive_preds
                .clone()
                .unwrap();
            let FuncIdx(succ_idx) = succ;
            let wasm_func_idx = succ_idx + self.base_func_offset;
            for k in &pred_transitive {
                let FuncIdx(k_idx) = *k;
                let f = &mut self.fns[k_idx as usize];
                _ = take(&mut f.preds);
                f.transitive_preds = None; // invalidate after severing preds
                for b in f.bundles.drain(..) {
                    f.function.instruction(ctx, &b)?;
                }
                for p in 0..params {
                    f.function.instruction(ctx, &Instruction::LocalGet(p))?;
                }
                f.function
                    .instruction(ctx, &Instruction::ReturnCall(wasm_func_idx))?;
                for _ in 0..ifs {
                    f.function.instruction(ctx, &Instruction::End)?;
                }
            }
        } else {
            self.add_pred(succ, pred);
        }
        Ok(())
    }

    fn flush_bundles(&mut self, ctx: &mut Context, idx: usize) -> Result<(), E> {
        let mut funcs = self.transitive_preds_of(idx).clone();
        for func in funcs {
            let f = &mut self.fns[func.0 as usize];
            for b in f.bundles.drain(..) {
                f.function.instruction(ctx, &b)?;
            }
        }
        Ok(())
    }

    /// Emit a call instruction with exception handling.
    /// The call is wrapped in a try-catch block to handle escapes via the specified tag.
    ///
    /// # Arguments
    /// * `target` - The function to call (static or dynamic)
    /// * `tag` - Exception tag configuration for escape handling
    /// * `pool` - Pool configuration for indirect calls
    pub fn call(
        &mut self,
        ctx: &mut Context,
        target: Target<Context, E>,
        tag: EscapeTag,
        pool: Pool,
    ) -> Result<(), E> {
        self.flush_bundles(ctx, self.fns.len() - 1)?;
        let EscapeTag {
            tag: TagIdx(tag_idx),
            ty: TypeIdx(ty_idx),
        } = tag;
        self.feed(
            ctx,
            &Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)),
        )?;
        self.feed(
            ctx,
            &Instruction::TryTable(
                wasm_encoder::BlockType::FunctionType(ty_idx),
                [Catch::One {
                    tag: tag_idx,
                    label: 0,
                }]
                .into_iter()
                .collect(),
            ),
        )?;
        match target {
            Target::Static {
                func: FuncIdx(func_idx),
            } => {
                let wasm_func_idx = func_idx + self.base_func_offset;
                self.feed(ctx, &Instruction::Call(wasm_func_idx))?;
            }
            Target::Dynamic { idx } => {
                idx.emit_snippet(ctx, &mut |ctx, a| self.feed(ctx, a))?;
                let Pool {
                    ty: TypeIdx(pool_ty),
                    table: TableIdx(pool_table),
                } = pool;
                self.feed(
                    ctx,
                    &Instruction::CallIndirect {
                        type_index: pool_ty,
                        table_index: pool_table,
                    },
                )?;
            }
        }
        self.feed(ctx, &Instruction::Return)?;
        self.feed(ctx, &Instruction::End)?;

        self.feed(ctx, &Instruction::End)?;
        Ok(())
    }
    /// Emit a return via exception throw.
    /// Loads the specified number of parameter locals and throws them with the tag.
    ///
    /// # Arguments
    /// * `params` - Number of parameters to pass through the exception
    /// * `tag` - Exception tag to throw
    pub fn ret(&mut self, ctx: &mut Context, params: u32, tag: EscapeTag) -> Result<(), E> {
        self.flush_bundles(ctx, self.fns.len() - 1)?;
        let EscapeTag {
            tag: TagIdx(tag_idx),
            ty: _,
        } = tag;
        for p in 0..params {
            self.feed(ctx, &Instruction::LocalGet(p))?;
        }
        self.feed(ctx, &Instruction::Throw(tag_idx))
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
    /// ```ignore
    /// # use yecta::{Reactor, JumpCallParams, FuncIdx, Pool, TableIdx, TypeIdx};
    /// # use wasm_encoder::{ValType, Function};
    /// # let mut reactor = Reactor::<std::convert::Infallible, Function>::default();
    /// # let pool = Pool { table: TableIdx(0), ty: TypeIdx(0) };
    /// // Simple unconditional jump
    /// let params = JumpCallParams::jump(FuncIdx(1), 2, pool);
    /// reactor.ji_with_params(params);
    /// ```
    pub fn ji_with_params(
        &mut self,
        ctx: &mut Context,
        params: JumpCallParams<Context, E>,
    ) -> Result<(), E> {
        let JumpCallParams {
            params: param_count,
            fixups,
            target,
            call,
            pool,
            condition,
        } = params;

        self.ji(ctx, param_count, &fixups, target, call, pool, condition)
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
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        call: Option<EscapeTag>,
        pool: Pool,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        self.flush_bundles(ctx, self.fns.len() - 1)?;
        // Track if statements for conditional branches
        if condition.is_some() {
            self.increment_if_stmts_for_predecessors(ctx)?;
        }

        match call {
            Some(escape_tag) => {
                self.emit_conditional_call(
                    ctx, params, fixups, target, escape_tag, pool, condition,
                )?;
            }
            None => {
                self.emit_conditional_jump(ctx, params, fixups, target, pool, condition)?;
            }
        }
        Ok(())
    }

    /// Increment if statement counter for all predecessor functions.
    ///
    /// If any function in the current reachable set has already reached
    /// `max_ifs_per_fn`, the entire current function group is sealed
    /// (`seal_for_split`) before the counter is incremented.  This prevents
    /// unbounded `if`-nesting and the O(N²) `End`-instruction explosion that
    /// follows from it.
    fn increment_if_stmts_for_predecessors(&mut self, ctx: &mut Context) -> Result<(), E> {
        if !self.fns.is_empty() {
            let tail_idx = self.fns.len() - 1;
            let reachable = self.transitive_preds_of(tail_idx).clone();
            let max_ifs = self.max_ifs_per_fn;
            let any_saturated = reachable
                .iter()
                .any(|&FuncIdx(i)| self.fns[i as usize].if_stmts >= max_ifs);
            if any_saturated {
                self.seal_for_split(ctx)?;
            }
        }
        let reachable_funcs = self.collect_reachable_predecessors();
        for FuncIdx(idx) in reachable_funcs {
            self.fns[idx as usize].if_stmts += 1;
        }
        Ok(())
    }

    /// Return the set of all functions reachable from the current tail by
    /// following predecessor edges transitively, including the tail itself.
    ///
    /// The result is computed lazily and cached in the tail entry's
    /// `transitive_preds` field.  Subsequent calls are O(1) until any
    /// predecessor edge changes (which sets the cache to `None`).
    fn collect_reachable_predecessors(&mut self) -> BTreeSet<FuncIdx> {
        if self.fns.is_empty() {
            return BTreeSet::new();
        }
        let tail = self.fns.len() - 1;
        self.transitive_preds_of(tail).clone()
    }

    /// Emit parameters with fixups applied.
    fn emit_params_with_fixups(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        for param_idx in 0..params {
            if let Some(fixup) = fixups.get(&param_idx) {
                fixup.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;
            } else {
                self.feed(ctx, &Instruction::LocalGet(param_idx))?;
            }
        }
        Ok(())
    }

    /// Restore parameters after a call, dropping fixed-up values and restoring original locals.
    fn restore_params_after_call(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        for param_idx in (0..params).rev() {
            if fixups.contains_key(&param_idx) {
                self.feed(ctx, &Instruction::Drop)?;
            } else {
                self.feed(ctx, &Instruction::LocalSet(param_idx))?;
            }
        }
        Ok(())
    }

    /// Emit a conditional call with exception handling.
    fn emit_conditional_call(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        escape_tag: EscapeTag,
        pool: Pool,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;
            self.feed(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
        }

        self.emit_params_with_fixups(ctx, params, fixups)?;
        self.call(ctx, target, escape_tag, pool)?;
        self.restore_params_after_call(ctx, params, fixups)?;

        if condition.is_some() {
            self.feed(ctx, &Instruction::Else)?;
        }
        Ok(())
    }

    /// Emit a conditional jump (no exception handling).
    fn emit_conditional_jump(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        pool: Pool,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        match target {
            Target::Static { func } => self.emit_static_jump(ctx, params, fixups, func, condition),
            Target::Dynamic { idx } => {
                self.emit_dynamic_jump(ctx, params, fixups, idx, pool, condition)
            }
        }
    }

    /// Emit a static (direct) jump to a known function.
    fn emit_static_jump(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        func: FuncIdx,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;
            self.feed(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;

            let FuncIdx(func_idx) = func;
            let wasm_func_idx = func_idx + self.base_func_offset;
            self.emit_params_with_fixups(ctx, params, fixups)?;
            self.feed(ctx, &Instruction::ReturnCall(wasm_func_idx))?;
            self.feed(ctx, &Instruction::Else)?;
        } else {
            // Unconditional jump: apply fixups to locals, then jump
            for (local_idx, fixup) in fixups.iter() {
                fixup.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;
                self.feed(ctx, &Instruction::LocalSet(*local_idx))?;
            }
            self.jmp(ctx, func, params)?;
        }
        Ok(())
    }

    /// Emit a dynamic (indirect) jump through a table.
    fn emit_dynamic_jump(
        &mut self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        idx: &(dyn Snippet<Context, E> + '_),
        pool: Pool,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;
            self.feed(ctx, &Instruction::If(wasm_encoder::BlockType::Empty))?;
        }

        self.emit_params_with_fixups(ctx, params, fixups)?;
        idx.emit_snippet(ctx, &mut |ctx, instr| self.feed(ctx, instr))?;

        let Pool {
            ty: TypeIdx(pool_ty),
            table: TableIdx(pool_table),
        } = pool;
        if condition.is_some() {
            self.feed(
                ctx,
                &Instruction::ReturnCallIndirect {
                    type_index: pool_ty,
                    table_index: pool_table,
                },
            )?;
            self.feed(ctx, &Instruction::Else)?;
        } else {
            self.seal(
                ctx,
                &Instruction::ReturnCallIndirect {
                    type_index: pool_ty,
                    table_index: pool_table,
                },
            )?;
        }
        Ok(())
    }
    /// Emit an unconditional jump to the target function.
    /// This creates control flow edges from all active functions to the target.
    ///
    /// # Arguments
    /// * `target` - The function index to jump to
    /// * `params` - Number of parameters to pass
    pub fn jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        // Use the per-entry transitive predecessor cache instead of a BFS.
        let reachable = self.collect_reachable_predecessors();
        for x in reachable {
            self.add_pred_checked(ctx, target, x, params)?;
        }
        Ok(())
    }
    /// Feed an instruction to all active functions.
    /// The instruction is added to the current function and all its predecessors.
    /// Each visited function's instruction count is incremented.
    pub fn feed(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        // collect_reachable_predecessors is O(1) — reads tail.transitive_preds.
        let reachable = self.collect_reachable_predecessors();
        for FuncIdx(idx) in reachable {
            self.fns[idx as usize]
                .function
                .instruction(ctx, instruction)?;
            self.fns[idx as usize].inst_count += 1;
        }
        Ok(())
    }

    /// Feed an instruction lazily to all active functions.
    /// The instruction is added lazily to the current function and all its predecessors.
    /// Each visited function's instruction count is incremented.
    pub fn feed_lazy(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        // collect_reachable_predecessors is O(1) — reads tail.transitive_preds.
        let reachable = self.collect_reachable_predecessors();
        for FuncIdx(idx) in reachable {
            //SAFETY: TODO
            self.fns[idx as usize]
                .bundles
                .push(unsafe { transmute(instruction.clone()) });
            self.fns[idx as usize].inst_count += 1;
        }
        Ok(())
    }

    /// Perform a barrier on all lazily fed instructions, ensuring they are emitted before any subsequent instructions.
    pub fn barrier(&mut self, ctx: &mut Context) -> Result<(), E> {
        if self.fns.is_empty() {
            return Ok(());
        }
        let tail_idx = self.fns.len() - 1;
        self.flush_bundles(ctx, tail_idx)
    }

    /// Seal the current function by emitting a final instruction and closing all if statements.
    /// This terminates the function and removes all predecessor edges.
    pub fn seal(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        if self.fns.is_empty() {
            return Ok(());
        }
        let tail_idx = self.fns.len() - 1;
        self.flush_bundles(ctx, tail_idx)?;
        let ifs = self.fns[tail_idx].if_stmts;
        let reachable = self.transitive_preds_of(tail_idx).clone();
        for &FuncIdx(idx) in &reachable {
            self.fns[idx as usize]
                .function
                .instruction(ctx, instruction)?;
            for _ in 0..ifs {
                self.fns[idx as usize]
                    .function
                    .instruction(ctx, &Instruction::End)?;
            }
            _ = take(&mut self.fns[idx as usize].preds);
            // Invalidate transitive cache after severing preds.
            self.fns[idx as usize].transitive_preds = None;
        }
        Ok(())
    }

    /// Get the total number of nested if statements for a function.
    fn total_ifs(&self, FuncIdx(p_idx): FuncIdx) -> usize {
        return self.fns[p_idx as usize].if_stmts;
    }
}
