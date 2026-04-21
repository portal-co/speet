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

//! ## See also
//!
//! - `docs/recompiler-guide.md` §1 — full rationale for the one-function-per-instruction
//!   model, `return_call` chain semantics, predecessor graph, and function splitting rules.
//! - `docs/lazy-store-alias-checking.md` — the `LazyStore` / `LocalPool` mechanism that
//!   enables deferred stores with runtime alias checking.
//! - `AGENTS.md` §1, §3 — agent guidance; explains why these patterns must not be "fixed".
//!
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

pub mod layout;
pub use layout::{LocalAllocator, LocalDeclarator, LocalLayout, LocalSlot, Mark};

pub mod slots;
pub use slots::{FunctionCount, PassthroughSlots, SlotAssigner};

use core::{
    convert::Infallible,
    marker::PhantomData,
    mem::{take, transmute},
    ops::{Deref, DerefMut},
};

use alloc::{
    collections::{btree_map::BTreeMap, btree_set::BTreeSet, vec_deque::VecDeque},
    vec::Vec,
};
use wasm_encoder::{BlockType, Catch, Function, Instruction, ValType};
use wax_core::build::InstructionSink;

extern crate alloc;

// ── EmitSink ──────────────────────────────────────────────────────────────────

/// Minimal emission interface used by [`speet_traps::TrapContext`].
///
/// Abstracting over this trait lets `TrapContext` drop its `F` type parameter:
/// the concrete sink (usually `Reactor<…>`) is erased to `dyn EmitSink` when
/// passed through the trap system.
///
/// `Reactor<Context, E, F, P>` provides the canonical implementation.  The
/// `emit_jmp` method delegates to `Reactor::jmp`, preserving full
/// predecessor-graph bookkeeping.
///
/// For test or stub sinks, implement `emit_jmp` to emit `unreachable` as a
/// safe fallback when no reactor is available.
pub trait EmitSink<Context, E> {
    /// Emit a single wasm instruction into the current function.
    fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E>;

    /// Emit an unconditional tail-call jump to `target`, forwarding `params`
    /// parameters.
    ///
    /// On a `Reactor` sink this calls `Reactor::jmp`, which records `target`
    /// as a successor and emits the full `local.get … return_call` sequence.
    fn emit_jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E>;
}

impl<Context, E, F, P, Gate> EmitSink<Context, E> for Reactor<Context, E, F, P, Gate>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Gate: SlotAssigner,
{
    #[inline]
    fn emit(&mut self, ctx: &mut Context, instr: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self.lock_global().len() - 1;
        self.feed_to(tail_idx, ctx, instr)
    }
    #[inline]
    fn emit_jmp(&mut self, ctx: &mut Context, target: FuncIdx, params: u32) -> Result<(), E> {
        let tail_idx = self.lock_global().len() - 1;
        self.jmp(tail_idx, ctx, target, params)
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum IndirectJumpKind {
    Table(TableIdx),
    Ref,
}
pub trait IndirectJumpHandler<Context, E> {
    fn indirect_jump(
        &self,
        ctx: &mut Context,
        target: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<IndirectJumpKind, E>;
}

/// A trait for local pool backends that can be swapped dynamically.
///
/// This trait defines the interface required by [`Reactor`] for managing
/// recyclable local variable indices. Implementations can provide different
/// allocation strategies while maintaining compatibility with the reactor.
pub trait LocalPoolBackend {
    /// Allocate one local of the requested type.
    ///
    /// Returns `Some(index)` if successful, or `None` if the pool is exhausted
    /// and no locals of the requested type are available.
    fn alloc(&mut self, ty: ValType) -> Option<u32>;

    /// Return a local index back to the pool for reuse.
    ///
    /// The index and its type are stored for future allocation. If the pool
    /// is full, this operation has no effect.
    fn free(&mut self, idx: u32, ty: ValType);

    /// Seed the pool with a contiguous range of i32 local indices.
    ///
    /// Adds `count` indices starting from `start` to the i32 pool. Used to
    /// pre-populate the pool with reserved local slots that will be used for
    /// deferred store operands and other temporary values.
    fn seed_i32(&mut self, start: u32, count: u32);

    /// Seed the pool with a contiguous range of i64 local indices.
    ///
    /// Adds `count` indices starting from `start` to the i64 pool. Used to
    /// pre-populate the pool with reserved local slots that will be used for
    /// deferred store operands and other temporary values.
    fn seed_i64(&mut self, start: u32, count: u32);
}

/// A fixed-capacity pool of recyclable wasm local indices.
///
/// The pool is split into two typed buckets — one for `i32` locals and one for
/// `i64` locals.  Both buckets are backed by stack-allocated fixed-size arrays
/// so there are no heap allocations in the hot path.
///
/// The pool is generic over a const capacity `N` that covers *both* buckets
/// combined: up to `N` i32 locals and up to `N` i64 locals may be held at the
/// same time.  In practice the pool is small (a handful of locals per
/// in-flight deferred store), so `N = 32` is more than sufficient.
///
/// # Allocation failure
/// If `alloc` returns `None` the caller is expected to flush all pending lazy
/// stores unconditionally and reset the pool, after which allocation will
/// succeed again.
pub struct LocalPool<const N: usize = 32> {
    i32s: [u32; N],
    i32_len: usize,
    i64s: [u32; N],
    i64_len: usize,
}

impl<const N: usize> LocalPool<N> {
    /// Create an empty pool.
    pub const fn new() -> Self {
        Self {
            i32s: [0; N],
            i32_len: 0,
            i64s: [0; N],
            i64_len: 0,
        }
    }

    /// Seed the pool with a contiguous range of i32 local indices
    /// `[first, first + count)`.
    pub fn seed_i32(&mut self, first: u32, count: u32) {
        for i in 0..count as usize {
            if self.i32_len < N {
                self.i32s[self.i32_len] = first + i as u32;
                self.i32_len += 1;
            }
        }
    }

    /// Seed the pool with a contiguous range of i64 local indices
    /// `[first, first + count)`.
    pub fn seed_i64(&mut self, first: u32, count: u32) {
        for i in 0..count as usize {
            if self.i64_len < N {
                self.i64s[self.i64_len] = first + i as u32;
                self.i64_len += 1;
            }
        }
    }
}

impl<const N: usize> LocalPoolBackend for LocalPool<N> {
    fn alloc(&mut self, ty: ValType) -> Option<u32> {
        match ty {
            ValType::I32 => {
                if self.i32_len == 0 {
                    return None;
                }
                self.i32_len -= 1;
                Some(self.i32s[self.i32_len])
            }
            ValType::I64 => {
                if self.i64_len == 0 {
                    return None;
                }
                self.i64_len -= 1;
                Some(self.i64s[self.i64_len])
            }
            // Only i32 and i64 are used for store operands and flags.
            _ => None,
        }
    }

    fn free(&mut self, idx: u32, ty: ValType) {
        match ty {
            ValType::I32 => {
                if self.i32_len < N {
                    self.i32s[self.i32_len] = idx;
                    self.i32_len += 1;
                }
            }
            ValType::I64 => {
                if self.i64_len < N {
                    self.i64s[self.i64_len] = idx;
                    self.i64_len += 1;
                }
            }
            _ => {}
        }
    }

    fn seed_i32(&mut self, start: u32, count: u32) {
        for i in 0..count as usize {
            if self.i32_len < N {
                self.i32s[self.i32_len] = start + i as u32;
                self.i32_len += 1;
            }
        }
    }

    fn seed_i64(&mut self, start: u32, count: u32) {
        for i in 0..count as usize {
            if self.i64_len < N {
                self.i64s[self.i64_len] = start + i as u32;
                self.i64_len += 1;
            }
        }
    }
}

impl<const N: usize> LocalPool<N> {
    /// Returns `true` if there are no free locals of *any* type in the pool.
    pub fn is_empty(&self) -> bool {
        self.i32_len == 0 && self.i64_len == 0
    }
}

impl<const N: usize> Default for LocalPool<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ── LazyStore ─────────────────────────────────────────────────────────────────

/// A store instruction that has been deferred via [`Reactor::feed_lazy`].
///
/// See `docs/lazy-store-alias-checking.md` for the full alias-checking protocol
/// and the role of each field.  See `AGENTS.md` §3 before modifying this type.
///
/// At deferral time the operands (`[addr, value]`) are popped from the wasm
/// value stack and saved into borrowed locals from the [`LocalPool`].  The
/// instruction itself is stored here so that later emission (either
/// conditionally before a load or unconditionally at a barrier) can
/// reconstruct the full sequence:
///
/// ```text
/// local.get addr_local
/// local.get val_local
/// [instr]
/// ```
///
/// Additionally, `emitted_local` (always i32) records at **runtime** whether
/// this store was already conditionally emitted before a load (1 = emitted,
/// 0 = not yet).  The unconditional flush path checks this flag to avoid
/// double-storing.
#[derive(Clone)]
pub struct LazyStore {
    /// Local holding the effective address.
    /// Type is `I32` for 32-bit memory (the default), `I64` for memory64.
    pub addr_local: u32,
    /// Value type of the address (`I32` or `I64`).
    pub addr_type: ValType,
    /// Local holding the value to store.
    pub val_local: u32,
    /// Value type of the stored value (`I32` or `I64`).
    pub val_type: ValType,
    /// Boolean local (i32): set to 1 at runtime when this store was
    /// conditionally emitted before a load; the barrier flush checks it.
    pub emitted_local: u32,
    /// The wasm store instruction (e.g. `I32Store`, `I64Store8`, …).
    pub instr: Instruction<'static>,
}

/// Index of a WebAssembly function in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct FuncIdx(pub u32);

/// Index of a WebAssembly tag (exception tag) in the module.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TagIdx(pub u32);

/// Index of a WebAssembly table in the module.
///
/// `TableIdx` also implements [`IndirectJumpHandler`], so it can be used
/// directly as the `handler` field of a [`Pool`].  The implementation emits
/// no additional instructions and returns
/// [`IndirectJumpKind::Table(self)`](IndirectJumpKind::Table), telling the
/// reactor to use a `call_indirect` / `return_call_indirect` against this
/// table index.
///
/// This means a plain `TableIdx` is sufficient for any recompiler that uses
/// a standard wasm table for indirect dispatch:
///
/// ```ignore
/// static TABLE: TableIdx = TableIdx(0);
/// let pool = Pool { handler: &TABLE, ty: TypeIdx(4) };
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct TableIdx(pub u32);

impl<Context, E> IndirectJumpHandler<Context, E> for TableIdx {
    /// No auxiliary instructions needed — the table index is carried in
    /// the returned [`IndirectJumpKind::Table`].
    fn indirect_jump(
        &self,
        _ctx: &mut Context,
        _target: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<IndirectJumpKind, E> {
        Ok(IndirectJumpKind::Table(*self))
    }
}

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
///
/// Carries the [`TypeIdx`] of the `call_indirect` / `return_call_indirect`
/// type signature and a reference to an [`IndirectJumpHandler`] that decides
/// at code-generation time how to dispatch: table lookup, `call_ref`, or any
/// custom scheme.
///
/// The common case — a plain wasm table — uses [`TableIdx`] directly as the
/// handler, since [`TableIdx`] implements [`IndirectJumpHandler`]:
///
/// ```ignore
/// // Store the table index somewhere with a lifetime at least as long as
/// // the Pool borrow.  A `static` is convenient:
/// static TABLE: TableIdx = TableIdx(0);
/// let pool = Pool { handler: &TABLE, ty: TypeIdx(4) };
/// ```
///
/// Custom dispatch logic (e.g. a call-ref scheme, a dispatcher function, or
/// a CET-hardened table) is supported by implementing [`IndirectJumpHandler`]
/// on any type and storing a reference to it here.
#[allow(missing_copy_implementations)]
pub struct Pool<'a, Context, E> {
    pub handler: &'a (dyn IndirectJumpHandler<Context, E> + 'a),
    pub ty: TypeIdx,
}

impl<'a, Context, E> Clone for Pool<'a, Context, E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, Context, E> Copy for Pool<'a, Context, E> {}

/// A no-op [`IndirectJumpHandler`] that emits `unreachable` and returns
/// `IndirectJumpKind::Ref`.
///
/// Useful when indirect jumps should never be reached at runtime but the
/// type system requires a valid handler.  For the common table-dispatch
/// case, prefer using [`TableIdx`] directly as the handler instead.
pub struct NoopHandler;

impl<Context, E> IndirectJumpHandler<Context, E> for NoopHandler {
    fn indirect_jump(
        &self,
        ctx: &mut Context,
        target: &mut (dyn InstructionSink<Context, E> + '_),
    ) -> Result<IndirectJumpKind, E> {
        target.instruction(ctx, &Instruction::Unreachable)?;
        Ok(IndirectJumpKind::Ref)
    }
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
    pub pool: Pool<'a, Context, E>,
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
    pub fn jump(func: FuncIdx, params: u32, pool: Pool<'a, Context, E>) -> Self {
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
    pub fn call(
        func: FuncIdx,
        params: u32,
        escape_tag: EscapeTag,
        pool: Pool<'a, Context, E>,
    ) -> Self {
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
        pool: Pool<'a, Context, E>,
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
    pub fn indirect_jump(
        idx: &'a (dyn Snippet<Context, E> + 'a),
        params: u32,
        pool: Pool<'a, Context, E>,
    ) -> Self {
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
pub const DEFAULT_MAX_INSTS_PER_FN: usize = 256;

/// Default maximum number of open `if` blocks that may accumulate in a single
/// generated function before a conditional branch forces a split.
pub const DEFAULT_MAX_IFS_PER_FN: usize = 16;

#[derive(Default)]
struct LockCfg {
    funcs: BTreeMap<usize, FuncCfg>,
    main: bool,
}

struct FuncCfg {
    ro: usize,
}

/// A reactor manages the generation of WebAssembly functions with control flow.
/// It handles function generation, control flow edges (predecessors), and nested if statements.
pub struct Reactor<Context, E = Infallible, F = Function, P = LocalPool, Gate = PassthroughSlots>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend,
    Gate: SlotAssigner,
{
    lock: spin::Mutex<LockCfg>,
    fns: core::cell::UnsafeCell<Vec<Entry<F>>>,
    lens: spin::Mutex<VecDeque<BTreeSet<FuncIdx>>>,
    phantom: PhantomData<(Context, E)>,
    /// Base offset added to all emitted function indices.
    /// Used when imports or helper functions precede the generated functions in the module.
    base_func_offset: u32,
    /// Maximum instructions per generated function before a forced split.
    max_insts_per_fn: usize,
    /// Maximum open `if` blocks per generated function before a forced split.
    max_ifs_per_fn: usize,
    /// Pool of recyclable local indices used to save/restore deferred store
    /// operands around alias-check `if` guards in lazy emission.
    pub local_pool: spin::Mutex<P>,
    /// Controls which guest PCs receive function slots and maps PCs to indices.
    slot_assigner: Gate,
}
impl<Context, E, F, P, Gate> Default for Reactor<Context, E, F, P, Gate>
where
    F: InstructionSink<Context, E>,
    P: LocalPoolBackend + Default,
    Gate: SlotAssigner + Default,
{
    fn default() -> Self {
        Self {
            fns: Default::default(),
            lens: Default::default(),
            phantom: Default::default(),
            base_func_offset: 0,
            max_insts_per_fn: DEFAULT_MAX_INSTS_PER_FN,
            max_ifs_per_fn: DEFAULT_MAX_IFS_PER_FN,
            local_pool: spin::Mutex::new(P::default()),
            lock: Default::default(),
            slot_assigner: Gate::default(),
        }
    }
}

impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner>
    Reactor<Context, E, F, P, Gate>
{
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
    pub fn with_base_func_offset(base_func_offset: u32) -> Self
    where
        P: Default,
        Gate: Default,
    {
        Self {
            fns: Default::default(),
            lens: Default::default(),
            phantom: Default::default(),
            base_func_offset,
            max_insts_per_fn: DEFAULT_MAX_INSTS_PER_FN,
            max_ifs_per_fn: DEFAULT_MAX_IFS_PER_FN,
            local_pool: spin::Mutex::new(P::default()),
            lock: Default::default(),
            slot_assigner: Gate::default(),
        }
    }

    /// Create a new reactor with a custom slot assigner.
    ///
    /// The slot assigner controls which guest PCs receive WASM function slots
    /// and maps each included PC to its sequential slot index.
    pub fn with_slot_assigner(gate: Gate) -> Self
    where
        P: Default,
    {
        Self {
            fns: Default::default(),
            lens: Default::default(),
            phantom: Default::default(),
            base_func_offset: 0,
            max_insts_per_fn: DEFAULT_MAX_INSTS_PER_FN,
            max_ifs_per_fn: DEFAULT_MAX_IFS_PER_FN,
            local_pool: spin::Mutex::new(P::default()),
            lock: Default::default(),
            slot_assigner: gate,
        }
    }

    /// Access the installed slot assigner.
    pub fn slot_assigner(&self) -> &Gate {
        &self.slot_assigner
    }

    /// Replace the slot assigner.
    pub fn set_slot_assigner(&mut self, gate: Gate) {
        self.slot_assigner = gate;
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
    bundles: Vec<LazyStore>,
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
    const_stack: Vec<Option<(u64, ValType)>>,
    locals_const: BTreeMap<u32, (u64, ValType)>,
    locals_virtual: BTreeSet<u32>,
    skip_depth: usize,
    block_frames: Vec<bool>, // true = taken-if frame (End should be skipped), false = normal
    /// Per-function one-slot peephole buffer.  Holds at most one static
    /// instruction awaiting possible combination with the next instruction.
    peephole: Option<Instruction<'static>>,
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
pub struct Fed<'a, Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner = PassthroughSlots> {
    pub reactor: &'a Reactor<Context, E, F, P, Gate>,
    pub tail_idx: usize,
}
impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner>
    InstructionSink<Context, E> for Fed<'_, Context, E, F, P, Gate>
{
    fn instruction(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        self.reactor.feed_to(self.tail_idx, ctx, instruction)
    }
}

impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner>
    InstructionSink<Context, E> for Reactor<Context, E, F, P, Gate>
{
    /// Emit `instruction` into the current tail function.
    ///
    /// This `InstructionSink` implementation allows `Reactor` to be passed
    /// directly to APIs that require `F: InstructionSink`, such as
    /// `speet_memory::CallbackContext::new`.
    fn instruction(&mut self, ctx: &mut Context, instruction: &Instruction<'_>) -> Result<(), E> {
        let tail_idx = self
            .fns
            .get_mut()
            .len()
            .checked_sub(1)
            .expect("Reactor::instruction (InstructionSink) called on empty reactor");
        self.feed_to(tail_idx, ctx, instruction)
    }
}
impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner>
    Reactor<Context, E, F, P, Gate>
{
    fn lock_global(&self) -> impl DerefMut<Target = Vec<Entry<F>>> + '_ {
        loop {
            let mut guard = self.lock.lock();
            if guard.main {
                continue;
            }
            if !guard.funcs.is_empty() {
                continue;
            }
            guard.main = true;
            struct Lock<'a, Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> {
                this: &'a Reactor<Context, E, F, P, Gate>,
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> Deref
                for Lock<'_, Context, E, F, P, Gate>
            {
                type Target = Vec<Entry<F>>;
                fn deref(&self) -> &Self::Target {
                    unsafe { &*self.this.fns.get() }
                }
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> DerefMut
                for Lock<'_, Context, E, F, P, Gate>
            {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    unsafe { &mut *self.this.fns.get() }
                }
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> Drop
                for Lock<'_, Context, E, F, P, Gate>
            {
                fn drop(&mut self) {
                    let mut guard = self.this.lock.lock();
                    guard.main = false;
                }
            }
            return Lock { this: self };
        }
    }
    fn lock_entry(&self, idx: usize, mut ro: bool) -> impl DerefMut<Target = Entry<F>> + '_ {
        loop {
            let mut guard = self.lock.lock();
            if guard.main {
                continue;
            }
            match ro {
                false => {
                    if guard.funcs.contains_key(&idx) {
                        continue;
                    }
                    guard.funcs.insert(idx, FuncCfg { ro: 0 });
                }
                true => {
                    match guard.funcs.get_mut(&idx) {
                        None => {
                            // No one holds this entry — claim it as a reader.
                            guard.funcs.insert(idx, FuncCfg { ro: 1 });
                        }
                        Some(x) if x.ro == 0 => {
                            // An exclusive writer holds the entry — wait.
                            continue;
                        }
                        Some(x) => {
                            // Existing readers — add ourselves.
                            x.ro += 1;
                        }
                    }
                }
            }
            struct Lock<'a, Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> {
                this: &'a Reactor<Context, E, F, P, Gate>,
                idx: usize,
                ro: bool,
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> Deref
                for Lock<'_, Context, E, F, P, Gate>
            {
                type Target = Entry<F>;
                fn deref(&self) -> &Self::Target {
                    match unsafe { &*self.this.fns.get() } {
                        g => &g[self.idx],
                    }
                }
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> DerefMut
                for Lock<'_, Context, E, F, P, Gate>
            {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    loop {
                        let mut guard = self.this.lock.lock();
                        if guard.funcs.get(&self.idx).map_or(false, |cfg| cfg.ro > 0) {
                            continue;
                        };
                        match unsafe { &mut *self.this.fns.get() } {
                            g => return &mut g[self.idx],
                        }
                    }
                }
            }
            impl<Context, E, F: InstructionSink<Context, E>, P: LocalPoolBackend, Gate: SlotAssigner> Drop
                for Lock<'_, Context, E, F, P, Gate>
            {
                fn drop(&mut self) {
                    let mut guard = self.this.lock.lock();
                    match self.ro {
                        false => {
                            let a = guard.funcs.remove(&self.idx).unwrap();
                            assert_eq!(a.ro, 0);
                        }
                        true => {
                            let remove = if let Some(x) = guard.funcs.get_mut(&self.idx) {
                                x.ro -= 1;
                                x.ro == 0
                            } else {
                                false
                            };
                            if remove {
                                guard.funcs.remove(&self.idx);
                            }
                        }
                    }
                }
            }
            return Lock {
                this: self,
                idx,
                ro,
            };
        }
    }
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
        if !self.fns.get_mut().is_empty() {
            let tail_idx = self.fns.get_mut().len() - 1;
            let reachable = self.transitive_preds_of(tail_idx).clone();
            let max_insts = self.max_insts_per_fn;
            let max_ifs = self.max_ifs_per_fn;
            let saturated = reachable.iter().any(|&FuncIdx(i)| {
                self.fns.get_mut()[i as usize].inst_count >= max_insts
                    || self.fns.get_mut()[i as usize].if_stmts >= max_ifs
            });
            if saturated {
                self.seal_for_split(ctx)?;
            }
        }
        let mut lock = self.lens.lock();
        while lock.len() < len as usize + 1 {
            lock.push_back(Default::default());
        }
        lock.iter_mut()
            .nth(len as usize)
            .unwrap()
            .insert(FuncIdx(self.fns.get_mut().len() as u32));

        // Build the direct predecessor set for the new entry from the lens queue.
        // Filter out self-references (the new entry's own index in lens[0] for len=0).
        let new_idx = FuncIdx(self.fns.get_mut().len() as u32);
        let direct_preds: BTreeSet<FuncIdx> = lock
            .pop_front()
            .into_iter()
            .flatten()
            .filter(|&p| p != new_idx)
            .collect();

        self.fns.get_mut().push(Entry {
            function: f,
            preds: direct_preds,
            if_stmts: 0,
            inst_count: 0,
            bundles: Vec::new(),
            transitive_preds: None,
            const_stack: Vec::new(),
            locals_const: BTreeMap::new(),
            locals_virtual: BTreeSet::new(),
            skip_depth: 0,
            block_frames: Vec::new(),
            peephole: None,
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
    fn seal_for_split(&self, ctx: &mut Context) -> Result<(), E> {
        // Step 1: read the tail index under a brief global lock, then drop it.
        let tail_idx = {
            let lock = self.lock_global();
            if lock.is_empty() {
                return Ok(());
            }
            lock.len() - 1
        };
        // Step 2: flush peephole — no lock held.
        self.flush_peephole(tail_idx, ctx)?;
        // Step 3: compute the reachable set via per-entry locks (no global lock).
        let reachable = self.transitive_preds_of(tail_idx);
        // Step 4: acquire the global lock once and operate on all reachable entries.
        let mut lock = self.lock_global();
        let ifs = lock[tail_idx].if_stmts;
        // Flush const stacks (needs mut entry access through the global lock).
        for &FuncIdx(fi) in &reachable {
            self.flush_const_stack(ctx, &mut lock[fi as usize], fi as usize)?;
        }
        for &FuncIdx(idx) in &reachable {
            lock[idx as usize]
                .function
                .instruction(ctx, &Instruction::Unreachable)?;
            for _ in 0..ifs {
                lock[idx as usize]
                    .function
                    .instruction(ctx, &Instruction::End)?;
            }
            _ = take(&mut lock[idx as usize].preds);
            lock[idx as usize].const_stack.clear();
            lock[idx as usize].locals_const.clear();
            lock[idx as usize].locals_virtual.clear();
            lock[idx as usize].skip_depth = 0;
            lock[idx as usize].block_frames.clear();
            // Invalidate transitive cache after severing preds.
            lock[idx as usize].transitive_preds = None;
        }
        // Clear stale predecessor assignments in the lens queue.
        self.lens.lock().clear();
        Ok(())
    }

    /// Return the transitive predecessor set for the entry at `idx`,
    /// computing and caching it via BFS if the cache is stale.
    ///
    /// The returned reference borrows `self` immutably after the cache is
    /// populated, so callers that need to mutate `self` afterward must clone
    /// the result first.
    fn transitive_preds_of(&self, idx: usize) -> BTreeSet<FuncIdx> {
        loop {
            let mut lock = self.lock_entry(idx, false);
            // If already cached, return immediately.
            if lock.transitive_preds.is_some() {
                return lock.transitive_preds.as_ref().unwrap().clone();
            }

            // BFS over direct preds to build the transitive set.
            let mut visited: BTreeSet<FuncIdx> = BTreeSet::new();
            visited.insert(FuncIdx(idx as u32));

            let mut stack: Vec<FuncIdx> = lock.preds.iter().cloned().collect();
            while let Some(p) = stack.pop() {
                if visited.contains(&p) {
                    continue;
                }
                visited.insert(p);
                let FuncIdx(pi) = p;
                let mut l = self.lock_entry(pi as usize, true);
                for &q in &l.preds {
                    if !visited.contains(&q) {
                        stack.push(q);
                    }
                }
            }

            lock.transitive_preds = Some(visited.clone());
        }
    }

    /// Add a predecessor edge from pred to succ in the control flow graph.
    ///
    /// Invalidates the transitive predecessor cache for `succ` and for every
    /// live entry that has `succ` in its cached transitive set (since those
    /// entries can now reach `pred` transitively too).
    fn add_pred(&self, succ: FuncIdx, pred: FuncIdx) {
        let FuncIdx(succ_idx) = succ;
        let mut lock = self.lock_global();
        match lock.get_mut(succ_idx as usize) {
            Some(a) => {
                a.preds.insert(pred);
                // Invalidate this entry's cache — it has a new predecessor.
                a.transitive_preds = None;
            }
            None => {
                // succ_idx is beyond the current fns vector — store in lens
                // for when that entry is created by next_with.
                let len = (succ_idx as usize)
                    .checked_sub(lock.len())
                    .expect("add_pred: succ_idx should be >= fns.len() in None branch")
                    as u32;
                let mut lens = self.lens.lock();
                while lens.len() < len as usize + 1 {
                    lens.push_back(Default::default());
                }
                lens.iter_mut().nth(len as usize).unwrap().insert(pred);
                // No live entry to invalidate; done.
                return;
            }
        }
        // Invalidate the cache of every live entry whose cached transitive set
        // included succ — those sets are now stale because they're missing the
        // new predecessors reachable through succ.
        for i in 0..lock.len() {
            if let Some(ref tp) = lock[i].transitive_preds {
                if tp.contains(&succ) {
                    lock[i].transitive_preds = None;
                }
            }
        }
    }
    /// Add a predecessor edge with cycle detection.
    /// If a cycle is detected, converts to return calls instead.
    fn add_pred_checked(
        &self,
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
            let mut lock = self.lock_global();
            let mut plock = self.local_pool.lock();
            // Clone the set so we can mutate fns while iterating.
            let pred_transitive: BTreeSet<FuncIdx> =
                lock[pred_idx as usize].transitive_preds.clone().unwrap();
            let FuncIdx(succ_idx) = succ;
            let wasm_func_idx = succ_idx + self.base_func_offset;
            for k in &pred_transitive {
                let FuncIdx(k_idx) = *k;
                let f = &mut lock[k_idx as usize];
                _ = take(&mut f.preds);
                f.transitive_preds = None; // invalidate after severing preds
                // Flush per-entry peephole before emitting ReturnCall.
                if let Some(prev) = f.peephole.take() {
                    f.function.instruction(ctx, &prev)?;
                    f.inst_count += 1;
                    Self::update_shadow_stack(f, &prev);
                }
                // Drain deferred stores: emit the flag-guarded unconditional flush.
                // We can't borrow `self.local_pool` and `f` simultaneously so we
                // drain into a local vec first.
                let stores: Vec<LazyStore> = f.bundles.drain(..).collect();
                for s in stores {
                    // Emit only if not already emitted before a load.
                    f.function
                        .instruction(ctx, &Instruction::LocalGet(s.emitted_local))?;
                    f.function.instruction(ctx, &Instruction::I32Eqz)?;
                    f.function
                        .instruction(ctx, &Instruction::If(BlockType::Empty))?;
                    f.function
                        .instruction(ctx, &Instruction::LocalGet(s.addr_local))?;
                    f.function
                        .instruction(ctx, &Instruction::LocalGet(s.val_local))?;
                    f.function.instruction(ctx, &s.instr)?;
                    f.function.instruction(ctx, &Instruction::End)?;
                    // Return locals to the pool.
                    plock.free(s.addr_local, s.addr_type);
                    plock.free(s.val_local, s.val_type);
                    plock.free(s.emitted_local, ValType::I32);
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

    pub fn flush_bundles(&self, ctx: &mut Context, idx: usize) -> Result<(), E> {
        let funcs = self.transitive_preds_of(idx).clone();
        let mut plock = self.local_pool.lock();
        for func in funcs {
            let mut lock = self.lock_entry(func.0 as usize, false);
            // Drain into a temporary vec to satisfy the borrow checker.
            let stores: Vec<LazyStore> = lock.bundles.drain(..).collect();
            for s in stores {
                let f = &mut *lock;
                // Emit only if not already conditionally emitted before a load.
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.emitted_local))?;
                f.function.instruction(ctx, &Instruction::I32Eqz)?;
                f.function
                    .instruction(ctx, &Instruction::If(BlockType::Empty))?;
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.addr_local))?;
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.val_local))?;
                f.function.instruction(ctx, &s.instr)?;
                f.function.instruction(ctx, &Instruction::End)?;
                // Return locals to the shared pool.
                plock.free(s.addr_local, s.addr_type);
                plock.free(s.val_local, s.val_type);
                plock.free(s.emitted_local, ValType::I32);
            }
        }
        Ok(())
    }

    /// Emit alias-guarded conditional stores for all pending lazy bundles
    /// before a load from `load_addr_local`.
    ///
    /// For each deferred store, emits:
    ///
    /// ```text
    /// ;; runtime alias check
    /// local.get store.addr_local
    /// local.get load_addr_local
    /// i32.eq  (or i64.eq for memory64)
    /// local.tee store.emitted_local   ;; record for the later unconditional flush
    /// if                              ;; if they alias, emit the store now
    ///   local.get store.addr_local
    ///   local.get store.val_local
    ///   [store instr]
    /// end
    /// ```
    ///
    /// The `emitted_local` flag is set to 1 at runtime when the addresses
    /// aliased.  The subsequent unconditional flush (at `barrier()` / control
    /// flow) skips stores whose flag is already set, preventing double-stores.
    ///
    /// `load_addr_type` must match the type of `load_addr_local` — `ValType::I32`
    /// for the default 32-bit memory model, `ValType::I64` for memory64.  Stores
    /// with a different `addr_type` than the load are skipped (they live in a
    /// different address space and cannot alias).
    ///
    /// All pending bundles remain in the queue; the unconditional flush will
    /// drain them and free their locals.
    pub fn flush_bundles_for_load(
        &self,
        ctx: &mut Context,
        load_addr_local: u32,
        load_addr_type: ValType,
        tail_idx: usize,
    ) -> Result<(), E> {
        if self.lock_global().is_empty() {
            return Ok(());
        }
        let funcs = self.transitive_preds_of(tail_idx).clone();
        for func in funcs {
            let mut f = self.lock_entry(func.0 as usize, false);
            for s in f.bundles.clone() {
                // Only alias-check stores in the same address space.
                if s.addr_type != load_addr_type {
                    continue;
                }
                // Alias check: store.addr == load.addr?
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.addr_local))?;
                f.function
                    .instruction(ctx, &Instruction::LocalGet(load_addr_local))?;
                let eq_instr = match load_addr_type {
                    ValType::I32 => Instruction::I32Eq,
                    ValType::I64 => Instruction::I64Eq,
                    // Other types cannot be linear-memory addresses; skip.
                    _ => continue,
                };
                f.function.instruction(ctx, &eq_instr)?;
                // Tee the result into emitted_local so the later flush can skip it.
                f.function
                    .instruction(ctx, &Instruction::LocalTee(s.emitted_local))?;
                f.function
                    .instruction(ctx, &Instruction::If(BlockType::Empty))?;
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.addr_local))?;
                f.function
                    .instruction(ctx, &Instruction::LocalGet(s.val_local))?;
                f.function.instruction(ctx, &s.instr)?;
                f.function.instruction(ctx, &Instruction::End)?;
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
        &self,
        ctx: &mut Context,
        target: Target<Context, E>,
        tag: EscapeTag,
        pool: Pool<'_, Context, E>,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.flush_peephole(tail_idx, ctx)?;
        self.flush_bundles(ctx, tail_idx)?;
        let EscapeTag {
            tag: TagIdx(tag_idx),
            ty: TypeIdx(ty_idx),
        } = tag;
        self.feed_to(
            tail_idx,
            ctx,
            &Instruction::Block(wasm_encoder::BlockType::FunctionType(ty_idx)),
        )?;
        self.feed_to(
            tail_idx,
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
                self.feed_to(tail_idx, ctx, &Instruction::Call(wasm_func_idx))?;
            }
            Target::Dynamic { idx } => {
                idx.emit_snippet(ctx, &mut |ctx, a| self.feed_to(tail_idx, ctx, a))?;
                let Pool {
                    ty: TypeIdx(pool_ty),
                    handler,
                } = pool;
                match handler.indirect_jump(
                    ctx,
                    &mut Fed {
                        reactor: self,
                        tail_idx,
                    },
                )? {
                    IndirectJumpKind::Table(TableIdx(pool_table)) => {
                        self.feed_to(
                            tail_idx,
                            ctx,
                            &Instruction::CallIndirect {
                                type_index: pool_ty,
                                table_index: pool_table,
                            },
                        )?;
                    }
                    IndirectJumpKind::Ref => {
                        self.feed_to(tail_idx, ctx, &Instruction::CallRef(pool_ty))?;
                    }
                }
            }
        }
        self.feed_to(tail_idx, ctx, &Instruction::Return)?;
        self.feed_to(tail_idx, ctx, &Instruction::End)?;

        self.feed_to(tail_idx, ctx, &Instruction::End)?;
        Ok(())
    }
    /// Emit a return via exception throw.
    /// Loads the specified number of parameter locals and throws them with the tag.
    ///
    /// # Arguments
    /// * `params` - Number of parameters to pass through the exception
    /// * `tag` - Exception tag to throw
    pub fn ret(
        &self,
        tail_idx: usize,
        ctx: &mut Context,
        params: u32,
        tag: EscapeTag,
    ) -> Result<(), E> {
        self.flush_peephole(tail_idx, ctx)?;
        self.flush_bundles(ctx, tail_idx)?;
        let EscapeTag {
            tag: TagIdx(tag_idx),
            ty: _,
        } = tag;
        for p in 0..params {
            self.feed_to(tail_idx, ctx, &Instruction::LocalGet(p))?;
        }
        self.feed_to(tail_idx, ctx, &Instruction::Throw(tag_idx))
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
        &self,
        ctx: &mut Context,
        params: JumpCallParams<Context, E>,
        tail_idx: usize,
    ) -> Result<(), E> {
        let JumpCallParams {
            params: param_count,
            fixups,
            target,
            call,
            pool,
            condition,
        } = params;

        self.ji(
            ctx,
            param_count,
            &fixups,
            target,
            call,
            pool,
            condition,
            tail_idx,
        )
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
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        call: Option<EscapeTag>,
        pool: Pool<'_, Context, E>,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        self.flush_bundles(ctx, tail_idx)?;
        // Track if statements for conditional branches
        if condition.is_some() {
            self.increment_if_stmts_for_predecessors(ctx, tail_idx)?;
        }

        match call {
            Some(escape_tag) => {
                self.emit_conditional_call(
                    ctx, params, fixups, target, escape_tag, pool, condition, tail_idx,
                )?;
            }
            None => {
                self.emit_conditional_jump(ctx, params, fixups, target, pool, condition, tail_idx)?;
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
    fn increment_if_stmts_for_predecessors(
        &self,
        ctx: &mut Context,
        tail_idx: usize,
    ) -> Result<(), E> {
        // Compute the reachable set via per-entry locks (no global lock held).
        let reachable = self.transitive_preds_of(tail_idx);
        // Check saturation: read if_stmts for each reachable entry.
        let max_ifs = self.max_ifs_per_fn;
        let saturated = reachable
            .iter()
            .any(|&FuncIdx(i)| self.lock_entry(i as usize, true).if_stmts >= max_ifs);
        if saturated {
            self.seal_for_split(ctx)?;
            // Reachable set changed after split; recompute and increment.
            let reachable2 = self.transitive_preds_of(tail_idx);
            for FuncIdx(idx) in reachable2 {
                self.lock_entry(idx as usize, false).if_stmts += 1;
            }
            return Ok(());
        }
        for FuncIdx(idx) in reachable {
            self.lock_entry(idx as usize, false).if_stmts += 1;
        }
        Ok(())
    }

    /// Emit parameters with fixups applied.
    fn emit_params_with_fixups(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        for param_idx in 0..params {
            if let Some(fixup) = fixups.get(&param_idx) {
                fixup.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;
            } else {
                self.feed_to(tail_idx, ctx, &Instruction::LocalGet(param_idx))?;
            }
        }
        Ok(())
    }

    /// Restore parameters after a call, dropping fixed-up values and restoring original locals.
    fn restore_params_after_call(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        for param_idx in (0..params).rev() {
            if fixups.contains_key(&param_idx) {
                self.feed_to(tail_idx, ctx, &Instruction::Drop)?;
            } else {
                self.feed_to(tail_idx, ctx, &Instruction::LocalSet(param_idx))?;
            }
        }
        Ok(())
    }

    /// Emit a conditional call with exception handling.
    fn emit_conditional_call(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        escape_tag: EscapeTag,
        pool: Pool<'_, Context, E>,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;
            self.feed_to(
                tail_idx,
                ctx,
                &Instruction::If(wasm_encoder::BlockType::Empty),
            )?;
        }

        self.emit_params_with_fixups(ctx, params, fixups, tail_idx)?;
        self.call(ctx, target, escape_tag, pool, tail_idx)?;
        self.restore_params_after_call(ctx, params, fixups, tail_idx)?;

        if condition.is_some() {
            self.feed_to(tail_idx, ctx, &Instruction::Else)?;
        }
        Ok(())
    }

    /// Emit a conditional jump (no exception handling).
    fn emit_conditional_jump(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        target: Target<Context, E>,
        pool: Pool<'_, Context, E>,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        match target {
            Target::Static { func } => {
                self.emit_static_jump(ctx, params, fixups, func, condition, tail_idx)
            }
            Target::Dynamic { idx } => {
                self.emit_dynamic_jump(ctx, params, fixups, idx, pool, condition, tail_idx)
            }
        }
    }

    /// Emit a static (direct) jump to a known function.
    fn emit_static_jump(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        func: FuncIdx,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;
            self.feed_to(
                tail_idx,
                ctx,
                &Instruction::If(wasm_encoder::BlockType::Empty),
            )?;

            let FuncIdx(func_idx) = func;
            let wasm_func_idx = func_idx + self.base_func_offset;
            self.emit_params_with_fixups(ctx, params, fixups, tail_idx)?;
            self.feed_to(tail_idx, ctx, &Instruction::ReturnCall(wasm_func_idx))?;
            self.feed_to(tail_idx, ctx, &Instruction::Else)?;
        } else {
            // Unconditional jump: apply fixups to locals, then jump
            for (local_idx, fixup) in fixups.iter() {
                fixup.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;
                self.feed_to(tail_idx, ctx, &Instruction::LocalSet(*local_idx))?;
            }
            self.jmp(tail_idx, ctx, func, params)?;
        }
        Ok(())
    }

    /// Emit a dynamic (indirect) jump through a table.
    fn emit_dynamic_jump(
        &self,
        ctx: &mut Context,
        params: u32,
        fixups: &BTreeMap<u32, &(dyn Snippet<Context, E> + '_)>,
        idx: &(dyn Snippet<Context, E> + '_),
        pool: Pool<'_, Context, E>,
        condition: Option<&(dyn Snippet<Context, E> + '_)>,
        tail_idx: usize,
    ) -> Result<(), E> {
        if let Some(cond_snippet) = condition {
            cond_snippet.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;
            self.feed_to(
                tail_idx,
                ctx,
                &Instruction::If(wasm_encoder::BlockType::Empty),
            )?;
        }

        self.emit_params_with_fixups(ctx, params, fixups, tail_idx)?;
        idx.emit_snippet(ctx, &mut |ctx, instr| self.feed_to(tail_idx, ctx, instr))?;

        let Pool {
            ty: TypeIdx(pool_ty),
            handler,
        } = pool;
        let i = match handler.indirect_jump(
            ctx,
            &mut Fed {
                tail_idx,
                reactor: self,
            },
        )? {
            IndirectJumpKind::Table(TableIdx(pool_table)) => Instruction::ReturnCallIndirect {
                type_index: pool_ty,
                table_index: pool_table,
            },
            IndirectJumpKind::Ref => Instruction::ReturnCallRef(pool_ty),
        };
        if condition.is_some() {
            self.feed_to(tail_idx, ctx, &i)?;
            self.feed_to(tail_idx, ctx, &Instruction::Else)?;
        } else {
            self.seal_to(tail_idx, ctx, &i)?;
        }
        Ok(())
    }
    /// Emit an unconditional jump to the target function.
    /// This creates control flow edges from all active functions to the target.
    ///
    /// # Arguments
    /// * `target` - The function index to jump to
    /// * `params` - Number of parameters to pass
    pub fn jmp(
        &self,
        target_idx: usize,
        ctx: &mut Context,
        target: FuncIdx,
        params: u32,
    ) -> Result<(), E> {
        self.flush_peephole(target_idx, ctx)?;
        // Use the per-entry transitive predecessor cache instead of a BFS.
        let reachable = self.transitive_preds_of(target_idx).clone();
        for x in reachable {
            self.add_pred_checked(ctx, target, x, params)?;
        }
        Ok(())
    }

    /// Feed an instruction to all active functions.
    /// The instruction is added to the current function and all its predecessors.
    /// Applies constant folding via per-entry shadow stacks and a reactor-level
    /// one-slot peephole optimizer.
    pub fn feed_to(
        &self,
        target: usize,
        ctx: &mut Context,
        instruction: &Instruction<'_>,
    ) -> Result<(), E> {
        let reachable = self.transitive_preds_of(target).clone();
        for FuncIdx(idx) in reachable {
            self.feed_to_entry(ctx, idx as usize, instruction)?;
        }
        Ok(())
    }

    /// Feed one instruction to one specific entry, applying the per-entry peephole.
    ///
    /// The peephole state is updated atomically under one lock acquisition.
    /// After releasing the lock, any previously-buffered instruction and/or
    /// the current instruction are forwarded to `feed_one`.
    fn feed_to_entry(&self, ctx: &mut Context, idx: usize, instruction: &Instruction<'_>) -> Result<(), E> {
        // Step 1: atomically check/update the per-entry peephole.
        let (prev_to_flush, current_buffered) = {
            let mut func = self.lock_entry(idx, false);
            let elide = match (&func.peephole, instruction) {
                (Some(Instruction::LocalGet(_)), Instruction::Drop) => true,
                (Some(Instruction::I32Const(0)), Instruction::I32Add) => true,
                (Some(Instruction::I32Const(1)), Instruction::I32Mul) => true,
                _ => false,
            };
            if elide {
                func.peephole = None;
                return Ok(());
            }
            let prev = func.peephole.take();
            let static_insn = Self::to_static_insn(instruction);
            let buffered = static_insn.is_some();
            func.peephole = static_insn;
            (prev, buffered)
        }; // lock released here
        // Step 2: emit previously-buffered instruction (if any).
        if let Some(prev) = prev_to_flush {
            self.feed_one(ctx, idx, &prev)?;
        }
        // Step 3: emit current instruction if it was not buffered.
        if !current_buffered {
            self.feed_one(ctx, idx, instruction)?;
        }
        Ok(())
    }

    /// Flush the per-entry peephole for every entry reachable from `target`.
    fn flush_peephole(&self, target: usize, ctx: &mut Context) -> Result<(), E> {
        let reachable = self.transitive_preds_of(target).clone();
        for FuncIdx(idx) in reachable {
            let prev = {
                let mut func = self.lock_entry(idx as usize, false);
                func.peephole.take()
            };
            if let Some(prev) = prev {
                self.feed_one(ctx, idx as usize, &prev)?;
            }
        }
        Ok(())
    }

    /// Convert an instruction to a `'static` version for peephole buffering.
    /// Returns `None` for instructions that cannot be trivially made static.
    fn to_static_insn(insn: &Instruction<'_>) -> Option<Instruction<'static>> {
        match insn {
            Instruction::LocalGet(n) => Some(Instruction::LocalGet(*n)),
            Instruction::LocalSet(n) => Some(Instruction::LocalSet(*n)),
            Instruction::LocalTee(n) => Some(Instruction::LocalTee(*n)),
            Instruction::GlobalGet(n) => Some(Instruction::GlobalGet(*n)),
            Instruction::GlobalSet(n) => Some(Instruction::GlobalSet(*n)),
            Instruction::I32Const(v) => Some(Instruction::I32Const(*v)),
            Instruction::I64Const(v) => Some(Instruction::I64Const(*v)),
            Instruction::I32Add => Some(Instruction::I32Add),
            Instruction::I32Sub => Some(Instruction::I32Sub),
            Instruction::I32Mul => Some(Instruction::I32Mul),
            Instruction::I32And => Some(Instruction::I32And),
            Instruction::I32Or => Some(Instruction::I32Or),
            Instruction::I32Xor => Some(Instruction::I32Xor),
            Instruction::I32Shl => Some(Instruction::I32Shl),
            Instruction::I32ShrS => Some(Instruction::I32ShrS),
            Instruction::I32ShrU => Some(Instruction::I32ShrU),
            Instruction::I32Eq => Some(Instruction::I32Eq),
            Instruction::I32Ne => Some(Instruction::I32Ne),
            Instruction::I32LtS => Some(Instruction::I32LtS),
            Instruction::I32LtU => Some(Instruction::I32LtU),
            Instruction::I32GtS => Some(Instruction::I32GtS),
            Instruction::I32GtU => Some(Instruction::I32GtU),
            Instruction::I32LeS => Some(Instruction::I32LeS),
            Instruction::I32LeU => Some(Instruction::I32LeU),
            Instruction::I32GeS => Some(Instruction::I32GeS),
            Instruction::I32GeU => Some(Instruction::I32GeU),
            Instruction::I32Eqz => Some(Instruction::I32Eqz),
            Instruction::I64Add => Some(Instruction::I64Add),
            Instruction::I64Sub => Some(Instruction::I64Sub),
            Instruction::I64Mul => Some(Instruction::I64Mul),
            Instruction::I64And => Some(Instruction::I64And),
            Instruction::I64Or => Some(Instruction::I64Or),
            Instruction::I64Xor => Some(Instruction::I64Xor),
            Instruction::I64Shl => Some(Instruction::I64Shl),
            Instruction::I64ShrS => Some(Instruction::I64ShrS),
            Instruction::I64ShrU => Some(Instruction::I64ShrU),
            Instruction::I64Eq => Some(Instruction::I64Eq),
            Instruction::I64Ne => Some(Instruction::I64Ne),
            Instruction::I64LtS => Some(Instruction::I64LtS),
            Instruction::I64LtU => Some(Instruction::I64LtU),
            Instruction::I64GtS => Some(Instruction::I64GtS),
            Instruction::I64GtU => Some(Instruction::I64GtU),
            Instruction::I64LeS => Some(Instruction::I64LeS),
            Instruction::I64LeU => Some(Instruction::I64LeU),
            Instruction::I64GeS => Some(Instruction::I64GeS),
            Instruction::I64GeU => Some(Instruction::I64GeU),
            Instruction::I64Eqz => Some(Instruction::I64Eqz),
            Instruction::I32WrapI64 => Some(Instruction::I32WrapI64),
            Instruction::I64ExtendI32S => Some(Instruction::I64ExtendI32S),
            Instruction::I64ExtendI32U => Some(Instruction::I64ExtendI32U),
            Instruction::Drop => Some(Instruction::Drop),
            Instruction::Return => Some(Instruction::Return),
            Instruction::Unreachable => Some(Instruction::Unreachable),
            Instruction::Nop => Some(Instruction::Nop),
            Instruction::End => Some(Instruction::End),
            Instruction::Else => Some(Instruction::Else),
            _ => None,
        }
    }

    /// Per-entry instruction dispatch with shadow-stack constant folding.
    fn feed_one(&self, ctx: &mut Context, idx: usize, insn: &Instruction<'_>) -> Result<(), E> {
        let mut func = self.lock_entry(idx,false);
        // Skip mode: drop instructions until skip_depth returns to 0.
        if func.skip_depth > 0 {
            match insn {
                Instruction::If(_) | Instruction::Block(_) | Instruction::Loop(_) => {
                    func.skip_depth += 1;
                }
                Instruction::End => {
                    func.skip_depth -= 1;
                }
                _ => {}
            }
            return Ok(());
        }

        // Try constant folding.
        if self.try_fold_one(&mut func, idx, insn) {
            return Ok(());
        }

        // Materialize any deferred constants the instruction needs.
        self.materialize_for(ctx, &mut func, idx, insn)?;

        // Emit the instruction.
        func.function.instruction(ctx, insn)?;
        func.inst_count += 1;

        // Update shadow stack for stack-pushing / stack-popping instructions.
        Self::update_shadow_stack(&mut func, insn);

        Ok(())
    }

    /// Try to constant-fold `insn` for entry `idx`.
    /// Returns `true` if the instruction was fully handled (not emitted, not counted).
    /// Returns `false` if the instruction should be emitted normally.
    fn try_fold_one(&self, entry: &mut Entry<F>, idx: usize, insn: &Instruction<'_>) -> bool {
        match insn {
            // ── Constant pushes: defer, push Some(v) ──────────────────
            Instruction::I32Const(v) => {
                entry.const_stack.push(Some((*v as u64, ValType::I32)));
                return true;
            }
            Instruction::I64Const(v) => {
                entry.const_stack.push(Some((*v as u64, ValType::I64)));
                return true;
            }

            // ── Drop: if top is a known constant, elide both ──────────
            Instruction::Drop => {
                if let Some(Some(_)) = entry.const_stack.last() {
                    entry.const_stack.pop();
                    return true;
                }
            }

            // ── i32 binary ops: fold if both operands known ────────────
            Instruction::I32Add
            | Instruction::I32Sub
            | Instruction::I32Mul
            | Instruction::I32And
            | Instruction::I32Or
            | Instruction::I32Xor
            | Instruction::I32Shl
            | Instruction::I32ShrS
            | Instruction::I32ShrU => {
                let len = entry.const_stack.len();
                if len >= 2 {
                    if let (Some(Some((b, _))), Some(Some((a, _)))) = (
                        entry.const_stack.get(len - 1),
                        entry.const_stack.get(len - 2),
                    ) {
                        let a = *a as i32;
                        let b = *b as i32;
                        let result: i32 = match insn {
                            Instruction::I32Add => a.wrapping_add(b),
                            Instruction::I32Sub => a.wrapping_sub(b),
                            Instruction::I32Mul => a.wrapping_mul(b),
                            Instruction::I32And => a & b,
                            Instruction::I32Or => a | b,
                            Instruction::I32Xor => a ^ b,
                            Instruction::I32Shl => a.wrapping_shl(b as u32 & 31),
                            Instruction::I32ShrS => a.wrapping_shr(b as u32 & 31),
                            Instruction::I32ShrU => ((a as u32).wrapping_shr(b as u32 & 31)) as i32,
                            _ => unreachable!(),
                        };
                        entry.const_stack.truncate(len - 2);
                        entry.const_stack.push(Some((result as u64, ValType::I32)));
                        return true;
                    }
                }
            }

            // ── i64 binary ops: fold if both operands known ────────────
            Instruction::I64Add
            | Instruction::I64Sub
            | Instruction::I64Mul
            | Instruction::I64And
            | Instruction::I64Or
            | Instruction::I64Xor
            | Instruction::I64Shl
            | Instruction::I64ShrS
            | Instruction::I64ShrU => {
                let len = entry.const_stack.len();
                if len >= 2 {
                    if let (Some(Some((b, _))), Some(Some((a, _)))) = (
                        entry.const_stack.get(len - 1),
                        entry.const_stack.get(len - 2),
                    ) {
                        let a = *a as i64;
                        let b = *b as i64;
                        let result: i64 = match insn {
                            Instruction::I64Add => a.wrapping_add(b),
                            Instruction::I64Sub => a.wrapping_sub(b),
                            Instruction::I64Mul => a.wrapping_mul(b),
                            Instruction::I64And => a & b,
                            Instruction::I64Or => a | b,
                            Instruction::I64Xor => a ^ b,
                            Instruction::I64Shl => a.wrapping_shl(b as u32 & 63),
                            Instruction::I64ShrS => a.wrapping_shr(b as u32 & 63),
                            Instruction::I64ShrU => ((a as u64).wrapping_shr(b as u32 & 63)) as i64,
                            _ => unreachable!(),
                        };
                        entry.const_stack.truncate(len - 2);
                        entry.const_stack.push(Some((result as u64, ValType::I64)));
                        return true;
                    }
                }
            }

            // ── i32 comparisons: fold if both known ───────────────────
            Instruction::I32Eq
            | Instruction::I32Ne
            | Instruction::I32LtS
            | Instruction::I32LtU
            | Instruction::I32GtS
            | Instruction::I32GtU
            | Instruction::I32LeS
            | Instruction::I32LeU
            | Instruction::I32GeS
            | Instruction::I32GeU => {
                let len = entry.const_stack.len();
                if len >= 2 {
                    if let (Some(Some((b, _))), Some(Some((a, _)))) = (
                        entry.const_stack.get(len - 1),
                        entry.const_stack.get(len - 2),
                    ) {
                        let a = *a as i32;
                        let b = *b as i32;
                        let au = a as u32;
                        let bu = b as u32;
                        let result: i32 = match insn {
                            Instruction::I32Eq => (a == b) as i32,
                            Instruction::I32Ne => (a != b) as i32,
                            Instruction::I32LtS => (a < b) as i32,
                            Instruction::I32LtU => (au < bu) as i32,
                            Instruction::I32GtS => (a > b) as i32,
                            Instruction::I32GtU => (au > bu) as i32,
                            Instruction::I32LeS => (a <= b) as i32,
                            Instruction::I32LeU => (au <= bu) as i32,
                            Instruction::I32GeS => (a >= b) as i32,
                            Instruction::I32GeU => (au >= bu) as i32,
                            _ => unreachable!(),
                        };
                        entry.const_stack.truncate(len - 2);
                        entry.const_stack.push(Some((result as u64, ValType::I32)));
                        return true;
                    }
                }
            }

            // ── i64 comparisons: fold if both known ───────────────────
            Instruction::I64Eq
            | Instruction::I64Ne
            | Instruction::I64LtS
            | Instruction::I64LtU
            | Instruction::I64GtS
            | Instruction::I64GtU
            | Instruction::I64LeS
            | Instruction::I64LeU
            | Instruction::I64GeS
            | Instruction::I64GeU => {
                let len = entry.const_stack.len();
                if len >= 2 {
                    if let (Some(Some((b, _))), Some(Some((a, _)))) = (
                        entry.const_stack.get(len - 1),
                        entry.const_stack.get(len - 2),
                    ) {
                        let a = *a as i64;
                        let b = *b as i64;
                        let au = a as u64;
                        let bu = b as u64;
                        let result: i32 = match insn {
                            Instruction::I64Eq => (a == b) as i32,
                            Instruction::I64Ne => (a != b) as i32,
                            Instruction::I64LtS => (a < b) as i32,
                            Instruction::I64LtU => (au < bu) as i32,
                            Instruction::I64GtS => (a > b) as i32,
                            Instruction::I64GtU => (au > bu) as i32,
                            Instruction::I64LeS => (a <= b) as i32,
                            Instruction::I64LeU => (au <= bu) as i32,
                            Instruction::I64GeS => (a >= b) as i32,
                            Instruction::I64GeU => (au >= bu) as i32,
                            _ => unreachable!(),
                        };
                        entry.const_stack.truncate(len - 2);
                        entry.const_stack.push(Some((result as u64, ValType::I32)));
                        return true;
                    }
                }
            }

            // ── i32 unary ─────────────────────────────────────────────
            Instruction::I32Eqz => {
                if let Some(Some((v, _))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    let result = (v as i32 == 0) as i32;
                    entry.const_stack.push(Some((result as u64, ValType::I32)));
                    return true;
                }
            }

            // ── i64 unary ─────────────────────────────────────────────
            Instruction::I64Eqz => {
                if let Some(Some((v, _))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    let result = (v as i64 == 0) as i32;
                    entry.const_stack.push(Some((result as u64, ValType::I32)));
                    return true;
                }
            }

            // ── i32 extend / wrap ─────────────────────────────────────
            Instruction::I32WrapI64 => {
                if let Some(Some((v, _))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    entry
                        .const_stack
                        .push(Some((v as i32 as u64, ValType::I32)));
                    return true;
                }
            }
            Instruction::I64ExtendI32S => {
                if let Some(Some((v, _))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    entry
                        .const_stack
                        .push(Some((v as i32 as i64 as u64, ValType::I64)));
                    return true;
                }
            }
            Instruction::I64ExtendI32U => {
                if let Some(Some((v, _))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    entry
                        .const_stack
                        .push(Some((v as u32 as u64, ValType::I64)));
                    return true;
                }
            }

            // ── local.set N ───────────────────────────────────────────
            Instruction::LocalSet(n) => {
                let n = *n;
                if let Some(Some((v, ty))) = entry.const_stack.last().cloned() {
                    entry.const_stack.pop();
                    entry.locals_const.insert(n, (v, ty));
                    entry.locals_virtual.insert(n);
                    return true; // Virtual store: don't emit local.set
                }
                // Unknown value: clear tracking for this local.
                entry.locals_const.remove(&n);
                entry.locals_virtual.remove(&n);
                // Fall through: emit local.set normally.
            }

            // ── local.tee N ───────────────────────────────────────────
            Instruction::LocalTee(n) => {
                let n = *n;
                if let Some(Some((v, ty))) = entry.const_stack.last().cloned() {
                    // Keep Some(v) on shadow stack (tee leaves value on stack).
                    entry.locals_const.insert(n, (v, ty));
                    entry.locals_virtual.insert(n);
                    return true; // Virtual tee
                }
                entry.locals_const.remove(&n);
                entry.locals_virtual.remove(&n);
                // Fall through: emit local.tee normally.
            }

            // ── local.get N ───────────────────────────────────────────
            Instruction::LocalGet(n) => {
                if let Some(&(v, ty)) = entry.locals_const.get(n) {
                    entry.const_stack.push(Some((v, ty)));
                    return true; // Replaced with inline const
                }
                // Unknown local: push None (update_shadow_stack will handle).
                // Fall through to emit + update_shadow_stack.
            }

            // ── If: constant-condition branch elimination ──────────────
            Instruction::If(_block_type) => {
                let cond = entry.const_stack.last().cloned();
                match cond {
                    Some(Some((v, _))) if v != 0 => {
                        // Known-true: don't emit If; inline the then-body and
                        // skip the else-body via skip_depth.
                        entry.const_stack.pop(); // consume condition
                        entry.block_frames.push(true); // TakenIf
                        return true; // Don't emit If
                    }
                    _ => {
                        // Unknown condition OR known-false: emit If normally.
                        // For known-false the condition constant stays on
                        // const_stack; materialize_for will emit I32Const(0)
                        // and the WASM runtime will select the else-branch.
                        entry.block_frames.push(false); // Normal
                        // Fall through to emit.
                    }
                }
            }

            // ── Block / Loop: track block frames ──────────────────────
            Instruction::Block(_) | Instruction::Loop(_) => {
                entry.block_frames.push(false); // Normal
                // Fall through to emit.
            }

            // ── End: may close a taken-if frame ───────────────────────
            Instruction::End => {
                if let Some(taken) = entry.block_frames.pop() {
                    if taken {
                        // This End closes a taken-if; don't emit it.
                        return true;
                    }
                }
                // Normal End: emit it.
            }

            // ── Else: handle taken-if by skipping else-body ───────────
            Instruction::Else => {
                // If the corresponding if was a taken-if, start skipping else.
                // Check the last block_frames entry.
                if let Some(&taken) = entry.block_frames.last() {
                    if taken {
                        // We're in the always-taken branch; skip the else body.
                        // Remove the TakenIf frame and enter skip mode.
                        entry.block_frames.pop();
                        entry.skip_depth += 1;
                        return true;
                    }
                }
                // Normal Else: emit it.
            }

            _ => {}
        }
        false
    }

    /// Materialize any deferred constants that the instruction needs from the shadow stack.
    /// For call-like instructions whose argument count is not statically known by
    /// `stack_pops`, every deferred constant is emitted first so the WASM stack
    /// contains the full set of real values before the call.
    fn materialize_for(
        &self,
        ctx: &mut Context,
        entry: &mut Entry<F>,
        idx: usize,
        insn: &Instruction<'_>,
    ) -> Result<(), E> {
        let pops = Self::stack_pops(insn);
        // Call-like instructions consume an unknown number of arguments that
        // stack_pops cannot determine without the function signature.  Flush
        // the entire shadow stack so those arguments are concretely on the
        // WASM stack before the call is emitted.
        let pops = if pops == 0 {
            let is_call = matches!(
                insn,
                Instruction::Call(_)
                    | Instruction::CallIndirect { .. }
                    | Instruction::ReturnCall(_)
                    | Instruction::ReturnCallIndirect { .. }
                    | Instruction::CallRef(_)
                    | Instruction::ReturnCallRef(_)
            );
            if is_call {
                entry.const_stack.len() // flush everything
            } else {
                return Ok(());
            }
        } else {
            pops
        };

        let len = entry.const_stack.len();
        let start = len.saturating_sub(pops);

        // Check if there are any deferred consts to materialize in range.
        let has_deferred = entry.const_stack[start..len].iter().any(|s| s.is_some());

        if !has_deferred {
            return Ok(());
        }

        // We need to emit deferred consts in order (bottom to top of the window).
        // But the WASM stack grows left-to-right, so we emit in order.
        // The slots at `start..len` correspond to the bottommost to topmost stack positions.
        // None slots are already on the WASM stack; Some slots need emitting.
        // We can only emit in stack order. If there's a None below a Some, we have
        // a gap. In that case, we can only materialize from the top contiguous Some group.
        for i in start..len {
            if let Some((v, ty)) = entry.const_stack[i] {
                let insn = match ty {
                    ValType::I32 => Instruction::I32Const(v as i32),
                    ValType::I64 => Instruction::I64Const(v as i64),
                    _ => Instruction::I64Const(v as i64),
                };
                entry.function.instruction(ctx, &insn)?;
                entry.inst_count += 1;
                entry.const_stack[i] = None; // now emitted to WASM stack
            }
        }
        Ok(())
    }

    /// Returns how many values from the top of the WASM stack the instruction consumes.
    /// This is a conservative estimate; unknown instructions return 0 (no materialization).
    fn stack_pops(insn: &Instruction<'_>) -> usize {
        match insn {
            Instruction::I32Const(_)
            | Instruction::I64Const(_)
            | Instruction::F32Const(_)
            | Instruction::F64Const(_) => 0,

            // Unary ops
            Instruction::I32Eqz
            | Instruction::I64Eqz
            | Instruction::I32Clz
            | Instruction::I32Ctz
            | Instruction::I32Popcnt
            | Instruction::I64Clz
            | Instruction::I64Ctz
            | Instruction::I64Popcnt
            | Instruction::I32WrapI64
            | Instruction::I64ExtendI32S
            | Instruction::I64ExtendI32U
            | Instruction::I32Extend8S
            | Instruction::I32Extend16S
            | Instruction::I64Extend8S
            | Instruction::I64Extend16S
            | Instruction::I64Extend32S
            | Instruction::F32Abs
            | Instruction::F32Neg
            | Instruction::F32Sqrt
            | Instruction::F64Abs
            | Instruction::F64Neg
            | Instruction::F64Sqrt => 1,

            // Binary ops
            Instruction::I32Add
            | Instruction::I32Sub
            | Instruction::I32Mul
            | Instruction::I32DivS
            | Instruction::I32DivU
            | Instruction::I32RemS
            | Instruction::I32RemU
            | Instruction::I32And
            | Instruction::I32Or
            | Instruction::I32Xor
            | Instruction::I32Shl
            | Instruction::I32ShrS
            | Instruction::I32ShrU
            | Instruction::I32Rotl
            | Instruction::I32Rotr
            | Instruction::I64Add
            | Instruction::I64Sub
            | Instruction::I64Mul
            | Instruction::I64DivS
            | Instruction::I64DivU
            | Instruction::I64RemS
            | Instruction::I64RemU
            | Instruction::I64And
            | Instruction::I64Or
            | Instruction::I64Xor
            | Instruction::I64Shl
            | Instruction::I64ShrS
            | Instruction::I64ShrU
            | Instruction::I64Rotl
            | Instruction::I64Rotr
            | Instruction::I32Eq
            | Instruction::I32Ne
            | Instruction::I32LtS
            | Instruction::I32LtU
            | Instruction::I32GtS
            | Instruction::I32GtU
            | Instruction::I32LeS
            | Instruction::I32LeU
            | Instruction::I32GeS
            | Instruction::I32GeU
            | Instruction::I64Eq
            | Instruction::I64Ne
            | Instruction::I64LtS
            | Instruction::I64LtU
            | Instruction::I64GtS
            | Instruction::I64GtU
            | Instruction::I64LeS
            | Instruction::I64LeU
            | Instruction::I64GeS
            | Instruction::I64GeU
            | Instruction::F32Eq
            | Instruction::F32Ne
            | Instruction::F32Lt
            | Instruction::F32Gt
            | Instruction::F32Le
            | Instruction::F32Ge
            | Instruction::F64Eq
            | Instruction::F64Ne
            | Instruction::F64Lt
            | Instruction::F64Gt
            | Instruction::F64Le
            | Instruction::F64Ge
            | Instruction::F32Add
            | Instruction::F32Sub
            | Instruction::F32Mul
            | Instruction::F32Div
            | Instruction::F32Min
            | Instruction::F32Max
            | Instruction::F32Copysign
            | Instruction::F64Add
            | Instruction::F64Sub
            | Instruction::F64Mul
            | Instruction::F64Div
            | Instruction::F64Min
            | Instruction::F64Max
            | Instruction::F64Copysign => 2,

            Instruction::Drop => 1,
            Instruction::Select => 3,

            Instruction::LocalGet(_) => 0,
            Instruction::LocalSet(_) => 1,
            Instruction::LocalTee(_) => 0,

            Instruction::GlobalSet(_) => 1,
            Instruction::GlobalGet(_) => 0,

            // If consumes the condition
            Instruction::If(_) => 1,
            Instruction::Block(_) | Instruction::Loop(_) => 0,
            Instruction::End | Instruction::Else | Instruction::Nop => 0,
            Instruction::Return | Instruction::Unreachable => 0,

            // Loads: consume address
            Instruction::I32Load(_)
            | Instruction::I64Load(_)
            | Instruction::F32Load(_)
            | Instruction::F64Load(_)
            | Instruction::I32Load8S(_)
            | Instruction::I32Load8U(_)
            | Instruction::I32Load16S(_)
            | Instruction::I32Load16U(_)
            | Instruction::I64Load8S(_)
            | Instruction::I64Load8U(_)
            | Instruction::I64Load16S(_)
            | Instruction::I64Load16U(_)
            | Instruction::I64Load32S(_)
            | Instruction::I64Load32U(_) => 1,

            // Stores: consume addr + value
            Instruction::I32Store(_)
            | Instruction::I64Store(_)
            | Instruction::F32Store(_)
            | Instruction::F64Store(_)
            | Instruction::I32Store8(_)
            | Instruction::I32Store16(_)
            | Instruction::I64Store8(_)
            | Instruction::I64Store16(_)
            | Instruction::I64Store32(_) => 2,

            _ => 0,
        }
    }

    /// Returns how many values the instruction pushes onto the WASM stack.
    fn stack_pushes(insn: &Instruction<'_>) -> usize {
        match insn {
            Instruction::I32Const(_)
            | Instruction::I64Const(_)
            | Instruction::F32Const(_)
            | Instruction::F64Const(_) => 1,

            // Unary ops: consume 1, push 1
            Instruction::I32Eqz
            | Instruction::I64Eqz
            | Instruction::I32Clz
            | Instruction::I32Ctz
            | Instruction::I32Popcnt
            | Instruction::I64Clz
            | Instruction::I64Ctz
            | Instruction::I64Popcnt
            | Instruction::I32WrapI64
            | Instruction::I64ExtendI32S
            | Instruction::I64ExtendI32U
            | Instruction::I32Extend8S
            | Instruction::I32Extend16S
            | Instruction::I64Extend8S
            | Instruction::I64Extend16S
            | Instruction::I64Extend32S
            | Instruction::F32Abs
            | Instruction::F32Neg
            | Instruction::F32Sqrt
            | Instruction::F64Abs
            | Instruction::F64Neg
            | Instruction::F64Sqrt => 1,

            // Binary ops: consume 2, push 1
            Instruction::I32Add
            | Instruction::I32Sub
            | Instruction::I32Mul
            | Instruction::I32DivS
            | Instruction::I32DivU
            | Instruction::I32RemS
            | Instruction::I32RemU
            | Instruction::I32And
            | Instruction::I32Or
            | Instruction::I32Xor
            | Instruction::I32Shl
            | Instruction::I32ShrS
            | Instruction::I32ShrU
            | Instruction::I32Rotl
            | Instruction::I32Rotr
            | Instruction::I64Add
            | Instruction::I64Sub
            | Instruction::I64Mul
            | Instruction::I64DivS
            | Instruction::I64DivU
            | Instruction::I64RemS
            | Instruction::I64RemU
            | Instruction::I64And
            | Instruction::I64Or
            | Instruction::I64Xor
            | Instruction::I64Shl
            | Instruction::I64ShrS
            | Instruction::I64ShrU
            | Instruction::I64Rotl
            | Instruction::I64Rotr
            | Instruction::I32Eq
            | Instruction::I32Ne
            | Instruction::I32LtS
            | Instruction::I32LtU
            | Instruction::I32GtS
            | Instruction::I32GtU
            | Instruction::I32LeS
            | Instruction::I32LeU
            | Instruction::I32GeS
            | Instruction::I32GeU
            | Instruction::I64Eq
            | Instruction::I64Ne
            | Instruction::I64LtS
            | Instruction::I64LtU
            | Instruction::I64GtS
            | Instruction::I64GtU
            | Instruction::I64LeS
            | Instruction::I64LeU
            | Instruction::I64GeS
            | Instruction::I64GeU
            | Instruction::F32Add
            | Instruction::F32Sub
            | Instruction::F32Mul
            | Instruction::F32Div
            | Instruction::F32Min
            | Instruction::F32Max
            | Instruction::F32Copysign
            | Instruction::F64Add
            | Instruction::F64Sub
            | Instruction::F64Mul
            | Instruction::F64Div
            | Instruction::F64Min
            | Instruction::F64Max
            | Instruction::F64Copysign
            | Instruction::F32Eq
            | Instruction::F32Ne
            | Instruction::F32Lt
            | Instruction::F32Gt
            | Instruction::F32Le
            | Instruction::F32Ge
            | Instruction::F64Eq
            | Instruction::F64Ne
            | Instruction::F64Lt
            | Instruction::F64Gt
            | Instruction::F64Le
            | Instruction::F64Ge => 1,

            Instruction::Drop => 0,
            Instruction::Select => 1,

            Instruction::LocalGet(_) => 1,
            Instruction::LocalSet(_) => 0,
            Instruction::LocalTee(_) => 1,

            Instruction::GlobalGet(_) => 1,
            Instruction::GlobalSet(_) => 0,

            Instruction::If(_) => 0,
            Instruction::Block(_) | Instruction::Loop(_) => 0,
            Instruction::End | Instruction::Else | Instruction::Nop => 0,
            Instruction::Return | Instruction::Unreachable => 0,

            Instruction::I32Load(_)
            | Instruction::I64Load(_)
            | Instruction::F32Load(_)
            | Instruction::F64Load(_)
            | Instruction::I32Load8S(_)
            | Instruction::I32Load8U(_)
            | Instruction::I32Load16S(_)
            | Instruction::I32Load16U(_)
            | Instruction::I64Load8S(_)
            | Instruction::I64Load8U(_)
            | Instruction::I64Load16S(_)
            | Instruction::I64Load16U(_)
            | Instruction::I64Load32S(_)
            | Instruction::I64Load32U(_) => 1,

            Instruction::I32Store(_)
            | Instruction::I64Store(_)
            | Instruction::F32Store(_)
            | Instruction::F64Store(_)
            | Instruction::I32Store8(_)
            | Instruction::I32Store16(_)
            | Instruction::I64Store8(_)
            | Instruction::I64Store16(_)
            | Instruction::I64Store32(_) => 0,

            _ => 0,
        }
    }

    /// Update the shadow stack after emitting a non-folded instruction.
    fn update_shadow_stack(entry: &mut Entry<F>, insn: &Instruction<'_>) {
        let pops = Self::stack_pops(insn);
        let pushes = Self::stack_pushes(insn);
        for _ in 0..pops {
            entry.const_stack.pop();
        }
        for _ in 0..pushes {
            entry.const_stack.push(None); // result is runtime-unknown
        }
    }

    /// Flush the shadow stack for entry `idx`, materializing all deferred constants.
    fn flush_const_stack(
        &self,
        ctx: &mut Context,
        entry: &mut Entry<F>,
        idx: usize,
    ) -> Result<(), E> {
        // Materialize any deferred constants still on the shadow stack.
        let stack: Vec<_> = entry.const_stack.drain(..).collect();
        for slot in stack {
            if let Some((v, ty)) = slot {
                let insn = match ty {
                    ValType::I32 => Instruction::I32Const(v as i32),
                    ValType::I64 => Instruction::I64Const(v as i64),
                    _ => Instruction::I64Const(v as i64),
                };
                entry.function.instruction(ctx, &insn)?;
                entry.inst_count += 1;
            }
        }
        // Materialize virtual locals: emit the deferred LocalSet instructions
        // so that the WASM local actually holds the constant value.  This is
        // required before any control-flow split (seal_for_split, seal_to, or
        // cycle-break) where the function may later be entered from a path
        // that has not seen the original constant-folded store.
        for n in entry.locals_virtual.iter().copied().collect::<alloc::vec::Vec<_>>() {
            if let Some(&(v, ty)) = entry.locals_const.get(&n) {
                let const_insn = match ty {
                    ValType::I32 => Instruction::I32Const(v as i32),
                    ValType::I64 => Instruction::I64Const(v as i64),
                    _ => Instruction::I64Const(v as i64),
                };
                entry.function.instruction(ctx, &const_insn)?;
                entry.function.instruction(ctx, &Instruction::LocalSet(n))?;
                entry.inst_count += 2;
            }
        }
        entry.locals_const.clear();
        entry.locals_virtual.clear();
        Ok(())
    }

    /// Flush const stacks for all entries reachable from the tail.
    fn flush_const_stacks_reachable(&self, ctx: &mut Context, tail_idx: usize) -> Result<(), E> {
        let mut lock = self.lock_global();
        if lock.is_empty() {
            return Ok(());
        }
        drop(lock);
        let reachable = self.transitive_preds_of(tail_idx).clone();
        lock = self.lock_global();
        for FuncIdx(idx) in reachable {
            self.flush_const_stack(ctx, &mut lock[idx as usize], idx as usize)?;
        }
        Ok(())
    }

    /// Defer a store instruction lazily.
    ///
    /// At call time `[addr, value]` must be on the wasm stack.  This method:
    ///
    /// 1. Attempts to allocate three locals from [`Reactor::local_pool`]:
    ///    - `addr_local` (`addr_type`) — saves the effective address
    ///    - `val_local` (`val_type`) — saves the value
    ///    - `emitted_local` (i32) — runtime flag indicating whether this store
    ///      was already conditionally emitted before a load
    ///
    /// 2. **Pool has space**: emits `LocalSet(val_local)`, `LocalSet(addr_local)`
    ///    (consuming the stack), initialises `emitted_local` to 0, and queues a
    ///    [`LazyStore`].
    ///
    /// 3. **Pool is exhausted**: flushes all existing lazy bundles
    ///    unconditionally first (freeing their locals), then emits the store
    ///    instruction eagerly.  This is the safe fallback that preserves
    ///    correctness at the cost of disabling further reordering for this
    ///    store.
    ///
    /// `addr_type` must match the address type on the stack: `ValType::I32`
    /// for the default 32-bit memory model, `ValType::I64` for memory64.
    ///
    /// `val_type` must be `ValType::I32` or `ValType::I64` matching the value
    /// type consumed by `instruction`.
    pub fn feed_lazy(
        &self,
        ctx: &mut Context,
        addr_type: ValType,
        val_type: ValType,
        instruction: &Instruction<'static>,
        tail_idx: usize,
    ) -> Result<(), E> {
        let mut local_pool = self.local_pool.lock();
        // Try to allocate all three locals up front.  If any allocation fails,
        // fall back to eager emission.  We allocate emitted_local last so that
        // partial allocation doesn't waste pool entries on a failed attempt.
        let addr_opt = local_pool.alloc(addr_type);
        let val_opt = local_pool.alloc(val_type);
        let flag_opt = local_pool.alloc(ValType::I32);

        match (addr_opt, val_opt, flag_opt) {
            (Some(addr_local), Some(val_local), Some(emitted_local)) => {
                // Consume [addr, value] from the stack into locals.  The value
                // is on top, so set it first.
                let reachable = self.transitive_preds_of(tail_idx).clone();
                for FuncIdx(idx) in reachable {
                    let mut func = self.lock_entry(idx as usize,false);
                    func.function
                        .instruction(ctx, &Instruction::LocalSet(val_local))?;
                    func.function
                        .instruction(ctx, &Instruction::LocalSet(addr_local))?;
                    // Initialise the emitted flag to 0.
                    func.function.instruction(ctx, &Instruction::I32Const(0))?;
                    func.function
                        .instruction(ctx, &Instruction::LocalSet(emitted_local))?;
                    func.inst_count += 4;
                    func.bundles.push(LazyStore {
                        addr_local,
                        addr_type,
                        val_local,
                        val_type,
                        emitted_local,
                        instr: instruction.clone(),
                    });
                }
            }
            (addr_opt, val_opt, flag_opt) => {
                // Return any successfully allocated locals before bailing.
                if let Some(l) = flag_opt {
                    local_pool.free(l, ValType::I32);
                }
                if let Some(l) = val_opt {
                    local_pool.free(l, val_type);
                }
                if let Some(l) = addr_opt {
                    local_pool.free(l, addr_type);
                }
                // Flush existing bundles so those locals are freed.
                let tail = self.lock_global().len().saturating_sub(1);
                self.flush_bundles(ctx, tail)?;
                // Emit this store eagerly — the stack still has [addr, value].
                self.feed_to(tail_idx, ctx, instruction)?;
            }
        }
        Ok(())
    }

    /// Seal the current function by emitting a final instruction and closing all if statements.
    /// This terminates the function and removes all predecessor edges.
    pub fn seal_to(
        &self,
        tail_idx: usize,
        ctx: &mut Context,
        instruction: &Instruction<'_>,
    ) -> Result<(), E> {
        self.flush_peephole(tail_idx, ctx)?;
        self.flush_bundles(ctx, tail_idx)?;
        self.flush_const_stacks_reachable(ctx, tail_idx)?;
        let ifs = self.lock_entry(tail_idx,true).if_stmts;
        let reachable = self.transitive_preds_of(tail_idx).clone();
        for &FuncIdx(idx) in &reachable {
            let mut func = self.lock_entry(idx as usize,false);
            func.function.instruction(ctx, instruction)?;
            for _ in 0..ifs {
                func.function.instruction(ctx, &Instruction::End)?;
            }
            _ = take(&mut func.preds);
            func.const_stack.clear();
            func.locals_const.clear();
            func.locals_virtual.clear();
            func.skip_depth = 0;
            func.block_frames.clear();
            // Invalidate transitive cache after severing preds.
            func.transitive_preds = None;
        }
        Ok(())
    }

    /// Get the total number of nested if statements for a function.
    fn total_ifs(&self, FuncIdx(p_idx): FuncIdx) -> usize {
        return self.lock_entry(p_idx as usize,true).if_stmts;
    }

    /// Consume this reactor and return all built functions in order.
    ///
    /// Call this after all instructions have been emitted and `seal` has been
    /// called on each function group.  The returned vector may be passed
    /// directly to a `wasm_encoder::CodeSection`.
    pub fn into_fns(mut self) -> Vec<F> {
        self.fns.get_mut().drain(..).map(|e| e.function).collect()
    }

    /// Return the number of functions currently held in this reactor.
    pub fn fn_count(&self) -> usize {
        self.lock_global().len()
    }

    /// Non-consuming variant of [`into_fns`](Self::into_fns).
    ///
    /// Moves all compiled functions out, clears the internal function vec and
    /// predecessor queue, and advances `base_func_offset` by the drained count.
    /// The reactor is immediately ready to compile the next binary unit.
    ///
    /// # Example
    /// ```ignore
    /// let fns_a = reactor.drain_fns();   // reactor now offset by fns_a.len()
    /// // ... compile more functions into reactor ...
    /// let fns_b = reactor.drain_fns();   // reactor offset by fns_a.len() + fns_b.len()
    /// ```
    pub fn drain_fns(&mut self) -> Vec<F> {
        let count = self.lock_global().len() as u32;
        let result = self.lock_global().drain(..).map(|e| e.function).collect();
        self.base_func_offset += count;
        self.lens.get_mut().clear();
        result
    }

    /// Run a closure with mutable access to the local-pool backend.
    ///
    /// Acquires the internal `spin::Mutex<P>` for the duration of the
    /// closure.  This is the preferred way to call `seed_i32`, `seed_i64`,
    /// `alloc`, or `free` on the pool without directly naming the mutex.
    ///
    /// # Example
    /// ```ignore
    /// reactor.with_local_pool(|p| p.seed_i32(first, count));
    /// ```
    pub fn with_local_pool<R>(&self, f: impl FnOnce(&mut P) -> R) -> R {
        f(&mut *self.local_pool.lock())
    }

    /// Borrow the current tail function as a [`Fed`] sink.
    ///
    /// The `tail_idx` is computed from the current function count, so this
    /// method is only correct when called in a context where exactly one
    /// function is being actively built (the sequential single-instruction
    /// translation loop).  For parallel emission, construct [`Fed`] directly
    /// with the appropriate `tail_idx`.
    ///
    /// # Panics
    /// Panics if the reactor contains no functions yet.
    pub fn tail(&self) -> Fed<'_, Context, E, F, P, Gate> {
        let tail_idx = self
            .fn_count()
            .checked_sub(1)
            .expect("Reactor::tail called on empty reactor");
        Fed { reactor: self, tail_idx }
    }
}
