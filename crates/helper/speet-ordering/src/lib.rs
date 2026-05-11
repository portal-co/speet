//! # speet-ordering
//!
//! Weakly-ordered memory emission helpers for speet recompilers.
//!
//! ## What this crate provides
//!
//! Guest ISAs such as RISC-V and MIPS have weak memory models: ordinary loads
//! and stores may be observed out of program order by other agents, and
//! explicit `FENCE` / `SYNC` instructions are required to impose ordering.
//! WebAssembly's plain memory instructions have *sequentially consistent*
//! semantics within a single thread but the host engine is free to reorder
//! them across functions or within a straight-line sequence under certain
//! conditions.
//!
//! This crate exposes a single knob — [`MemOrder`] — that controls how a
//! recompiler emits memory instructions:
//!
//! * **[`MemOrder::Strong`]** (default) — every load and store is emitted
//!   eagerly via [`Reactor::feed`].  `FENCE`/`SYNC` become a `barrier()`
//!   call which flushes any pending lazy instructions.  This mode is correct
//!   for binaries that rely on sequential consistency (e.g. some RISC-V
//!   programs written against the Linux/TSO assumption).
//!
//! * **[`MemOrder::Relaxed`]** — stores are emitted with
//!   [`Reactor::feed_lazy`], which defers them until the next control-flow
//!   boundary (or explicit `FENCE`/`SYNC`).  This allows the yecta reactor
//!   to sink stores past intervening pure instructions and, critically, to
//!   deduplicate stores that appear in every predecessor of a join point.
//!   Loads remain eager.  `FENCE`/`SYNC` call `reactor.barrier()` to commit
//!   all pending stores in program order.
//!
//! ## Atomic subflag
//!
//! [`AtomicOpts`] is a separate opt-in flag that, when enabled, switches the
//! *wasm instruction* used for stores and loads to their `I32AtomicStore` /
//! `I64AtomicStore` / `I32AtomicLoad` / `I64AtomicLoad` equivalents.  This
//! is **independent** of `MemOrder`: you can have relaxed ordering with plain
//! stores, or strong ordering with atomic stores.  The atomic subflag exists
//! to prepare infrastructure for shared-memory correctness without forcing
//! all recompiler output to use the heavier atomic instructions.
//!
//! ## Integration
//!
//! Recompilers hold a `MemOrder` value and call [`emit_store`] /
//! [`emit_load`] / [`emit_fence`] in place of bare `reactor.feed(…)` calls
//! for memory instructions:
//!
//! ```ignore
//! // In translate_store:
//! speet_ordering::emit_store(ctx, &mut self.reactor, self.mem_order, &store_instr)?;
//!
//! // In translate_load:
//! speet_ordering::emit_load(ctx, &mut self.reactor, &load_instr)?;
//!
//! // In FENCE / SYNC handling:
//! speet_ordering::emit_fence(ctx, &mut self.reactor, self.mem_order)?;
//! ```
//!
//! ## See also
//!
//! - `docs/lazy-store-alias-checking.md` — full design: alias-check `if` blocks,
//!   `emitted_local` double-store prevention, pool sizing, memory64 notes.
//! - `AGENTS.md` §3 — agent guidance; explains why lazy deferral and alias checks
//!   must not be collapsed into eager flushes.

#![no_std]

use speet_link_core::ReactorContext;
use wasm_encoder::{Instruction, MemArg, ValType};
use wax_core::build::InstructionSink;

// ── MemOrder ──────────────────────────────────────────────────────────────────

/// Controls whether stores are emitted eagerly or lazily.
///
/// See the [crate-level documentation](self) for a full description of each
/// variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MemOrder {
    /// Every store is emitted immediately via [`Reactor::feed`].
    ///
    /// `FENCE`/`SYNC` instructions flush any already-pending lazy stores via
    /// [`Reactor::barrier`] and are otherwise no-ops.
    ///
    /// This is the **default** and is always correct.
    #[default]
    Strong,

    /// Stores are emitted via [`Reactor::feed_lazy`], allowing the reactor to
    /// defer and reorder them until the next control-flow boundary or explicit
    /// `FENCE`/`SYNC`.
    ///
    /// Loads remain eager.  `FENCE`/`SYNC` call [`Reactor::barrier`] to
    /// commit all pending stores in program order before any subsequent
    /// instructions.
    ///
    /// Only enable this for binaries that conform to the RISC-V / MIPS weak
    /// memory model and do not rely on TSO-like store visibility.
    Relaxed,
}

// ── AtomicOpts ────────────────────────────────────────────────────────────────

/// Optional subflag: substitute plain loads/stores with wasm atomic variants.
///
/// This is independent of [`MemOrder`].  It exists to prepare the
/// infrastructure for shared-memory correctness without making the default
/// output heavier.
///
/// When `use_atomic_insns` is `true`, [`emit_store`] and [`emit_load`] will
/// replace the wasm instruction with its `I32Atomic*` / `I64Atomic*`
/// equivalent (preserving the `MemArg`).  Instructions that have no atomic
/// counterpart (e.g. `F32Store`, sub-32-bit stores that already go through
/// an integer narrowing) are emitted as-is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AtomicOpts {
    /// When `true`, use wasm atomic load/store instructions.
    pub use_atomic_insns: bool,
}

impl AtomicOpts {
    /// Atomic instructions disabled (the default).
    pub const NONE: Self = Self {
        use_atomic_insns: false,
    };
    /// Atomic instructions enabled.
    pub const ATOMIC: Self = Self {
        use_atomic_insns: true,
    };
}

// ── Instruction substitution ──────────────────────────────────────────────────

/// Attempt to substitute a plain wasm load instruction with its atomic
/// equivalent.  Returns the (possibly replaced) instruction.
///
/// Only `I32Load` and `I64Load` have direct atomic counterparts; all other
/// load variants are returned unchanged.
fn atomic_load_equiv(instr: &Instruction<'static>) -> Instruction<'static> {
    match instr {
        Instruction::I32Load(m) => Instruction::I32AtomicLoad(*m),
        Instruction::I64Load(m) => Instruction::I64AtomicLoad(*m),
        other => other.clone(),
    }
}

/// Attempt to substitute a plain wasm store instruction with its atomic
/// equivalent.  Returns the (possibly replaced) instruction.
///
/// Only `I32Store` and `I64Store` have direct atomic counterparts; all other
/// store variants (sub-word stores, float stores) are returned unchanged.
fn atomic_store_equiv(instr: &Instruction<'static>) -> Instruction<'static> {
    match instr {
        Instruction::I32Store(m) => Instruction::I32AtomicStore(*m),
        Instruction::I64Store(m) => Instruction::I64AtomicStore(*m),
        other => other.clone(),
    }
}

/// Return the wasm value type of the *value* operand consumed by a store
/// instruction, or `None` if the store cannot be lazily deferred (e.g.
/// float stores, whose value type has no counterpart in the i32/i64 local
/// pool).
///
/// All narrow integer stores (`I32Store8`, `I32Store16`, `I32Store32`) take an
/// i32 value; wide stores (`I64Store`, `I64Store8`, …) take an i64 value.
/// Float stores (`F32Store`, `F64Store`) return `None` — they are always
/// emitted eagerly.
fn store_val_type(instr: &Instruction<'static>) -> Option<ValType> {
    match instr {
        Instruction::I64Store(_)
        | Instruction::I64Store8(_)
        | Instruction::I64Store16(_)
        | Instruction::I64Store32(_)
        | Instruction::I64AtomicStore(_)
        | Instruction::I64AtomicStore8(_)
        | Instruction::I64AtomicStore16(_)
        | Instruction::I64AtomicStore32(_) => Some(ValType::I64),
        // Float stores cannot be saved in the i32/i64 local pool.
        Instruction::F32Store(_) | Instruction::F64Store(_) => None,
        // Everything else (I32Store, I32Store8, I32Store16, atomic variants) → i32.
        _ => Some(ValType::I32),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Emit a memory **store** instruction.
///
/// * When `order` is [`MemOrder::Strong`]: the instruction is emitted
///   eagerly via [`Reactor::feed`].
/// * When `order` is [`MemOrder::Relaxed`]: the instruction is queued with
///   [`Reactor::feed_lazy`] so the reactor can defer it until the next
///   control-flow boundary.  Float stores (`F32Store`, `F64Store`) are always
///   emitted eagerly regardless of `order` because their value type (`f32` /
///   `f64`) cannot be saved in the integer local pool.
///
/// If `atomic.use_atomic_insns` is `true` and the instruction has an atomic
/// counterpart, the atomic form is used instead.
///
/// The `instr` argument must be a `'static` instruction (i.e. one that does
/// not borrow a name or byte string) because `feed_lazy` requires
/// `'static`.  All integer store instructions satisfy this.
///
/// `addr_type` is the wasm value type of the address operand on the stack:
/// `ValType::I32` for ordinary (32-bit) linear memory, `ValType::I64` for
/// memory64.  This is recorded in the [`LazyStore`] so that alias checks
/// against loads use the correct equality instruction.
pub fn emit_store<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    order: MemOrder,
    atomic: AtomicOpts,
    addr_type: ValType,
    instr: Instruction<'static>,
    tail_idx: usize,
) -> Result<(), E> {
    let instr = if atomic.use_atomic_insns {
        atomic_store_equiv(&instr)
    } else {
        instr
    };
    match order {
        MemOrder::Strong => rctx.feed(ctx, tail_idx, &instr),
        MemOrder::Relaxed => {
            match store_val_type(&instr) {
                Some(vt) => rctx.feed_lazy(ctx, addr_type, vt, &instr, tail_idx),
                // Float stores are always eager — no pool support for f32/f64.
                None => rctx.feed(ctx, tail_idx, &instr),
            }
        }
    }
}

/// Emit a memory **load** instruction.
///
/// Before emitting the load, any deferred stores that might alias
/// `addr_local` are conditionally flushed via
/// [`Reactor::flush_bundles_for_load`].  Each pending store is guarded by a
/// runtime equality check between its saved address local and `addr_local`;
/// if they match at runtime the store is emitted immediately so the load
/// sees the up-to-date value.
///
/// The `addr_local` argument must hold the effective load address at the
/// point of the call; `addr_type` must be `ValType::I32` for ordinary memory
/// and `ValType::I64` for memory64.  The address must already be on the wasm
/// stack as well (it is consumed by the actual load instruction); callers
/// should `local.tee` it into `addr_local` just before calling this function:
///
/// ```text
/// ;; address computation ...
/// local.tee addr_local          ;; save for alias check; keep on stack
/// emit_load(…, addr_local, addr_type, …)
/// ```
///
/// If `atomic.use_atomic_insns` is `true` and the instruction has an atomic
/// counterpart, the atomic form is used instead.
pub fn emit_load<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    addr_local: u32,
    addr_type: ValType,
    atomic: AtomicOpts,
    instr: Instruction<'static>,
    tail_idx: usize,
) -> Result<(), E> {
    let instr = if atomic.use_atomic_insns {
        atomic_load_equiv(&instr)
    } else {
        instr
    };
    // Flush any pending stores that might alias this load's address.
    rctx.flush_for_load(ctx, addr_local, addr_type, tail_idx)?;
    rctx.feed(ctx, tail_idx, &instr)
}

// ── Atomic RMW helpers ────────────────────────────────────────────────────────

/// Width of an atomic read-modify-write operation.
///
/// Both RISC-V (A extension) and MIPS (future) expose 32-bit AMO/LL-SC;
/// RISC-V RV64 uses the same 32-bit variant of the encoding but the context
/// in this crate models them explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmwWidth {
    /// 32-bit (wasm `i32.atomic.rmw.*`)
    W32,
    /// 64-bit (wasm `i64.atomic.rmw.*`)
    W64,
}

/// ISA-neutral read-modify-write operation.
///
/// Maps directly onto the wasm `i32.atomic.rmw.*` / `i64.atomic.rmw.*` family
/// where a direct counterpart exists.  `Min`, `Max`, `Minu`, and `Maxu` have
/// no single wasm RMW opcode and are synthesised with a `cmpxchg` loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmwOp {
    /// Atomic exchange (AMOSWAP).
    Swap,
    /// Atomic fetch-and-add (AMOADD).
    Add,
    /// Atomic fetch-and-xor (AMOXOR).
    Xor,
    /// Atomic fetch-and-and (AMOAND).
    And,
    /// Atomic fetch-and-or (AMOOR).
    Or,
    /// Atomic fetch-and-signed-min (AMOMIN).  Synthesised via cmpxchg.
    Min,
    /// Atomic fetch-and-signed-max (AMOMAX).  Synthesised via cmpxchg.
    Max,
    /// Atomic fetch-and-unsigned-min (AMOMINU).  Synthesised via cmpxchg.
    Minu,
    /// Atomic fetch-and-unsigned-max (AMOMAXU).  Synthesised via cmpxchg.
    Maxu,
}

/// The alignment (log₂ bytes) for an [`RmwWidth`] access.
fn rmw_align(w: RmwWidth) -> u32 {
    match w {
        RmwWidth::W32 => 2, // 4-byte aligned
        RmwWidth::W64 => 3, // 8-byte aligned
    }
}

/// The canonical wasm [`MemArg`] (offset 0, natural alignment) for an RMW
/// access of the given width into memory index 0.
fn rmw_memarg(w: RmwWidth) -> MemArg {
    MemArg {
        offset: 0,
        align: rmw_align(w),
        memory_index: 0,
    }
}

/// Emit an **atomic load-reserved** (`LR.W` / `LR.D`).
///
/// The effective address is already on the wasm stack when this is called.
/// The loaded value is left on the stack; the caller is responsible for
/// storing it into the destination local.
///
/// Before emitting the load, any pending lazy stores that alias `addr_local`
/// are conditionally flushed (see [`emit_load`]).  `addr_local` must hold the
/// same effective address that is on the stack.
///
/// When `atomic.use_atomic_insns` is `true` this emits an
/// `I32AtomicLoad` / `I64AtomicLoad`.  On a shared-memory wasm target those
/// instructions provide the sequentially-consistent ordering that an LR
/// requires.
///
/// When `atomic.use_atomic_insns` is `false` — an environment that does not
/// support the wasm threads proposal — this falls back to a plain
/// `I32Load` / `I64Load`.  There is no hardware reservation mechanism in this
/// path; correctness relies on single-threaded execution.
///
/// The `_order` argument is accepted for API symmetry; wasm atomic loads are
/// always sequentially consistent within a thread regardless of the guest
/// ordering annotation.
pub fn emit_lr<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    width: RmwWidth,
    atomic: AtomicOpts,
    addr_local: u32,
    addr_type: ValType,
    _order: MemOrder,
    tail_idx: usize,
) -> Result<(), E> {
    let m = rmw_memarg(width);
    let instr = if atomic.use_atomic_insns {
        match width {
            RmwWidth::W32 => Instruction::I32AtomicLoad(m),
            RmwWidth::W64 => Instruction::I64AtomicLoad(m),
        }
    } else {
        match width {
            RmwWidth::W32 => Instruction::I32Load(m),
            RmwWidth::W64 => Instruction::I64Load(m),
        }
    };
    // Flush any pending stores that might alias this load.
    rctx.flush_for_load(ctx, addr_local, addr_type, tail_idx)?;
    rctx.feed(ctx, tail_idx, &instr)
}

/// Emit an **atomic store-conditional** (`SC.W` / `SC.D`).
///
/// Stack on entry: `[addr, value]`.
/// In the single-threaded wasm model SC always succeeds; the caller writes
/// the success value (0 for RISC-V, 1 for MIPS) into the destination register
/// after this call.
///
/// When `atomic.use_atomic_insns` is `true` this emits an
/// `I32AtomicStore` / `I64AtomicStore`.
///
/// When `atomic.use_atomic_insns` is `false` this emits a plain
/// `I32Store` / `I64Store`.  The `order` argument controls whether the store
/// is emitted eagerly ([`MemOrder::Strong`]) or deferred via `feed_lazy`
/// ([`MemOrder::Relaxed`]), exactly as [`emit_store`] does for ordinary
/// stores.  Guest atomic instructions that carry `.rl` (release) or `.aqrl`
/// semantics should therefore pass the recompiler's `mem_order` field here so
/// that a relaxed-mode binary still enjoys store sinking.
pub fn emit_sc<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    width: RmwWidth,
    atomic: AtomicOpts,
    order: MemOrder,
    tail_idx: usize,
) -> Result<(), E> {
    let m = rmw_memarg(width);
    if atomic.use_atomic_insns {
        let instr = match width {
            RmwWidth::W32 => Instruction::I32AtomicStore(m),
            RmwWidth::W64 => Instruction::I64AtomicStore(m),
        };
        rctx.feed(ctx, tail_idx, &instr)
    } else {
        // Plain store — eligible for feed_lazy under MemOrder::Relaxed.
        let instr = match width {
            RmwWidth::W32 => Instruction::I32Store(m),
            RmwWidth::W64 => Instruction::I64Store(m),
        };
        emit_store(ctx, rctx, order, AtomicOpts::NONE, ValType::I32, instr, tail_idx)
    }
}

/// Emit an **atomic read-modify-write** sequence.
///
/// Stack on entry: `[addr, src]` (both values must also be available via
/// `addr_local` and `src_local` so the cmpxchg retry loop can reload them).
/// Stack on exit: `[old_value]` — the value at the memory location *before*
/// the operation.
///
/// # Atomic vs. non-atomic paths
///
/// When `atomic.use_atomic_insns` is `true`:
/// - `Swap`, `Add`, `Xor`, `And`, `Or` lower to a single wasm
///   `i32/i64.atomic.rmw.*` instruction.
/// - `Min`, `Max`, `Minu`, `Maxu` are synthesised with an
///   `i32/i64.atomic.rmw.cmpxchg` retry loop.
///
/// When `atomic.use_atomic_insns` is `false` (no wasm threads proposal):
/// - `Swap`, `Add`, `Xor`, `And`, `Or` lower to a plain load, the scalar
///   operation, and a plain store.  The store respects `order` so that a
///   `MemOrder::Relaxed` recompiler still defers it via `feed_lazy`.
/// - `Min`, `Max`, `Minu`, `Maxu` use the same load/op/store sequence
///   without any CAS loop — correct for single-threaded wasm.
///
/// In both paths `scratch_local` must be a wasm local of the same value type
/// as `width`; it is used only by the min/max synthesis paths and is ignored
/// for all direct-operation cases.
pub fn emit_rmw<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    width: RmwWidth,
    op: RmwOp,
    atomic: AtomicOpts,
    order: MemOrder,
    addr_local: u32,
    src_local: u32,
    scratch_local: u32,
    tail_idx: usize,
) -> Result<(), E> {
    let m = rmw_memarg(width);

    // ── Direct scalar ops: Swap / Add / Xor / And / Or ────────────────────
    //
    // These five operations have both a direct wasm atomic RMW opcode (used
    // when wasm atomics are available) and a straightforward non-atomic
    // lowering (used when they are not).
    //
    // Non-atomic lowering sequence for e.g. Add (stack entry: [addr, src]):
    //   addr                     ;; drop src, stash addr, re-push in order
    //   src already in src_local
    //   load old from addr        ;; old
    //   tee scratch               ;; old (also stashed)
    //   local.get addr_local      ;; old, addr
    //   local.get scratch_local   ;; old, addr, old
    //   local.get src_local       ;; old, addr, old, src
    //   [op]                      ;; old, addr, new
    //   store(addr, new)          ;; old  (addr+new consumed by store)
    //
    // For Swap the "op" is simply to discard old and use src directly:
    //   tee scratch → old on stack
    //   addr, src → store
    //   old stays as result

    // Identify whether this is a direct op (not a min/max synthesis).
    let direct_atomic: Option<Instruction<'static>> = if atomic.use_atomic_insns {
        match (width, op) {
            (RmwWidth::W32, RmwOp::Swap) => Some(Instruction::I32AtomicRmwXchg(m)),
            (RmwWidth::W64, RmwOp::Swap) => Some(Instruction::I64AtomicRmwXchg(m)),
            (RmwWidth::W32, RmwOp::Add) => Some(Instruction::I32AtomicRmwAdd(m)),
            (RmwWidth::W64, RmwOp::Add) => Some(Instruction::I64AtomicRmwAdd(m)),
            (RmwWidth::W32, RmwOp::Xor) => Some(Instruction::I32AtomicRmwXor(m)),
            (RmwWidth::W64, RmwOp::Xor) => Some(Instruction::I64AtomicRmwXor(m)),
            (RmwWidth::W32, RmwOp::And) => Some(Instruction::I32AtomicRmwAnd(m)),
            (RmwWidth::W64, RmwOp::And) => Some(Instruction::I64AtomicRmwAnd(m)),
            (RmwWidth::W32, RmwOp::Or) => Some(Instruction::I32AtomicRmwOr(m)),
            (RmwWidth::W64, RmwOp::Or) => Some(Instruction::I64AtomicRmwOr(m)),
            _ => None,
        }
    } else {
        None
    };

    if let Some(instr) = direct_atomic {
        // Single wasm atomic RMW; consumes [addr, src], leaves [old].
        return rctx.feed(ctx, tail_idx, &instr);
    }

    // ── Non-atomic direct ops: load / scalar-op / store ───────────────────
    //
    // Reached when !use_atomic_insns AND op is Swap/Add/Xor/And/Or.
    // All min/max ops fall through to the section below regardless.
    let is_direct_nonatomic = !atomic.use_atomic_insns
        && matches!(
            op,
            RmwOp::Swap | RmwOp::Add | RmwOp::Xor | RmwOp::And | RmwOp::Or
        );

    if is_direct_nonatomic {
        // Entry stack: [addr, src] — consumed here; values also in locals.
        rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop src (reload from local)
        rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop addr (reload from local)

        // Load old value.
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?;
        let load_instr = match width {
            RmwWidth::W32 => Instruction::I32Load(m),
            RmwWidth::W64 => Instruction::I64Load(m),
        };
        rctx.flush_for_load(ctx, addr_local, ValType::I32, tail_idx)?;
        rctx.feed(ctx, tail_idx, &load_instr)?; // old
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(scratch_local))?; // stash old; old on stack

        if op == RmwOp::Swap {
            // new = src; drop old from compute stack, keep it as result below.
            // Stack: old.  We need [addr, src] for the store.
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?; // old, addr
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, src
        } else {
            // Compute new = op(old, src).  Stack: old.
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?; // old, addr
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, old
            rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, old, src
            match (width, op) {
                (RmwWidth::W32, RmwOp::Add) => rctx.feed(ctx, tail_idx, &Instruction::I32Add)?,
                (RmwWidth::W64, RmwOp::Add) => rctx.feed(ctx, tail_idx, &Instruction::I64Add)?,
                (RmwWidth::W32, RmwOp::Xor) => rctx.feed(ctx, tail_idx, &Instruction::I32Xor)?,
                (RmwWidth::W64, RmwOp::Xor) => rctx.feed(ctx, tail_idx, &Instruction::I64Xor)?,
                (RmwWidth::W32, RmwOp::And) => rctx.feed(ctx, tail_idx, &Instruction::I32And)?,
                (RmwWidth::W64, RmwOp::And) => rctx.feed(ctx, tail_idx, &Instruction::I64And)?,
                (RmwWidth::W32, RmwOp::Or) => rctx.feed(ctx, tail_idx, &Instruction::I32Or)?,
                (RmwWidth::W64, RmwOp::Or) => rctx.feed(ctx, tail_idx, &Instruction::I64Or)?,
                _ => unreachable!(),
            }
            // Stack: old, addr, new
        }

        // Store new to addr; respects MemOrder for feed_lazy eligibility.
        let store_instr = match width {
            RmwWidth::W32 => Instruction::I32Store(m),
            RmwWidth::W64 => Instruction::I64Store(m),
        };
        emit_store(
            ctx,
            rctx,
            order,
            AtomicOpts::NONE,
            ValType::I32,
            store_instr,
            tail_idx,
        )?;
        // Stack: old
        return Ok(());
    }

    // ── Synthesised min/max ────────────────────────────────────────────────
    //
    // Wasm structured encoding:
    //
    //   ;; entry stack: [addr, src]  (both consumed; reloaded from locals)
    //   block $exit (result valtype)
    //     loop $retry
    //       local.get addr_local
    //       i32/i64.atomic.load          ;; old
    //       local.tee scratch_local      ;; old (stashed)
    //       local.get src_local          ;; old, src
    //       [relational op]              ;; pred
    //       local.get scratch_local      ;; pred, old
    //       local.get src_local          ;; pred, old, src
    //       select                       ;; new = pred ? old : src
    //                                    ;;   = keep old when pred is true
    //       ;; cmpxchg needs [addr, expected, replacement]
    //       local.get addr_local         ;; new, addr  ← wrong order
    //       ;; swap trick: stash new, push addr, push expected, push new
    //       local.set scratch_local      ;; addr (new stashed)  ← but scratch has old!
    //
    // The problem: we need both `old` (as cmpxchg expected) and `new`
    // (as replacement) simultaneously, but we only have one scratch local.
    //
    // Solution: use `scratch_local` for `old` (tee it there), and compute
    // `new` without losing `old`, by ordering the select to put the result
    // (new) on the stack last, then immediately issue cmpxchg.
    //
    // cmpxchg operand order: addr expected replacement → old_at_addr
    // We build:
    //   addr_local.get                ;; addr
    //   scratch_local.get             ;; addr, old   (= expected)
    //   [new computed inline on stack];; addr, old, new   (= replacement)
    //   cmpxchg                       ;; got
    //   ;; check: got == old?
    //   local.get scratch_local       ;; got, old
    //   i32/i64.eq                    ;; same?
    //   br_if 1 ($exit)              ;; exit if CAS succeeded (got==old)
    //   br 0 ($retry)                ;; else retry
    //   end ;; loop
    //   ;; unreachable — loop only exits via br_if
    //   unreachable
    //   end ;; block → old is the block result (from br_if value)
    //
    // The `br_if $exit` with value requires the old value on the stack:
    //   got, old → eq → pred; but we need to carry `old` as block result.
    //   So: tee got into scratch2, compare, br_if carries the tee'd value.
    //   Since we only have one scratch, we instead:
    //     1. tee got (not stash), compare got==old_stash, br_if
    //   But we need `old` as the result, not `got` — and we can't br_if
    //   carrying `got` when we want `old`.
    //
    // Cleanest approach: after CAS, if got==old then `got` IS `old` so
    // we can use `got` as the block result directly.
    //
    //   block $exit (result valtype)
    //     loop $retry
    //       ;; compute new, then cmpxchg
    //       local.get addr_local
    //       scratch = atomic_load(addr)   scratch_local ← old
    //       new = op(scratch, src)         on stack
    //       ;; cmpxchg [addr, expected=scratch, replacement=new]
    //       local.get addr_local           ;; new, addr
    //       local.get scratch_local        ;; new, addr, expected
    //       ;; need [addr, expected, new] but have [new, addr, expected]
    //       ;; Use a second approach: don't put new on stack yet.
    //
    // Final clean sequence (compute new last):
    //
    //   block $exit (result valtype)
    //     loop $retry
    //       ;; load old
    //       local.get addr_local
    //       atomic.load                    ;; old
    //       local.tee scratch_local        ;; old
    //       ;; push addr, expected, then compute new inline as replacement
    //       local.get addr_local           ;; old, addr
    //       local.get scratch_local        ;; old, addr, expected(=old)
    //       ;; compute new = op(old, src) — need old and src
    //       local.get scratch_local        ;; old, addr, expected, old
    //       local.get src_local            ;; old, addr, expected, old, src
    //       [relational]                   ;; old, addr, expected, pred
    //       local.get scratch_local        ;; old, addr, expected, pred, old
    //       local.get src_local            ;; old, addr, expected, pred, old, src
    //       select                         ;; old, addr, expected, new(=replacement)
    //       cmpxchg                        ;; old, got
    //       local.get scratch_local        ;; old, got, old_expected
    //       i32/i64.eq                     ;; old, same?
    //       br_if 1                        ;; if same: br to $exit carrying `old`
    //       drop                           ;; discard `old` from stack (loop again)
    //       br 0                           ;; retry
    //     end ;; loop
    //     unreachable
    //   end ;; block
    //
    // This uses only `scratch_local` (one extra local of the value type).
    // The entry stack [addr, src] is consumed immediately by the loop setup
    // (they are only accessed via locals after that).

    use wasm_encoder::BlockType;

    let val_ty = match width {
        RmwWidth::W32 => wasm_encoder::ValType::I32,
        RmwWidth::W64 => wasm_encoder::ValType::I64,
    };
    let block_ty = BlockType::Result(val_ty);

    if !atomic.use_atomic_insns {
        // ── Non-atomic min/max: plain load / compute / store ──────────────
        //
        // Without wasm atomics the best we can do is a non-atomic
        // load-compute-store.  Correct for single-threaded wasm.
        //
        // Entry stack: [addr, src] — drop both, reload from locals.
        rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop src
        rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop addr

        // old = load(addr)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?;
        let load_instr = match width {
            RmwWidth::W32 => Instruction::I32Load(m),
            RmwWidth::W64 => Instruction::I64Load(m),
        };
        rctx.flush_for_load(ctx, addr_local, ValType::I32, tail_idx)?;
        rctx.feed(ctx, tail_idx, &load_instr)?; // old
        rctx.feed(ctx, tail_idx, &Instruction::LocalTee(scratch_local))?; // stash old; old on stack

        // Compute new = op(old, src) via select.
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?; // old, addr (for store)
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, old
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, old, src
        match (width, op) {
            (RmwWidth::W32, RmwOp::Min) => rctx.feed(ctx, tail_idx, &Instruction::I32LtS)?,
            (RmwWidth::W64, RmwOp::Min) => rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?,
            (RmwWidth::W32, RmwOp::Max) => rctx.feed(ctx, tail_idx, &Instruction::I32GtS)?,
            (RmwWidth::W64, RmwOp::Max) => rctx.feed(ctx, tail_idx, &Instruction::I64GtS)?,
            (RmwWidth::W32, RmwOp::Minu) => rctx.feed(ctx, tail_idx, &Instruction::I32LtU)?,
            (RmwWidth::W64, RmwOp::Minu) => rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?,
            (RmwWidth::W32, RmwOp::Maxu) => rctx.feed(ctx, tail_idx, &Instruction::I32GtU)?,
            (RmwWidth::W64, RmwOp::Maxu) => rctx.feed(ctx, tail_idx, &Instruction::I64GtU)?,
            _ => unreachable!(),
        }
        // stack: old, addr, pred
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, pred, old
        rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, pred, old, src
        rctx.feed(ctx, tail_idx, &Instruction::Select)?; // old, addr, new

        // store new at addr; respects MemOrder.
        let store_instr = match width {
            RmwWidth::W32 => Instruction::I32Store(m),
            RmwWidth::W64 => Instruction::I64Store(m),
        };
        emit_store(
            ctx,
            rctx,
            order,
            AtomicOpts::NONE,
            ValType::I32,
            store_instr,
            tail_idx,
        )?;
        // stack: old
        return Ok(());
    }

    // ── Atomic min/max via cmpxchg retry loop ─────────────────────────────
    //
    // See the long comment above for the full encoding rationale.
    //
    // Consume the entry [addr, src] stack — both values live in locals.
    rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop src from entry stack
    rctx.feed(ctx, tail_idx, &Instruction::Drop)?; // drop addr from entry stack

    rctx.feed(ctx, tail_idx, &Instruction::Block(block_ty))?;
    rctx.feed(ctx, tail_idx, &Instruction::Loop(BlockType::Empty))?;

    // Load old atomically.
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?;
    let load_instr = match width {
        RmwWidth::W32 => Instruction::I32AtomicLoad(m),
        RmwWidth::W64 => Instruction::I64AtomicLoad(m),
    };
    // Flush any deferred stores that alias the RMW address before reading.
    rctx.flush_for_load(ctx, addr_local, ValType::I32, tail_idx)?;
    rctx.feed(ctx, tail_idx, &load_instr)?;
    rctx.feed(ctx, tail_idx, &Instruction::LocalTee(scratch_local))?; // old stashed; old on stack

    // Push addr, expected for cmpxchg
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(addr_local))?; // old, addr
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, expected

    // Compute new = op(old, src) as replacement
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, expected, old
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, expected, old, src

    // Emit the comparison operator; consumes [old, src], leaves [pred]
    match (width, op) {
        (RmwWidth::W32, RmwOp::Min) => rctx.feed(ctx, tail_idx, &Instruction::I32LtS)?,
        (RmwWidth::W64, RmwOp::Min) => rctx.feed(ctx, tail_idx, &Instruction::I64LtS)?,
        (RmwWidth::W32, RmwOp::Max) => rctx.feed(ctx, tail_idx, &Instruction::I32GtS)?,
        (RmwWidth::W64, RmwOp::Max) => rctx.feed(ctx, tail_idx, &Instruction::I64GtS)?,
        (RmwWidth::W32, RmwOp::Minu) => rctx.feed(ctx, tail_idx, &Instruction::I32LtU)?,
        (RmwWidth::W64, RmwOp::Minu) => rctx.feed(ctx, tail_idx, &Instruction::I64LtU)?,
        (RmwWidth::W32, RmwOp::Maxu) => rctx.feed(ctx, tail_idx, &Instruction::I32GtU)?,
        (RmwWidth::W64, RmwOp::Maxu) => rctx.feed(ctx, tail_idx, &Instruction::I64GtU)?,
        _ => unreachable!(),
    }
    // stack: old, addr, expected, pred
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, addr, expected, pred, old
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(src_local))?; // old, addr, expected, pred, old, src
    // select: pred ? old : src  (pred true → keep old = it IS the min/max)
    rctx.feed(ctx, tail_idx, &Instruction::Select)?; // old, addr, expected, new

    // cmpxchg: [addr, expected, replacement] → got
    let cmpxchg_instr = match width {
        RmwWidth::W32 => Instruction::I32AtomicRmwCmpxchg(m),
        RmwWidth::W64 => Instruction::I64AtomicRmwCmpxchg(m),
    };
    rctx.feed(ctx, tail_idx, &cmpxchg_instr)?; // old, got

    // Check CAS result: got == old?
    rctx.feed(ctx, tail_idx, &Instruction::LocalGet(scratch_local))?; // old, got, expected
    let eq_instr = match width {
        RmwWidth::W32 => Instruction::I32Eq,
        RmwWidth::W64 => Instruction::I64Eq,
    };
    rctx.feed(ctx, tail_idx, &eq_instr)?; // old, same?

    // br_if 1 ($exit): if CAS succeeded, break carrying `old` (depth 1 = block)
    rctx.feed(ctx, tail_idx, &Instruction::BrIf(1))?;

    // CAS failed: drop stale `old`, retry
    rctx.feed(ctx, tail_idx, &Instruction::Drop)?;
    rctx.feed(ctx, tail_idx, &Instruction::Br(0))?;

    rctx.feed(ctx, tail_idx, &Instruction::End)?; // end loop
    rctx.feed(ctx, tail_idx, &Instruction::Unreachable)?; // unreachable: loop exits only via br_if
    rctx.feed(ctx, tail_idx, &Instruction::End)?; // end block; result = old

    Ok(())
}

/// Emit a **fence** (memory barrier).
///
/// * Under [`MemOrder::Strong`]: calls [`Reactor::barrier`] to flush any
///   pending lazy stores that may have been queued.  In practice the strong
///   path never queues anything, so this is always a no-op with zero
///   overhead.
/// * Under [`MemOrder::Relaxed`]: calls [`Reactor::barrier`] to commit all
///   deferred stores before any subsequent instruction, faithfully
///   implementing the guest `FENCE`/`SYNC` semantics.
///
/// This function does **not** emit a wasm `AtomicFence` instruction.  Plain
/// wasm memory instructions are already sequentially ordered within a single
/// thread; the fence is only needed to control the *lazy-deferral* buffer
/// maintained by yecta.
pub fn emit_fence<Context, E, RC: ReactorContext<Context, E> + ?Sized>(
    ctx: &mut Context,
    rctx: &mut RC,
    _order: MemOrder,
    tail_idx: usize,
) -> Result<(), E> {
    // Both Strong and Relaxed flush the bundle buffer.  Under Strong this is
    // always a no-op (empty buffer); under Relaxed it commits pending stores.
    rctx.flush_bundles(ctx, tail_idx)
}

// ── MemorySink ────────────────────────────────────────────────────────────────

/// Object-safe sink for memory load/store emission.
///
/// Abstracts over two concrete implementations:
///
/// * [`ReactorMemorySink`] — delegates to the yecta reactor with lazy-store
///   deferral and alias-check logic baked in; used by native recompilers.
/// * [`EagerMemorySink`] — forwards every instruction directly to an
///   underlying [`InstructionSink`] without deferral; used by `speet-wasm`.
///
/// The `tail_idx`, `order`, and `atomic` parameters that the free-function
/// API requires are baked into the concrete impl at construction time so that
/// callers (mappers, `DirectMemory`) need not carry them.
pub trait MemorySink<Context, E>: InstructionSink<Context, E> {
    /// Emit a store instruction.
    ///
    /// The stack at the call site must have `[mapped_address, value]` on top.
    /// `addr_type` is the wasm type of the address operand (`I32` for
    /// ordinary memory, `I64` for memory64).
    ///
    /// Integer stores may be deferred by the underlying reactor; float stores
    /// are always emitted eagerly.
    fn feed_store(
        &mut self,
        ctx: &mut Context,
        addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E>;

    /// Emit a load instruction, flushing any deferred stores that alias
    /// `addr_local` first.
    ///
    /// The stack at the call site must have the load address on top (already
    /// tee'd into `addr_local`).  `addr_type` is the wasm type of the address.
    fn feed_load(
        &mut self,
        ctx: &mut Context,
        addr_local: u32,
        addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E>;

    /// Flush all deferred stores (equivalent to a `FENCE`/`SYNC`).
    fn flush_all(&mut self, ctx: &mut Context) -> Result<(), E>;
}

// ── ReactorMemorySink ─────────────────────────────────────────────────────────

/// [`MemorySink`] backed by a [`ReactorContext`].
///
/// Bakes in `tail_idx`, `order`, and `atomic` so that callers only supply the
/// per-instruction arguments.
pub struct ReactorMemorySink<'a, RC: ?Sized> {
    /// The underlying reactor context.
    pub rctx: &'a mut RC,
    /// The function-local tail index forwarded to all reactor calls.
    pub tail_idx: usize,
    /// Controls eager vs. lazy store emission.
    pub order: MemOrder,
    /// Controls plain vs. atomic instruction substitution.
    pub atomic: AtomicOpts,
}

impl<'a, RC: ?Sized> ReactorMemorySink<'a, RC> {
    /// Construct a `ReactorMemorySink`.
    #[inline]
    pub fn new(rctx: &'a mut RC, tail_idx: usize, order: MemOrder, atomic: AtomicOpts) -> Self {
        Self { rctx, tail_idx, order, atomic }
    }
}

impl<Context, E, RC: ReactorContext<Context, E> + ?Sized> InstructionSink<Context, E>
    for ReactorMemorySink<'_, RC>
{
    #[inline]
    fn instruction(
        &mut self,
        ctx: &mut Context,
        instruction: &Instruction<'_>,
    ) -> Result<(), E> {
        self.rctx.feed(ctx, self.tail_idx, instruction)
    }
}

impl<Context, E, RC: ReactorContext<Context, E> + ?Sized> MemorySink<Context, E>
    for ReactorMemorySink<'_, RC>
{
    #[inline]
    fn feed_store(
        &mut self,
        ctx: &mut Context,
        addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        emit_store(ctx, self.rctx, self.order, self.atomic, addr_type, instr, self.tail_idx)
    }

    #[inline]
    fn feed_load(
        &mut self,
        ctx: &mut Context,
        addr_local: u32,
        addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        emit_load(ctx, self.rctx, addr_local, addr_type, self.atomic, instr, self.tail_idx)
    }

    #[inline]
    fn flush_all(&mut self, ctx: &mut Context) -> Result<(), E> {
        emit_fence(ctx, self.rctx, self.order, self.tail_idx)
    }
}

// ── EagerMemorySink ───────────────────────────────────────────────────────────

/// [`MemorySink`] backed by any [`InstructionSink`].
///
/// Every load and store is emitted immediately — no lazy deferral, no alias
/// checks.  Used by `speet-wasm` where the underlying sink is a plain
/// `InstructionSink` (not a reactor).
pub struct EagerMemorySink<'a, F: ?Sized> {
    /// The underlying instruction sink.
    pub sink: &'a mut F,
}

impl<'a, F: ?Sized> EagerMemorySink<'a, F> {
    /// Construct an `EagerMemorySink` wrapping any `InstructionSink`.
    #[inline]
    pub fn new(sink: &'a mut F) -> Self {
        Self { sink }
    }
}

impl<Context, E, F: InstructionSink<Context, E> + ?Sized> InstructionSink<Context, E>
    for EagerMemorySink<'_, F>
{
    #[inline]
    fn instruction(
        &mut self,
        ctx: &mut Context,
        instruction: &Instruction<'_>,
    ) -> Result<(), E> {
        self.sink.instruction(ctx, instruction)
    }
}

impl<Context, E, F: InstructionSink<Context, E> + ?Sized> MemorySink<Context, E>
    for EagerMemorySink<'_, F>
{
    #[inline]
    fn feed_store(
        &mut self,
        ctx: &mut Context,
        _addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        self.sink.instruction(ctx, &instr)
    }

    #[inline]
    fn feed_load(
        &mut self,
        ctx: &mut Context,
        _addr_local: u32,
        _addr_type: ValType,
        instr: Instruction<'static>,
    ) -> Result<(), E> {
        self.sink.instruction(ctx, &instr)
    }

    #[inline]
    fn flush_all(&mut self, _ctx: &mut Context) -> Result<(), E> {
        Ok(())
    }
}
