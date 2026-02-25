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

#![no_std]

use wasm_encoder::{Instruction, MemArg};
use wax_core::build::InstructionSink;
use yecta::Reactor;

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
    pub const NONE: Self = Self { use_atomic_insns: false };
    /// Atomic instructions enabled.
    pub const ATOMIC: Self = Self { use_atomic_insns: true };
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

// ── Public API ────────────────────────────────────────────────────────────────

/// Emit a memory **store** instruction.
///
/// * When `order` is [`MemOrder::Strong`]: the instruction is emitted
///   eagerly via [`Reactor::feed`].
/// * When `order` is [`MemOrder::Relaxed`]: the instruction is queued with
///   [`Reactor::feed_lazy`] so the reactor can defer it until the next
///   control-flow boundary.
///
/// If `atomic.use_atomic_insns` is `true` and the instruction has an atomic
/// counterpart, the atomic form is used instead.
///
/// The `instr` argument must be a `'static` instruction (i.e. one that does
/// not borrow a name or byte string) because `feed_lazy` requires
/// `'static`.  All integer store instructions satisfy this.
pub fn emit_store<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    reactor: &mut Reactor<Context, E, F>,
    order: MemOrder,
    atomic: AtomicOpts,
    instr: Instruction<'static>,
) -> Result<(), E> {
    let instr = if atomic.use_atomic_insns {
        atomic_store_equiv(&instr)
    } else {
        instr
    };
    match order {
        MemOrder::Strong => reactor.feed(ctx, &instr),
        MemOrder::Relaxed => reactor.feed_lazy(ctx, &instr),
    }
}

/// Emit a memory **load** instruction.
///
/// Loads are always emitted eagerly regardless of `MemOrder`, because
/// reordering a load past a subsequent store would violate load-acquire
/// semantics even on a weak ISA.
///
/// If `atomic.use_atomic_insns` is `true` and the instruction has an atomic
/// counterpart, the atomic form is used instead.
pub fn emit_load<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    reactor: &mut Reactor<Context, E, F>,
    atomic: AtomicOpts,
    instr: Instruction<'static>,
) -> Result<(), E> {
    let instr = if atomic.use_atomic_insns {
        atomic_load_equiv(&instr)
    } else {
        instr
    };
    reactor.feed(ctx, &instr)
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
pub fn emit_fence<Context, E, F: InstructionSink<Context, E>>(
    ctx: &mut Context,
    reactor: &mut Reactor<Context, E, F>,
    _order: MemOrder,
) -> Result<(), E> {
    // Both Strong and Relaxed flush the bundle buffer.  Under Strong this is
    // always a no-op (empty buffer); under Relaxed it commits pending stores.
    reactor.barrier(ctx)
}
