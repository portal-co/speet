//! # speet-traps
//!
//! Pluggable trap hooks for speet architecture recompilers.
//!
//! This crate defines two orthogonal extension points that can be installed
//! into any architecture recompiler:
//!
//! ## 1. Instruction traps — [`InstructionTrap`]
//!
//! Fire once per translated guest instruction, before the instruction body
//! is emitted.  Use cases:
//!
//! - **Context switching**: emit a check before every instruction and jump to
//!   a scheduler thunk if a preemption flag is set.
//! - **Static debugging**: insert breakpoint hooks, per-instruction counters,
//!   or coverage bitmaps.
//! - **Instruction filtering**: suppress entire classes of instructions by
//!   returning [`TrapAction::Skip`].
//!
//! ## 2. Jump traps — [`JumpTrap`]
//!
//! Fire before every control-flow transfer (branch, call, return, indirect
//! jump).  Use cases:
//!
//! - **ROP detection**: validate that indirect-jump targets are in a
//!   pre-approved set ([`impls::RopDetectTrap`]).
//! - **CFI**: check call/return balance with a shadow-stack counter
//!   ([`impls::CfiReturnTrap`]).
//! - **Control-flow tracing**: log every transfer to a wasm import
//!   ([`impls::TraceLogTrap`]).
//! - **Dynamic binary translation guards**: prevent jumps to translated code
//!   that hasn't been validated yet.
//!
//! ## Usage
//!
//! A recompiler embeds a [`TrapConfig`] and calls:
//!
//! - `config.extra_locals_iter()` in `init_function` to chain trap locals
//!   into the function's local declaration.
//! - `config.set_extra_locals_base(arch_count)` after counting architecture
//!   locals.
//! - `config.on_instruction(&info, ctx, &mut self.reactor)` at the start of
//!   `translate_instruction`.
//! - `config.on_jump(&info, ctx, &mut self.reactor)` before each jump site.
//!
//! ## `TrapAction::Skip` and skip snippets
//!
//! When a trap returns [`TrapAction::Skip`], the recompiler suppresses the
//! normal instruction body (or jump) and instead calls the trap's
//! `skip_snippet`.  By default `skip_snippet` emits `unreachable`; override
//! it to redirect to a violation handler:
//!
//! ```ignore
//! impl JumpTrap<…> for MyTrap {
//!     fn on_jump(&mut self, info: &JumpInfo, ctx, trap_ctx) -> Result<TrapAction, E> {
//!         // … check condition …
//!         Ok(TrapAction::Skip)
//!     }
//!     fn skip_snippet(&self, info: &JumpInfo, ctx, skip_ctx) -> Result<(), E> {
//!         // Redirect to a violation handler wasm function.
//!         skip_ctx.jump(ctx, HANDLER_FUNC_IDX, NUM_PARAMS)
//!     }
//! }
//! ```
//!
//! ## Composition
//!
//! Multiple traps of the same kind can be composed using:
//!
//! - [`impls::ChainedTrap<A, B>`] — a static, zero-cost pair (A runs first).
//! - `Vec<Box<dyn InstructionTrap<…>>>` / `Vec<Box<dyn JumpTrap<…>>>` — a
//!   dynamic list; each element's `on_*` is called in order until a `Skip`.
//!
//! ## `no_std`
//!
//! This crate is `no_std` with `extern crate alloc` for `Vec`, `Box`, and
//! `String`-free allocations.

#![no_std]

extern crate alloc;

pub mod config;
pub mod context;
pub mod impls;
pub mod insn;
pub mod jump;

// Flat re-exports for the most commonly used items.
pub use config::TrapConfig;
pub use context::{TrapContext, reactor_jump, reactor_jump_if};
pub use impls::{
    CfiReturnTrap, ChainedTrap, CounterTrap, NullTrap, RopDetectTrap, TraceLogTrap,
};
pub use insn::{ArchTag, InsnClass, InstructionInfo, InstructionTrap, TrapAction};
pub use jump::{JumpInfo, JumpKind, JumpTrap};
pub use yecta::{LocalLayout, LocalSlot};
