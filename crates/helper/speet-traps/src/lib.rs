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
//!   pre-approved set ([`hardening::RopDetectTrap`]).
//! - **CFI**: validate return targets against native CFI metadata
//!   ([`security::CfiReturnTrap`]).
//! - **Control-flow tracing**: log every transfer to a wasm import
//!   ([`tracing::TraceLogTrap`]).
//! - **Dynamic binary translation guards**: prevent jumps to translated code
//!   that hasn't been validated yet.
//!
//! ## Categorised implementations
//!
//! Trap implementations are organised into three purpose-built modules:
//!
//! | Module | Kind | Contents |
//! |--------|------|---------|
//! | [`tracing`] | observation | [`CounterTrap`], [`TraceLogTrap`] |
//! | [`security`] | CFI enforcement | [`CfiReturnTrap`] |
//! | [`hardening`] | anti-\*OP | [`RopDetectTrap`] |
//! | [`impls`] | utility | [`NullTrap`], [`ChainedTrap`] |
//!
//! ## Usage
//!
//! A recompiler embeds a [`TrapConfig`] and calls it at three phases:
//!
//! **Phase 1 — setup** (once, after installing traps):
//! ```ignore
//! // Arch recompiler appends arch params to its owned layout, then:
//! self.traps.declare_params(&mut self.layout);
//! self.locals_mark = self.layout.mark();  // mark = total_params
//! self.total_params = self.locals_mark.total_locals;
//! ```
//!
//! **Phase 2 — per function** (in `init_function`):
//! ```ignore
//! self.layout.rewind(&self.locals_mark);
//! // append arch non-param locals (temps, pool, …) …
//! self.traps.declare_locals(&mut self.layout);
//! reactor.next_with(ctx, f(&mut self.layout.iter_since(&self.locals_mark)), depth)?;
//! ```
//!
//! **Phase 3 — firing** (in translate_instruction / jump sites):
//! ```ignore
//! self.traps.on_instruction(&info, ctx, &mut self.reactor, &self.layout)?;
//! self.traps.on_jump(&info, ctx, &mut self.reactor, &self.layout)?;
//! ```
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
//!
//! ## See also
//!
//! - `docs/trap-hooks.md` — full design: two-phase parameter/local protocol,
//!   composition strategies, and rationale for each built-in implementation.
//! - `AGENTS.md` §2, §4 — agent guidance on parameter vs. local choice and
//!   free-function pattern for Reactor jumps.

#![no_std]

extern crate alloc;

pub mod config;
pub mod context;
pub mod hardening;
pub mod impls;
pub mod insn;
pub mod jump;
pub mod security;
pub mod tracing;

// Flat re-exports for the most commonly used items.
pub use config::TrapConfig;
pub use context::TrapContext;
pub use hardening::RopDetectTrap;
pub use impls::{ChainedTrap, NullTrap};
pub use insn::{ArchTag, InsnClass, InstructionInfo, InstructionTrap, TrapAction};
pub use jump::{JumpInfo, JumpKind, JumpTrap};
pub use security::CfiReturnTrap;
pub use tracing::{CounterTrap, TraceLogTrap};
pub use yecta::{ConstPeek, LocalDeclarator, LocalLayout, LocalSlot, Mark};
