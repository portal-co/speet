//! # speet-reach
//!
//! Standardized, shared instruction skip system for speet recompilers.
//!
//! This crate enables external tools (profilers, static analysers, disassemblers)
//! to supply a set of *seed* instruction PCs and have speet automatically expand
//! that set transitively by following jumps, calls, fallthroughs, and branches.
//! Instructions outside the resulting [`ReachableSet`] are then skipped during
//! translation, emitting compact `unreachable` stubs in their place.
//!
//! ## Two-step usage
//!
//! ```ignore
//! use speet_reach::{ReachabilitySpec, compute_reachable, ReachabilityFilter};
//! use speet_riscv::cfg::RiscVCfgDecoder;
//! use rv_asm::Xlen;
//!
//! // Step 1 — build the transitive reachable set.
//! let spec = ReachabilitySpec::single(entry_pc);
//! let decoder = RiscVCfgDecoder { xlen: Xlen::Rv32 };
//! let reachable = compute_reachable(&spec, &binary_bytes, base_addr, &decoder);
//!
//! // Step 2 — install the filter and translate.
//! recompiler.set_instruction_trap(ReachabilityFilter::new(reachable));
//! recompiler.translate_bytes(&mut ctx, &binary_bytes, base_addr, &mut factory)?;
//! ```
//!
//! ## Optionality
//!
//! When no [`ReachabilityFilter`] is installed, the recompiler behaves exactly
//! as before.  The feature is purely opt-in.
//!
//! ## Transitivity
//!
//! [`compute_reachable`] performs a BFS over the CFG.  Any instruction reachable
//! by following a chain of jumps, calls, conditional branches, or sequential
//! fallthroughs from a seed is included — even if not explicitly listed in
//! [`ReachabilitySpec::seeds`].
//!
//! Indirect jump targets (runtime-computed) cannot be followed statically.
//! External tools should add known indirect targets to `seeds` directly.
//!
//! ## `no_std`
//!
//! This crate is `no_std` with `extern crate alloc`.

#![no_std]

extern crate alloc;

mod compute;
mod edges;
mod filter;
mod spec;

pub use compute::{ReachableSet, compute_reachable};
pub use edges::{CfgDecoder, CfgEdges};
pub use filter::ReachabilityFilter;
pub use spec::ReachabilitySpec;
