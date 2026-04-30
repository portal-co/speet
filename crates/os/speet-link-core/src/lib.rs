//! # speet-link-core — Multi-binary linking traits and types
//!
//! `speet-link-core` provides the abstract types and traits shared by all
//! components in the multi-binary linking pipeline.  Arch recompilers depend
//! only on this crate; the concrete linker and scheduler live in `speet-linker`
//! and `speet-schedule`.
//!
//! ## Key types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`unit::FuncType`] | WASM function signature (byte-encoded, `Ord + Hash`) |
//! | [`unit::BinaryUnit`] | Functions + types + exports for one translated binary |
//! | [`context::ReactorContext`] | Unified interface a recompiler borrows |
//! | [`recompiler::Recompile`] | Trait an arch recompiler implements |
//! | [`linker::LinkerPlugin`] | Callback per committed `BinaryUnit` |
//! | [`shim::ShimSpec`] / [`shim::emit_shim`] | ABI shim emitter |
//!
//! The concrete `Linker` lives in `speet-linker`.
//! The `FuncSchedule` orchestrator lives in `speet-schedule`.
//! Module-level accumulation (`MegabinaryBuilder`, `MegabinaryOutput`,
//! `ElementsOwned`) lives in `speet-module-builder`.

#![no_std]

extern crate alloc;

pub mod context;
pub mod layout;
pub mod linker;
pub mod recompiler;
pub mod shim;
pub mod unit;

#[cfg(test)]
mod tests;

// Flat re-exports for the most commonly used items.
pub use context::{BaseContext, FedContext, ReactorContext, ReactorAdapter, TrapReactorAdapter};
pub use layout::{FuncLayout, FuncSlot};
pub use linker::LinkerPlugin;
pub use recompiler::Recompile;
pub use shim::{MemWidth, ParamSource, Place, SavePair, ShimSpec, emit_shim};
pub use unit::{BinaryUnit, DataSegment, FuncType};
