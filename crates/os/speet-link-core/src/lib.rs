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
pub mod image_layout;
pub mod layout;
pub mod layout_params;
pub mod linker;
pub mod oob;
pub mod recompiler;
pub mod shim;
pub mod unit;

#[cfg(test)]
mod tests;

// Flat re-exports for the most commonly used items.
pub use context::{BaseContext, FedContext, ReactorContext, ReactorAdapter, TrapReactorAdapter};
pub use image_layout::{
    DataSectionSpec, GuestImageLayout, LibrarySpec, MemoryModel, RelocKindTag, RelocSpec,
    ZERO_OFFSET_MAX_DATA_END,
};
pub use layout::{EntityIndexSpace, FuncLayout, FuncSlot, IndexSlot, IndexSpace};
pub use layout_params::{
    IndirectTableIdxSnippet, ParamSlotMap, RuntimeLayoutParams, TextBaseSnippet, TextBaseSource,
    default_runtime_layout_params,
};
pub use linker::LinkerPlugin;
pub use oob::{JitConfig, OobConfig};
pub use recompiler::Recompile;
pub use shim::{MemWidth, ParamSource, Place, SavePair, ShimSpec, emit_shim};
pub use unit::{BinaryUnit, DataSegment, FuncType};
