//! # speet-link — Multi-binary linking system
//!
//! `speet-link` provides the infrastructure to translate multiple guest
//! binaries (potentially from different architectures) into a single merged
//! WASM module.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │  Linker<Context, E, F, P, Plugin>            │
//! │  ├── reactor: Reactor<Context, E, F, P>      │
//! │  ├── traps:   TrapConfig<…>                  │
//! │  ├── layout:  LocalLayout                    │
//! │  └── plugin:  impl LinkerPlugin<F>           │
//! └──────────────────┬───────────────────────────┘
//!                    │ impl ReactorContext<Context, E>
//!                    ▼
//!         ┌──────────────────────┐
//!         │  Recompile<C, E, F>  │  (arch recompiler trait)
//!         │  reset_for_next_binary(ctx, args)
//!         │  drain_unit(ctx, entry_points) → BinaryUnit<F>
//!         └──────────────────────┘
//!
//! LinkerPlugin::on_unit(BinaryUnit<F>)
//!         ▼
//!  speet_module_builder::MegabinaryBuilder<F>
//!  ├── types:             Vec<FuncType>  → TypeSection
//!  ├── func_type_indices: Vec<u32>       → FunctionSection
//!  ├── fns:               Vec<F>         → CodeSection
//!  └── exports:           Vec<(str, u32)> → ExportSection
//! ```
//!
//! ## Key types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`unit::FuncType`] | WASM function signature (byte-encoded, `Ord + Hash`) |
//! | [`unit::BinaryUnit`] | Functions + types + exports for one translated binary |
//! | [`context::ReactorContext`] | Unified interface a recompiler borrows |
//! | [`recompiler::Recompile`] | Trait an arch recompiler implements |
//! | [`linker::Linker`] | Owns reactor + traps; implements `ReactorContext` |
//! | [`linker::LinkerPlugin`] | Callback per committed `BinaryUnit` |
//! | [`shim::ShimSpec`] / [`shim::emit_shim`] | ABI shim emitter |
//!
//! Module-level accumulation ([`MegabinaryBuilder`], [`MegabinaryOutput`],
//! [`ElementsOwned`]) lives in `speet-module-builder`.
//!
//! ## End-to-end example (two-pass)
//!
//! ```ignore
//! use speet_link::linker::Linker;
//! use speet_link::schedule::FuncSchedule;
//! use speet_module_builder::{MegabinaryBuilder, assemble};
//!
//! let mut linker = Linker::with_plugin(MegabinaryBuilder::new());
//!
//! // --- Registration phase ---
//! let mut schedule = FuncSchedule::new();
//!
//! let n_wasm   = WasmFrontend::parse_fn_count(&wasm_bytes)?;
//! let n_native = rc.count_fns(&native_bytes);
//!
//! let wasm_slot   = schedule.push(n_wasm,   |ctx_rc, ctx| { /* translate WASM   */ });
//! let native_slot = schedule.push(n_native, |ctx_rc, ctx| { /* translate native */ });
//!
//! // Layout is final — read absolute indices before any translation.
//! let offsets = IndexOffsets { func: schedule.layout().base(native_slot), .. };
//!
//! // --- Emit phase ---
//! schedule.execute(&mut linker, &mut ctx);
//!
//! // Assemble the final module.
//! let module = assemble(linker.plugin.finish());
//! ```

#![no_std]

extern crate alloc;

pub mod context;
pub mod layout;
pub mod linker;
pub mod recompiler;
pub mod schedule;
pub mod shim;
pub mod unit;

#[cfg(test)]
mod tests;

// Flat re-exports for the most commonly used items.
pub use context::{BaseContext, ReactorContext};
pub use layout::{FuncLayout, FuncSlot};
pub use linker::{Linker, LinkerPlugin};
pub use recompiler::Recompile;
pub use schedule::FuncSchedule;
pub use shim::{MemWidth, ParamSource, Place, SavePair, ShimSpec, emit_shim};
pub use unit::{BinaryUnit, DataSegment, FuncType};
