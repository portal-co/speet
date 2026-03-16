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
//!  MegabinaryBuilder<F>
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
//! | [`builder::MegabinaryBuilder`] | Accumulates units; deduplicates types |
//! | [`shim::ShimSpec`] / [`shim::emit_shim`] | ABI shim emitter |
//!
//! ## End-to-end example
//!
//! ```ignore
//! use speet_link::linker::Linker;
//! use speet_link::builder::MegabinaryBuilder;
//! use speet_link::shim::{ShimSpec, emit_shim};
//! use speet_link::unit::{BinaryUnit, FuncType};
//! use wasm_encoder::ValType;
//!
//! // Create a linker with the accumulator plugin.
//! let mut linker = Linker::with_plugin(MegabinaryBuilder::new());
//!
//! // Install shared traps (optional).
//! // linker.traps.set_instruction_trap(&mut my_trap);
//!
//! // Binary 1: x86-64
//! let mut x86 = X86Recompiler::new_with_base_rip(0x400000);
//! x86.setup(&mut linker);
//! translate_bytes(&bytes1, &mut x86, &mut linker);
//! linker.commit(&mut x86, vec![("main".into(), linker.base_func_offset())]);
//!
//! // Raw shim: cross-arch bridge
//! let shim_idx = linker.total_fn_count();
//! let shim = emit_shim(&ShimSpec { callee_func_idx: 0, … });
//! linker.commit_raw(BinaryUnit {
//!     fns: vec![shim],
//!     base_func_offset: shim_idx,
//!     entry_points: vec![("bridge".into(), shim_idx)],
//!     func_types: vec![FuncType::from_val_types(&[ValType::I32; 26], &[])],
//! });
//!
//! // Retrieve and assemble the final module.
//! let out = linker.plugin.finish();
//! // out.types             → TypeSection
//! // out.func_type_indices → FunctionSection
//! // out.fns               → CodeSection
//! // out.exports           → ExportSection
//! ```

#![no_std]

extern crate alloc;

pub mod builder;
pub mod context;
pub mod linker;
pub mod recompiler;
pub mod shim;
pub mod unit;

#[cfg(test)]
mod tests;

// Flat re-exports for the most commonly used items.
pub use builder::{MegabinaryBuilder, MegabinaryOutput};
pub use context::{BaseContext, ReactorContext};
pub use linker::{Linker, LinkerPlugin};
pub use recompiler::Recompile;
pub use shim::{MemWidth, ParamSource, Place, SavePair, ShimSpec, emit_shim};
pub use unit::{BinaryUnit, DataSegment, FuncType};
