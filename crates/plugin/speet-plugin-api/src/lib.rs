//! `speet-plugin-api` — the Context/E-erased plugin-facing trait layer for
//! speet's external plugin system.
//!
//! Five resource-kind plugin traits mirror speet's five internal extension
//! points one-for-one ([`arch::ArchPlugin`], [`memory::AddressMapperPlugin`] /
//! [`memory::MemoryAccessPlugin`], [`table::TablePlugin`],
//! [`object_model::ObjectModelPlugin`], [`target::TargetPlugin`]), so that an
//! external (including non-Rust, including proprietary) implementation can
//! be adapted back into speet's *unchanged* internal traits by
//! `speet-plugin-adapter`.
//!
//! **Core invariant: plugin traits never see `Context` or `E`.** Every
//! method either takes/returns plain data, or returns a
//! [`snippet::CodeSnippet`] (encoded WASM instruction bytes) that the
//! host-side adapter re-parses and forwards through the real
//! `dyn InstructionSink<Context, E>`. Do not add a `Context`/`E`/`F` generic
//! parameter to anything in this crate — see `docs/guides/plugin-api.md`.
//!
//! This crate is `#![no_std]` + `alloc`-only and depends on nothing but
//! `wasm-encoder` (for [`snippet::CodeSnippet`]'s convenience builder), so a
//! `.wasm`-guest plugin author writing in Rust can target
//! `wasm32-unknown-unknown` without pulling in `std`, and so a non-Rust
//! subprocess plugin author's documentation burden is "read this crate's
//! docs," not "understand the rest of the speet workspace."

#![no_std]

extern crate alloc;

pub mod arch;
pub mod error;
pub mod external_target;
pub mod imports;
pub mod memory;
pub mod object_model;
pub mod remote;
pub mod snippet;
pub mod table;
pub mod target;
pub mod wire;

pub use arch::ArchPlugin;
pub use error::{PResult, PluginError};
pub use external_target::{
    CallingConvention, ExternalTargetEntry, ExternalTargetPlugin, ExternalTargetTable,
    LibraryId, PltHook, PltHookTable, PltHookTarget,
};
pub use imports::HostImports;
pub use memory::{AddressMapperPlugin, MemoryAccessPlugin};
pub use object_model::ObjectModelPlugin;
pub use snippet::CodeSnippet;
pub use table::TablePlugin;
pub use target::TargetPlugin;

/// Which of the five resource kinds a plugin implements. Closed for v1 —
/// adding a sixth kind is a documented version bump (see
/// `docs/guides/plugin-api.md`), not an open string tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluginKind {
    Arch,
    Memory,
    Table,
    ObjectModel,
    Target,
}

impl PluginKind {
    /// Parse the manifest's `kind = ...` value (see `speet-plugin-host`'s
    /// manifest format).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "arch" => Some(PluginKind::Arch),
            "memory" => Some(PluginKind::Memory),
            "table" => Some(PluginKind::Table),
            "object-model" => Some(PluginKind::ObjectModel),
            "target" => Some(PluginKind::Target),
            _ => None,
        }
    }
}
