//! `speet-plugin-adapter` — bridges `speet-plugin-api` plugin traits back
//! into speet's existing, unmodified internal traits. A plugin-backed
//! implementation becomes just another value satisfying the internal
//! trait's bound (e.g. `Box<dyn AddressMapper<Context, E>>`), so every
//! current call site keeps working exactly as today. See
//! `docs/guides/plugin-api.md`.
//!
//! Depends on the internal crates it bridges into, but never on any
//! `speet-plugin-host*` backend crate — it only consumes the trait objects
//! a `PluginRegistry` (or static-mode construction) hands back, regardless
//! of which host produced them.

#![no_std]

extern crate alloc;

pub mod arch;
pub mod memory;
pub mod object_model;
pub mod replay;
pub mod reverse;
pub mod table;
pub mod target;

pub use arch::{ArchPluginBinaryArgs, ArchPluginRecompiler};
pub use memory::{PluginAddressMapper, PluginMemoryAccess};
pub use object_model::PluginObjectModel;
pub use replay::replay_snippet;
pub use reverse::{
    BuiltinAddressMapperAsPlugin, BuiltinMemoryAccessAsPlugin, BuiltinObjectModelAsPlugin,
    BuiltinTableAsPlugin, BuiltinTargetAsPlugin,
};
pub use speet_plugin_api::snippet::CodeSnippet;
pub use table::PluginIndirectJumpHandler;
pub use target::{materialize_syscall_table, PluginModuleTargetDeclarator};
