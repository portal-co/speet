//! # speet-memory
//!
//! Architecture-agnostic memory and local-variable abstractions for speet
//! recompilers.
//!
//! ## What this crate provides
//!
//! ### [`LocalLayout`]
//! A runtime table that maps named local-variable *slots* (specified as
//! `(count, ValType)` groups) to contiguous wasm local indices.  Each
//! recompiler calls [`LocalLayout::build`] once to produce a frozen layout
//! object, then queries [`LocalLayout::base`] to locate groups of locals.
//! An owned `LocalLayout` also implements
//! `Iterator<Item = (u32, ValType)>` so it can be fed directly to
//! `reactor.next_with(ctx, f(&mut layout.iter()), len)`.
//!
//! ### [`CallbackContext`] / [`MapperCallback`]
//! A generic wrapper around any `InstructionSink` that lets callbacks emit
//! wasm instructions without knowing the concrete sink type.
//! `MapperCallback` is the trait for virtual-to-physical address translation
//! hooks.  Blanket impls are provided for `FnMut` closures.
//!
//! ### [`MemoryEmitter`]
//! A helper that encodes the full address-computation + optional mapper call
//! + load/store pattern into a pair of methods:
//! [`MemoryEmitter::emit_load`] and [`MemoryEmitter::emit_store`].
//! Callers supply the computed guest address (already on the wasm stack)
//! and an `AddressWidth` flag; the emitter handles the width-dependent
//! `I32Add`/`I64Add`, optional `I32WrapI64`, the mapper invocation, and
//! finally the appropriately-typed wasm memory instruction.
//!
//! ### Page-table helpers
//! [`standard_page_table_mapper`], [`standard_page_table_mapper_32`],
//! [`multilevel_page_table_mapper`], [`multilevel_page_table_mapper_32`],
//! and [`PageTableBase`] are the page-table code-generation helpers
//! previously duplicated in `speet-riscv`.

#![no_std]
extern crate alloc;

pub mod layout;
pub mod mapper;
pub mod mem;
pub mod paging;

pub use layout::{LocalLayout, LocalSlot};
pub use mapper::{CallbackContext, MapperCallback};
pub use mem::{AddressWidth, IntWidth, LoadKind, MemoryEmitter, StoreKind};
pub use paging::{
    PageMapLocals, PageTableBase,
    multilevel_page_table_mapper, multilevel_page_table_mapper_32,
    standard_page_table_mapper, standard_page_table_mapper_32,
};
