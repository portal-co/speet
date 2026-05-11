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
//! ### [`CallbackContext`] / [`MapperCallback`] / [`AddressMapper`]
//! A generic wrapper around any `InstructionSink` that lets callbacks emit
//! wasm instructions without knowing the concrete sink type.
//! `AddressMapper` (aliased as `MapperCallback`) is the trait for
//! virtual-to-physical address translation hooks.
//!
//! ### [`MemoryAccess`] / [`DirectMemory`]
//! `MemoryAccess` is the high-level trait that owns a complete load/store
//! sequence for a single memory region, combining the mapper, memory index,
//! and address/integer width.  `DirectMemory<M>` is the standard
//! implementation backed by any `AddressMapper`.
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
pub mod r#virtual;

pub use layout::{LocalLayout, LocalSlot};
pub use mapper::{AddressMapper, CallbackContext, ChunkedMapper, DirectMemory, MapperCallback, MemoryAccess, StackedMapper};
pub use yecta::LocalDeclarator;
pub use mem::{AddressWidth, IntWidth, LoadKind, StoreKind};
pub use paging::{
    MultilevelPageTableMapper, MultilevelPageTableMapper32, PageMapLocals, PageTableBase,
    StandardPageTableMapper, StandardPageTableMapper32, multilevel_page_table_mapper,
    multilevel_page_table_mapper_32, standard_page_table_mapper, standard_page_table_mapper_32,
};
pub use r#virtual::{BaseKind, VirtualMemory};
