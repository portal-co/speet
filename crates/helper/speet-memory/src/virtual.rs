//! [`VirtualMemory`] — Phase-1 registration for a guest virtual memory space.
//!
//! The mapper system in `speet-memory` translates guest virtual addresses to
//! host WASM memory addresses.  Before the 5f0cc9c module-building unification,
//! mappers received hardcoded WASM memory and global indices that ignored the
//! `EntityIndexSpace` two-phase discipline, making multi-binary linking fragile.
//!
//! `VirtualMemory` is the fix: call [`VirtualMemory::register`] in Phase 1 to
//! reserve one WASM memory (and optionally one mutable global for the page-table
//! base address).  Pass the resolved indices to the mapper constructors in
//! Phase 2 via [`VirtualMemory::memory_idx`] and [`VirtualMemory::page_table_base`].
//!
//! ## Example
//!
//! ```ignore
//! // Phase 1 — reserve indices.
//! let vm = VirtualMemory::register(&mut entity_space, true /* global base */);
//!
//! // Phase 2 — construct mapper with resolved indices.
//! let mem_idx  = vm.memory_idx(&entity_space);
//! let pg_base  = vm.page_table_base(&entity_space, 0x1000_0000);
//! let sec_base = PageTableBase::Constant(0x2000_0000);
//! let mut mapper = standard_page_table_mapper(pg_base, sec_base, mem_idx, true);
//! mapper.declare_locals(&mut rctx.layout_mut());
//! recompiler.set_mapper_callback(&mut mapper);
//!
//! // Phase 2 — declare the memory and optional global in MegabinaryBuilder.
//! // IMPORTANT: declarations must happen in the same order as register() calls.
//! builder.declare_memory(MemoryType { minimum: 16, .. MemoryType::default() });
//! if vm.global_slot.is_some() {
//!     builder.declare_global(GlobalType { val_type: ValType::I64, mutable: true, shared: false },
//!                            ConstExpr::i64_const(0));
//! }
//! ```

use speet_link_core::layout::{EntityIndexSpace, IndexSlot};
use crate::paging::PageTableBase;

/// Tracks pre-declared WASM memory (and optional global) for one guest virtual
/// address space.
///
/// Created in Phase 1 via [`register`](Self::register); provides resolved
/// indices in Phase 2 via [`memory_idx`](Self::memory_idx) and
/// [`page_table_base`](Self::page_table_base).
#[derive(Clone, Copy, Debug)]
pub struct VirtualMemory {
    /// Pre-declared WASM memory slot.
    pub memory_slot: IndexSlot,
    /// Pre-declared global slot for the page-table base address, if any.
    pub global_slot: Option<IndexSlot>,
}

impl VirtualMemory {
    /// Register one WASM memory (and optionally one mutable global) in Phase 1.
    ///
    /// * `entity_space` — the linker's pre-declaration index space.
    /// * `with_global_base` — if `true`, also reserves one mutable `i64`
    ///   global to hold the page-table base address at runtime.  Set to
    ///   `false` when the base address is a compile-time constant.
    pub fn register(entity_space: &mut EntityIndexSpace, with_global_base: bool) -> Self {
        let memory_slot = entity_space.memories.append(1);
        let global_slot = with_global_base.then(|| entity_space.globals.append(1));
        Self { memory_slot, global_slot }
    }

    /// Resolved WASM memory index (Phase 2).
    #[inline]
    pub fn memory_idx(&self, entity_space: &EntityIndexSpace) -> u32 {
        entity_space.memories.base(self.memory_slot)
    }

    /// Resolved WASM global index for the page-table base, if registered.
    #[inline]
    pub fn global_idx(&self, entity_space: &EntityIndexSpace) -> Option<u32> {
        self.global_slot.map(|s| entity_space.globals.base(s))
    }

    /// Build a [`PageTableBase`] for use with the page-table mapper constructors.
    ///
    /// If a global was registered, returns `PageTableBase::Global(idx)`.
    /// Otherwise returns `PageTableBase::Constant(fallback)`.
    pub fn page_table_base(
        &self,
        entity_space: &EntityIndexSpace,
        fallback: u64,
    ) -> PageTableBase {
        match self.global_idx(entity_space) {
            Some(g) => PageTableBase::Global(g),
            None    => PageTableBase::Constant(fallback),
        }
    }
}
