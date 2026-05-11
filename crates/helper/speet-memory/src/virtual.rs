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
//! Phase 2 via [`VirtualMemory::memory_idx`] and
//! [`VirtualMemory::page_table_base_deferred`].
//!
//! ## Example — global base
//!
//! ```ignore
//! // Phase 1 — reserve indices.
//! let vm = VirtualMemory::register(&mut entity_space, BaseKind::Global);
//!
//! // Phase 2 — construct mapper with resolved indices.
//! let sec_base = PageTableBase::Constant(0x2000_0000);
//! let mut mem = vm.standard_mapper(
//!     &entity_space, sec_base, true,
//!     AddressWidth::W64 { memory64: false }, IntWidth::I64,
//! );
//! mem.declare_locals(&mut rctx.layout_mut());
//! recompiler.set_memory_access(Box::new(mem));
//!
//! // Phase 2 — declare the memory and optional global in MegabinaryBuilder.
//! // IMPORTANT: declarations must happen in the same order as register() calls.
//! builder.declare_memory(MemoryType { minimum: 16, ..MemoryType::default() });
//! if vm.global_slot.is_some() {
//!     builder.declare_global(
//!         GlobalType { val_type: ValType::I64, mutable: true, shared: false },
//!         ConstExpr::i64_const(0),
//!     );
//! }
//! ```
//!
//! ## Example — param base
//!
//! ```ignore
//! // Phase 1 — reserve indices (no global needed).
//! let vm = VirtualMemory::register(&mut entity_space, BaseKind::Param);
//!
//! // Phase 2 — construct mapper.  declare_params() replaces PageTableBase::Param.
//! let sec_base = PageTableBase::Constant(0x2000_0000);
//! let mut mem = vm.standard_mapper(
//!     &entity_space, sec_base, true,
//!     AddressWidth::W64 { memory64: false }, IntWidth::I64,
//! );
//! // mem.declare_params() is called by the linker via LocalDeclarator chain.
//! recompiler.set_memory_access(Box::new(mem));
//! ```

use speet_link_core::layout::{EntityIndexSpace, IndexSlot};
use crate::mem::{AddressWidth, IntWidth};
use crate::mapper::DirectMemory;
use crate::paging::{
    PageTableBase, StandardPageTableMapper, StandardPageTableMapper32,
    MultilevelPageTableMapper, MultilevelPageTableMapper32,
    standard_page_table_mapper, standard_page_table_mapper_32,
    multilevel_page_table_mapper, multilevel_page_table_mapper_32,
};

// ── BaseKind ───────────────────────────────────────────────────────────────────

/// Specifies where the page-table base address comes from at runtime.
///
/// Passed to [`VirtualMemory::register`] to control whether a WASM global is
/// reserved, and what [`PageTableBase`] variant is produced in Phase 2.
#[derive(Clone, Copy, Debug)]
pub enum BaseKind {
    /// Compile-time constant; the value is baked directly into instructions.
    Constant(u64),
    /// Runtime value stored in a WASM mutable global (reserved in Phase 1).
    Global,
    /// Runtime value that arrives as a WASM function parameter injected by the
    /// linker's `LocalDeclarator` chain.  [`PageTableBase::Param`] is produced
    /// and replaced with `Local(idx)` when `declare_params` is called.
    Param,
}

// ── VirtualMemory ──────────────────────────────────────────────────────────────

/// Tracks pre-declared WASM memory (and optional global) for one guest virtual
/// address space.
///
/// Created in Phase 1 via [`register`](Self::register); provides resolved
/// indices in Phase 2 via [`memory_idx`](Self::memory_idx),
/// [`page_table_base_deferred`](Self::page_table_base_deferred), and the
/// factory mapper methods.
#[derive(Clone, Copy, Debug)]
pub struct VirtualMemory {
    /// Pre-declared WASM memory slot.
    pub memory_slot: IndexSlot,
    /// Pre-declared global slot for the page-table base address, if any.
    pub global_slot: Option<IndexSlot>,
    /// How the page-table base address is supplied at runtime.
    pub base_kind: BaseKind,
}

impl VirtualMemory {
    /// Register one WASM memory (and, when `base_kind` is `Global`, one mutable
    /// global) in Phase 1.
    ///
    /// * `entity_space` — the linker's pre-declaration index space.
    /// * `base_kind` — controls whether a global is reserved and which
    ///   [`PageTableBase`] variant is produced in Phase 2.
    pub fn register(entity_space: &mut EntityIndexSpace, base_kind: BaseKind) -> Self {
        let memory_slot = entity_space.memories.append(1);
        let global_slot = matches!(base_kind, BaseKind::Global)
            .then(|| entity_space.globals.append(1));
        Self { memory_slot, global_slot, base_kind }
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
    /// * `BaseKind::Constant(c)` → `PageTableBase::Constant(c)`
    /// * `BaseKind::Global`      → `PageTableBase::Global(resolved_idx)`
    /// * `BaseKind::Param`       → `PageTableBase::Param` (sentinel; must be
    ///   resolved via `declare_params` before `emit_load` is called)
    pub fn page_table_base_deferred(&self, entity_space: &EntityIndexSpace) -> PageTableBase {
        match self.base_kind {
            BaseKind::Constant(c) => PageTableBase::Constant(c),
            BaseKind::Global => {
                let idx = self.global_idx(entity_space)
                    .expect("VirtualMemory registered without Global but base_kind is Global");
                PageTableBase::Global(idx)
            }
            BaseKind::Param => PageTableBase::Param,
        }
    }

    // ── Factory methods ────────────────────────────────────────────────────────

    /// Build a [`DirectMemory`]`<`[`StandardPageTableMapper`]`>` with the
    /// resolved memory index and page-table base from this `VirtualMemory`.
    pub fn standard_mapper(
        &self,
        entity_space: &EntityIndexSpace,
        security_directory_base: impl Into<PageTableBase>,
        use_i64: bool,
        addr_width: AddressWidth,
        int_width: IntWidth,
    ) -> DirectMemory<StandardPageTableMapper> {
        let mapper = standard_page_table_mapper(
            self.page_table_base_deferred(entity_space),
            security_directory_base,
            self.memory_idx(entity_space),
            use_i64,
        );
        DirectMemory::new(mapper, self.memory_idx(entity_space), addr_width, int_width)
    }

    /// Build a [`DirectMemory`]`<`[`StandardPageTableMapper32`]`>` with the
    /// resolved memory index and page-table base from this `VirtualMemory`.
    pub fn standard_mapper_32(
        &self,
        entity_space: &EntityIndexSpace,
        security_directory_base: impl Into<PageTableBase>,
        use_i64: bool,
        addr_width: AddressWidth,
        int_width: IntWidth,
    ) -> DirectMemory<StandardPageTableMapper32> {
        let mapper = standard_page_table_mapper_32(
            self.page_table_base_deferred(entity_space),
            security_directory_base,
            self.memory_idx(entity_space),
            use_i64,
        );
        DirectMemory::new(mapper, self.memory_idx(entity_space), addr_width, int_width)
    }

    /// Build a [`DirectMemory`]`<`[`MultilevelPageTableMapper`]`>` with the
    /// resolved memory index and L3-table base from this `VirtualMemory`.
    pub fn multilevel_mapper(
        &self,
        entity_space: &EntityIndexSpace,
        security_directory_base: impl Into<PageTableBase>,
        use_i64: bool,
        addr_width: AddressWidth,
        int_width: IntWidth,
    ) -> DirectMemory<MultilevelPageTableMapper> {
        let mapper = multilevel_page_table_mapper(
            self.page_table_base_deferred(entity_space),
            security_directory_base,
            self.memory_idx(entity_space),
            use_i64,
        );
        DirectMemory::new(mapper, self.memory_idx(entity_space), addr_width, int_width)
    }

    /// Build a [`DirectMemory`]`<`[`MultilevelPageTableMapper32`]`>` with the
    /// resolved memory index and L3-table base from this `VirtualMemory`.
    pub fn multilevel_mapper_32(
        &self,
        entity_space: &EntityIndexSpace,
        security_directory_base: impl Into<PageTableBase>,
        use_i64: bool,
        addr_width: AddressWidth,
        int_width: IntWidth,
    ) -> DirectMemory<MultilevelPageTableMapper32> {
        let mapper = multilevel_page_table_mapper_32(
            self.page_table_base_deferred(entity_space),
            security_directory_base,
            self.memory_idx(entity_space),
            use_i64,
        );
        DirectMemory::new(mapper, self.memory_idx(entity_space), addr_width, int_width)
    }
}
