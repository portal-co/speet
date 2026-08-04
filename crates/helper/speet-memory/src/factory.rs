//! Shared [`MemoryAccess`] factory for native emitters and [`speet_wasm::WasmFrontend`].

use alloc::boxed::Box;
use core::marker::PhantomData;
use crate::mapper::{DirectMemory, HostOffsetMapper, MemoryAccess};
use crate::mem::{AddressWidth, IntWidth};
use crate::paging::PageTableBase;
use speet_link_core::image_layout::MemoryModel;
use speet_link_core::ParamSlotMap;
use speet_ordering::MemorySink;
use yecta::{CellIdx, LocalDeclarator, LocalLayout};

/// Default guest linear memory index (memory64).
pub const GUEST_MEMORY_INDEX: u32 = 0;

/// Guest memory access selected from [`GuestImageLayout::memory_model`](speet_link_core::GuestImageLayout::memory_model).
pub enum LayoutMemoryAccess<Context, E> {
    Linear(DirectMemory<()>, PhantomData<(Context, E)>),
    HostOffset(DirectMemory<HostOffsetMapper>, PhantomData<(Context, E)>),
}

impl<Context, E> LocalDeclarator for LayoutMemoryAccess<Context, E> {
    fn declare_params(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        match self {
            Self::Linear(m, _) => m.declare_params(cell, layout),
            Self::HostOffset(m, _) => m.declare_params(cell, layout),
        }
    }

    fn declare_locals(&mut self, cell: CellIdx, layout: &mut LocalLayout) {
        match self {
            Self::Linear(m, _) => m.declare_locals(cell, layout),
            Self::HostOffset(m, _) => m.declare_locals(cell, layout),
        }
    }
}

impl<Context, E> MemoryAccess<Context, E> for LayoutMemoryAccess<Context, E> {
    fn bind_layout_slots(&mut self, layout: &LocalLayout, slots: &ParamSlotMap) {
        if let Self::HostOffset(m, _) = self {
            m.mapper.bind_host_mem_base(layout, slots.host_mem_base);
        }
    }

    fn data_memory_index(&self) -> Option<u32> {
        match self {
            Self::Linear(m, _) => Some(m.data_memory_index),
            Self::HostOffset(m, _) => Some(m.data_memory_index),
        }
    }

    fn transforms_address(&self) -> bool {
        match self {
            Self::Linear(_, _) => false,
            Self::HostOffset(_, _) => true,
        }
    }

    fn emit_load(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: crate::mem::LoadKind,
    ) -> Result<(), E> {
        match self {
            Self::Linear(m, _) => m.emit_load(ctx, sink, kind),
            Self::HostOffset(m, _) => m.emit_load(ctx, sink, kind),
        }
    }

    fn emit_store_addr(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
    ) -> Result<(), E> {
        match self {
            Self::Linear(m, _) => m.emit_store_addr(ctx, sink),
            Self::HostOffset(m, _) => m.emit_store_addr(ctx, sink),
        }
    }

    fn emit_store_insn(
        &mut self,
        ctx: &mut Context,
        sink: &mut dyn MemorySink<Context, E>,
        kind: crate::mem::StoreKind,
    ) -> Result<(), E> {
        match self {
            Self::Linear(m, _) => m.emit_store_insn(ctx, sink, kind),
            Self::HostOffset(m, _) => m.emit_store_insn(ctx, sink, kind),
        }
    }
}

/// Build guest memory access for the given [`MemoryModel`].
pub fn memory_access_for_model<Context, E>(
    model: MemoryModel,
) -> Box<dyn MemoryAccess<Context, E>>
where
    Context: 'static,
    E: 'static,
{
    match model {
        // ZeroOffset shares identity address math with OwnedLinear; text-hole
        // policy is enforced at data-init / runtime, not in the mapper.
        MemoryModel::OwnedLinear | MemoryModel::ZeroOffset => {
            Box::new(LayoutMemoryAccess::<Context, E>::Linear(
                DirectMemory::new(
                    (),
                    GUEST_MEMORY_INDEX,
                    AddressWidth::W64 { memory64: true },
                    IntWidth::I64,
                ),
                PhantomData,
            ))
        }
        MemoryModel::HostOffset => Box::new(LayoutMemoryAccess::<Context, E>::HostOffset(
            DirectMemory::new(
                HostOffsetMapper::new(PageTableBase::Param, true),
                GUEST_MEMORY_INDEX,
                AddressWidth::W64 { memory64: true },
                IntWidth::I64,
            ),
            PhantomData,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_link_core::RuntimeLayoutParams;

    #[test]
    fn host_offset_binds_after_params() {
        let mut layout = LocalLayout::empty();
        let mut params = RuntimeLayoutParams::new();
        params.declare_params(CellIdx(0), &mut layout);
        let mut mem =
            memory_access_for_model::<(), core::convert::Infallible>(MemoryModel::HostOffset);
        mem.bind_layout_slots(&layout, &params.slots);
    }

    #[test]
    fn zero_offset_is_identity_mapper() {
        let mem =
            memory_access_for_model::<(), core::convert::Infallible>(MemoryModel::ZeroOffset);
        assert!(!mem.transforms_address());
        assert_eq!(mem.data_memory_index(), Some(GUEST_MEMORY_INDEX));
    }
}
