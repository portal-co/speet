//! Relocation-aware data linking and GOT patching for [`GuestImageLayout`].

use std::collections::BTreeMap;
use std::vec::Vec;
use speet_link_core::image_layout::{GuestImageLayout, RelocKindTag};
use speet_plugin_api::external_target::LibraryId;

use crate::frontend::DataSegment;
use crate::plt::PltCallPlan;

/// Apply relocations and patch GOT cells with redirect shim guest PCs.
pub fn link_data_segments(
    layout: &GuestImageLayout,
    plt_plan: Option<&PltCallPlan>,
    shim_index_by_symbol: &BTreeMap<String, u32>,
) -> Vec<DataSegment> {
    let mut sections: Vec<Vec<u8>> = layout
        .data_sections
        .iter()
        .map(|s| s.bytes.clone())
        .collect();

    for reloc in &layout.relocs {
        if reloc.section >= sections.len() {
            continue;
        }
        let Some(value) = resolve_reloc_value(reloc, layout, plt_plan, shim_index_by_symbol) else {
            continue;
        };
        let off = reloc.offset as usize;
        if off + 8 <= sections[reloc.section].len() {
            sections[reloc.section][off..off + 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    // Patch PLT GOT slots with shim guest PCs when allocated.
    if let Some(plan) = plt_plan {
        patch_got_from_plan(layout, &mut sections, plan, shim_index_by_symbol);
    }

    layout
        .data_sections
        .iter()
        .zip(sections)
        .filter(|(spec, _)| {
            // ZeroOffset: never place bytes that overlap the unrecompiled text hole.
            if !layout.memory_model.text_unmapped() {
                return true;
            }
            let end = spec.addr.saturating_add(spec.bytes.len() as u64);
            end <= layout.text_base || spec.addr >= layout.text_end()
        })
        .map(|(spec, bytes)| DataSegment {
            addr: spec.addr,
            bytes,
        })
        .collect()
}

fn resolve_reloc_value(
    reloc: &speet_link_core::image_layout::RelocSpec,
    layout: &GuestImageLayout,
    plt_plan: Option<&PltCallPlan>,
    shim_index_by_symbol: &BTreeMap<String, u32>,
) -> Option<u64> {
    match reloc.kind {
        RelocKindTag::Abs64 | RelocKindTag::GotPcRel | RelocKindTag::Plt32 => {}
        RelocKindTag::Other => return None,
    }

    if let Some(sym) = &reloc.target_symbol {
        if let Some(&shim_i) = shim_index_by_symbol.get(sym) {
            return Some(layout.shim_guest_pc(shim_i));
        }
        if let Some(plan) = plt_plan {
            for entry in plan.targets.iter() {
                if entry.label == *sym {
                    if let Some(&idx) = plan.wasm_import_by_addr.get(&(entry.library, entry.address))
                    {
                        let _ = idx;
                        if let Some(&shim_i) = shim_index_by_symbol.get(sym) {
                            return Some(layout.shim_guest_pc(shim_i));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Patch GOT cells for both WASM-import and native-shim redirect addresses.
fn patch_got_from_plan(
    layout: &GuestImageLayout,
    sections: &mut [Vec<u8>],
    plan: &PltCallPlan,
    shim_index_by_symbol: &BTreeMap<String, u32>,
) {
    let mut addrs: BTreeMap<(LibraryId, u64), ()> = BTreeMap::new();
    for key in plan.wasm_import_by_addr.keys() {
        addrs.insert(*key, ());
    }
    for key in plan.native_shim_by_addr.keys() {
        addrs.insert(*key, ());
    }
    for &(library, addr) in addrs.keys() {
        let label = plan.targets.lookup(library, addr).unwrap_or("").to_string();
        if let Some(&shim_i) = shim_index_by_symbol.get(&label) {
            let shim_pc = layout.shim_guest_pc(shim_i);
            patch_got_at_addr(layout, sections, addr, shim_pc);
        }
    }
}

fn patch_got_at_addr(
    layout: &GuestImageLayout,
    sections: &mut [Vec<u8>],
    guest_addr: u64,
    value: u64,
) {
    for (i, spec) in layout.data_sections.iter().enumerate() {
        if guest_addr >= spec.addr && guest_addr < spec.addr + spec.bytes.len() as u64 {
            let off = (guest_addr - spec.addr) as usize;
            if i < sections.len() && off + 8 <= sections[i].len() {
                sections[i][off..off + 8].copy_from_slice(&value.to_le_bytes());
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_link_core::image_layout::{DataSectionSpec, MemoryModel};

    #[test]
    fn patches_abs64_reloc() {
        let layout = GuestImageLayout {
            text_base: 0x1000,
            text_len: 0x100,
            slot_granularity: 1,
            data_sections: vec![DataSectionSpec {
                name: ".data".into(),
                addr: 0x2000,
                bytes: vec![0; 16],
            }],
            relocs: vec![speet_link_core::image_layout::RelocSpec {
                offset: 8,
                section: 0,
                target_symbol: Some("write".into()),
                kind: RelocKindTag::Abs64,
                addend: 0,
            }],
            libraries: vec![],
            memory_model: MemoryModel::OwnedLinear,
        };
        let mut shims = BTreeMap::new();
        shims.insert("write".into(), 0);
        let segs = link_data_segments(&layout, None, &shims);
        let expected = layout.shim_guest_pc(0);
        assert_eq!(&segs[0].bytes[8..16], &expected.to_le_bytes());
    }

    #[test]
    fn zero_offset_drops_sections_overlapping_text_hole() {
        let layout = GuestImageLayout {
            text_base: 0x1000,
            text_len: 0x100,
            slot_granularity: 1,
            data_sections: vec![
                DataSectionSpec {
                    name: ".text_fake".into(),
                    addr: 0x1000,
                    bytes: vec![0x90; 16],
                },
                DataSectionSpec {
                    name: ".rodata".into(),
                    addr: 0x2000,
                    bytes: b"hi".to_vec(),
                },
            ],
            relocs: vec![],
            libraries: vec![],
            memory_model: MemoryModel::ZeroOffset,
        };
        let segs = link_data_segments(&layout, None, &BTreeMap::new());
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].addr, 0x2000);
        assert_eq!(segs[0].bytes, b"hi");
    }
}
