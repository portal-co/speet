//! [`GuestImageLayout`] — shared contract for text translation, data init,
//! GOT patching, catalog, and link.

use alloc::string::String;
use alloc::vec::Vec;

/// How guest virtual addresses map to host WASM linear memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MemoryModel {
    /// Guest pointer = offset into owned `__wasm_mem` (current thin runtime default).
    #[default]
    OwnedLinear,
    /// `physical = guest_virtual + host_mem_base` (see `HostOffsetMapper`).
    HostOffset,
}

/// One initialized guest data section (`.data`, `.rodata`, GOT-ish regions).
#[derive(Clone, Debug)]
pub struct DataSectionSpec {
    pub name: String,
    pub addr: u64,
    pub bytes: Vec<u8>,
}

/// A relocation targeting a guest data or code section.
#[derive(Clone, Debug)]
pub struct RelocSpec {
    /// Byte offset within the target section's `bytes`.
    pub offset: u64,
    /// Index into [`GuestImageLayout::data_sections`].
    pub section: usize,
    /// Symbol name or section-relative target (resolved at link time).
    pub target_symbol: Option<String>,
    /// Relocation kind discriminator (matches `binary_io::RelocKind` when loaded).
    pub kind: RelocKindTag,
    pub addend: i64,
}

/// Portable relocation kind tag (mirrors `binary_io::RelocKind` without the dep).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocKindTag {
    Abs64,
    GotPcRel,
    Plt32,
    Other,
}

/// A dynamic library dependency (`DT_NEEDED` / `LC_LOAD_DYLIB`).
#[derive(Clone, Debug)]
pub struct LibrarySpec {
    pub name: String,
}

/// Single struct consumed by text translation, data init, GOT patching, catalog, and link.
#[derive(Clone, Debug)]
pub struct GuestImageLayout {
    /// ELF/Mach-O `.text` load VA (runtime override via layout param).
    pub text_base: u64,
    pub text_len: usize,
    /// Arch-specific decode slot size (1=x86_64, 4=aarch64/mips/riscv).
    pub slot_granularity: u64,
    pub data_sections: Vec<DataSectionSpec>,
    pub relocs: Vec<RelocSpec>,
    pub libraries: Vec<LibrarySpec>,
    pub memory_model: MemoryModel,
}

impl GuestImageLayout {
    /// Guest PC of the halt stub: one byte past the last translated decode slot.
    pub fn halt_guest_pc(&self) -> u64 {
        self.text_base + self.text_len as u64
    }

    /// Guest PC of redirect shim `i` (synthetic region after halt in VA space).
    pub fn shim_guest_pc(&self, i: u32) -> u64 {
        self.halt_guest_pc() + (i as u64 + 1) * self.slot_granularity
    }

    /// Map a runtime guest PC to a WASM table index.
    pub fn guest_pc_to_table_idx(
        &self,
        pc: u64,
        runtime_text_base: u64,
        base_func_offset: u32,
    ) -> Option<u32> {
        if pc < runtime_text_base {
            return None;
        }
        let rel = pc - runtime_text_base;
        if rel % self.slot_granularity != 0 {
            return None;
        }
        let slot = rel / self.slot_granularity;
        Some(base_func_offset + slot as u32)
    }

    /// Minimum guest VA among data sections (for HostOffset seeding).
    pub fn data_base_min(&self) -> u64 {
        self.data_sections
            .iter()
            .map(|s| s.addr)
            .min()
            .unwrap_or(0)
    }
}

#[cfg(feature = "binary-load")]
mod binary_load {
    use super::*;
    use binary_io::{LoadedBinary, RelocKind, SectionKind};

    impl GuestImageLayout {
        /// Extract layout from a loaded native binary.
        pub fn from_loaded_binary(bin: &LoadedBinary) -> Self {
            let text = bin
                .sections
                .iter()
                .find(|s| matches!(s.kind, SectionKind::Text))
                .expect("LoadedBinary must have a .text section");

            let slot_granularity = match bin.arch {
                binary_io::BinArch::X86_64 => 1,
                binary_io::BinArch::AArch64 => 4,
            };

            let data_sections: Vec<DataSectionSpec> = bin
                .sections
                .iter()
                .filter(|s| {
                    matches!(s.kind, SectionKind::Data | SectionKind::RoData)
                        && !s.data.is_empty()
                })
                .map(|s| DataSectionSpec {
                    name: s.name.clone(),
                    addr: s.addr,
                    bytes: s.data.clone(),
                })
                .collect();

            let section_index: alloc::collections::BTreeMap<_, _> = bin
                .sections
                .iter()
                .enumerate()
                .map(|(i, s)| (&s.name as &str, i))
                .collect();

            let relocs: Vec<RelocSpec> = bin
                .relocs
                .iter()
                .filter_map(|r| {
                    let kind = match r.kind {
                        RelocKind::X86Abs64 | RelocKind::A64Abs64 => RelocKindTag::Abs64,
                        RelocKind::X86GotPcRel => RelocKindTag::GotPcRel,
                        RelocKind::X86Plt32 => RelocKindTag::Plt32,
                        _ => RelocKindTag::Other,
                    };
                    let target_symbol = match &r.symbol {
                        binary_io::RelocTarget::Symbol(name) => Some(name.clone()),
                        binary_io::RelocTarget::SectionRel(_) => None,
                    };
                    Some(RelocSpec {
                        offset: r.offset,
                        section: r.section,
                        target_symbol,
                        kind,
                        addend: r.addend,
                    })
                })
                .collect();

            let libraries = bin
                .dyn_deps
                .iter()
                .map(|name| LibrarySpec { name: name.clone() })
                .collect();

            let _ = section_index; // reserved for future section-name lookup

            Self {
                text_base: text.addr,
                text_len: text.data.len(),
                slot_granularity,
                data_sections,
                relocs,
                libraries,
                memory_model: MemoryModel::OwnedLinear,
            }
        }
    }
}
