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
    /// Same address math as [`OwnedLinear`] (`guest_va` == WASM linear offset,
    /// `host_mem_base` is compile-time 0), but unrecompiled `.text` is **not**
    /// placed into the mirror at [`GuestImageLayout::text_base`]. That range
    /// stays unmapped / intercepted so host BridgeSupport never sees original
    /// machine code; PC math still uses `text_base`. See `docs/future/zero-offset.md`.
    ZeroOffset,
    /// `physical = guest_virtual + host_mem_base` (see `HostOffsetMapper`).
    HostOffset,
}

impl MemoryModel {
    /// Identity guest VA → WASM offset (no `host_mem_base` add).
    pub fn is_identity_offset(self) -> bool {
        matches!(self, Self::OwnedLinear | Self::ZeroOffset)
    }

    /// Unrecompiled `.text` must not be `memory.init`'d into the linear mirror.
    pub fn text_unmapped(self) -> bool {
        matches!(self, Self::ZeroOffset)
    }
}

/// Soft cap matching the thin runtime's default `__wasm_mem` reservation
/// (256 × 64 KiB). [`GuestImageLayout::from_loaded_binary`] selects
/// [`MemoryModel::ZeroOffset`] only when `data_end_max` fits under this.
pub const ZERO_OFFSET_MAX_DATA_END: u64 = 256 * 65536;

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

    /// Exclusive end of the unrecompiled text VA range `[text_base, text_end)`.
    pub fn text_end(&self) -> u64 {
        self.text_base.saturating_add(self.text_len as u64)
    }

    /// True if `va` lies in the unrecompiled text hole (ZeroOffset policy).
    pub fn va_in_text_hole(&self, va: u64) -> bool {
        self.memory_model.text_unmapped() && va >= self.text_base && va < self.text_end()
    }

    /// Highest exclusive VA that data sections need committed in the mirror.
    pub fn data_end_max(&self) -> u64 {
        self.data_sections
            .iter()
            .map(|s| s.addr.saturating_add(s.bytes.len() as u64))
            .max()
            .unwrap_or(0)
    }

    /// Use ZeroOffset memory model (text unmapped; identity guest VA offsets).
    pub fn with_zero_offset(mut self) -> Self {
        self.memory_model = MemoryModel::ZeroOffset;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_offset_helpers() {
        let layout = GuestImageLayout {
            text_base: 0x1000,
            text_len: 0x40,
            slot_granularity: 4,
            data_sections: vec![DataSectionSpec {
                name: ".rodata".into(),
                addr: 0x2000,
                bytes: vec![1, 2, 3, 4],
            }],
            relocs: vec![],
            libraries: vec![],
            memory_model: MemoryModel::ZeroOffset,
        };
        assert!(layout.memory_model.text_unmapped());
        assert!(layout.memory_model.is_identity_offset());
        assert!(layout.va_in_text_hole(0x1000));
        assert!(!layout.va_in_text_hole(0x1040));
        assert_eq!(layout.data_end_max(), 0x2004);
        assert_eq!(layout.text_end(), 0x1040);
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

            let mut layout = Self {
                text_base: text.addr,
                text_len: text.data.len(),
                slot_granularity,
                data_sections,
                relocs,
                libraries,
                memory_model: MemoryModel::OwnedLinear,
            };
            // Low-VA images: ZeroOffset (text unmapped, identity offsets).
            // High-VA PIE: keep OwnedLinear until HostOffset packing lands.
            if layout.data_end_max() <= ZERO_OFFSET_MAX_DATA_END {
                layout.memory_model = MemoryModel::ZeroOffset;
            }
            layout
        }
    }
}
