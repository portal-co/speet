//! Compile-time catalog of translated guest functions for fn-ptr stub mapping.

use binary_io::BinArch;

use crate::frontend::{halt_addr, Translated};

/// One translated guest decode slot (including the halt sentinel).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestFuncEntry {
    /// Guest virtual PC this slot decodes (`start_addr + local_idx * granularity`).
    pub guest_pc: u64,
    /// Absolute WASM function index (`n_imports + local_idx`).
    pub wasm_func_idx: u32,
    /// 0-based index into `Translated::fns` (halt uses `fns.len()`).
    pub local_func_idx: u32,
    /// True for the halt stub at `start_addr + text_len`.
    pub is_halt: bool,
}

/// Static map from guest PC → per-function host stub / WASM func index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestFuncCatalog {
    pub start_addr: u64,
    pub granularity: u64,
    pub n_imports: u32,
    pub n_redirect_shims: u32,
    pub entries: Vec<GuestFuncEntry>,
}

impl GuestFuncCatalog {
    /// Build a catalog from link-time layout (no [`Translated`] in hand).
    pub fn from_layout(
        start_addr: u64,
        arch: BinArch,
        n_imports: u32,
        n_translated_slots: u32,
        text_len: usize,
        n_redirect_shims: u32,
    ) -> Self {
        let granularity = slot_granularity(arch);
        let mut entries = Vec::with_capacity(n_translated_slots as usize + n_redirect_shims as usize + 1);
        for local_idx in 0..n_translated_slots {
            entries.push(GuestFuncEntry {
                guest_pc: start_addr + local_idx as u64 * granularity,
                wasm_func_idx: n_imports + local_idx,
                local_func_idx: local_idx,
                is_halt: false,
            });
        }
        for shim_i in 0..n_redirect_shims {
            let local_idx = n_translated_slots + shim_i;
            entries.push(GuestFuncEntry {
                guest_pc: speet_link_core::GuestImageLayout {
                    text_base: start_addr,
                    text_len,
                    slot_granularity: granularity,
                    data_sections: vec![],
                    relocs: vec![],
                    libraries: vec![],
                    memory_model: speet_link_core::MemoryModel::OwnedLinear,
                }
                .shim_guest_pc(shim_i),
                wasm_func_idx: n_imports + local_idx,
                local_func_idx: local_idx,
                is_halt: false,
            });
        }
        let halt_local = n_translated_slots + n_redirect_shims;
        let halt_pc = halt_addr(start_addr, text_len);
        entries.push(GuestFuncEntry {
            guest_pc: halt_pc,
            wasm_func_idx: n_imports + halt_local,
            local_func_idx: halt_local,
            is_halt: true,
        });
        Self {
            start_addr,
            granularity,
            n_imports,
            n_redirect_shims,
            entries,
        }
    }

    /// Derive layout from a finished WASM module plus guest `.text` metadata.
    pub fn from_wasm(
        wasm: &[u8],
        start_addr: u64,
        arch: BinArch,
        text_len: usize,
        n_redirect_shims: u32,
    ) -> Self {
        let n_imports = crate::drive::import_func_count(wasm);
        let n_code = crate::drive::code_func_count(wasm);
        // code funcs = translated + shims + halt [+ optional data_init]
        let n_translated_slots = n_code
            .saturating_sub(1 + n_redirect_shims)
            .saturating_sub(if crate::drive::has_data_init_export(wasm) { 1 } else { 0 });
        Self::from_layout(start_addr, arch, n_imports, n_translated_slots, text_len, n_redirect_shims)
    }

    pub fn to_stub_entries(&self) -> Vec<speet_rt::GuestStubEntry> {
        self.entries
            .iter()
            .map(|e| speet_rt::GuestStubEntry {
                guest_pc: e.guest_pc,
                wasm_func_idx: e.wasm_func_idx,
                local_func_idx: e.local_func_idx,
                is_halt: e.is_halt,
            })
            .collect()
    }

    pub fn from_translated(
        translated: &Translated,
        start_addr: u64,
        arch: BinArch,
        n_imports: u32,
        text_len: usize,
        n_redirect_shims: u32,
    ) -> Self {
        Self::from_layout(
            start_addr,
            arch,
            n_imports,
            translated.fns.len() as u32,
            text_len,
            n_redirect_shims,
        )
    }

    pub fn pc_to_entry(&self, guest_pc: u64) -> Option<&GuestFuncEntry> {
        self.entries.iter().find(|e| e.guest_pc == guest_pc)
    }

    pub fn entry_func(&self, entry_func_idx: u32) -> Option<&GuestFuncEntry> {
        self.entries.iter().find(|e| e.local_func_idx == entry_func_idx && !e.is_halt)
    }

    pub fn halt_entry(&self) -> &GuestFuncEntry {
        self.entries
            .iter()
            .find(|e| e.is_halt)
            .expect("catalog always includes halt entry")
    }
}

fn slot_granularity(arch: BinArch) -> u64 {
    match arch {
        BinArch::X86_64 => 1,
        BinArch::AArch64 => 4,
        BinArch::RiscV64 => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::Translated;
    use wasm_encoder::{Function, Instruction, ValType};

    #[test]
    fn catalog_maps_halt_and_entry() {
        let mut f = Function::new([]);
        f.instruction(&Instruction::Unreachable);
        f.instruction(&Instruction::End);
        let translated = Translated {
            fns: vec![f.clone(), f],
            params: vec![ValType::I64; 4],
            unsupported: vec![],
            entry_func_idx: 1,
        };
        let catalog = GuestFuncCatalog::from_translated(
            &translated,
            0x1000,
            BinArch::X86_64,
            5,
            12,
            0,
        );
        assert_eq!(catalog.entries.len(), 3);
        assert_eq!(catalog.entry_func(1).unwrap().guest_pc, 0x1001);
        assert_eq!(catalog.halt_entry().guest_pc, 0x100c);
        assert_eq!(catalog.halt_entry().wasm_func_idx, 5 + 2);
        assert!(catalog.pc_to_entry(0x1001).is_some());
        assert!(catalog.pc_to_entry(0x1002).is_none());
    }
}
