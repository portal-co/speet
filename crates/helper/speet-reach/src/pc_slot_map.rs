//! [`PcSlotMap`] — correct PC-to-slot-index mapping for native ISA recompilers.
//!
//! Replaces the formula-based `PassthroughSlots` approach with a pre-decoded
//! sorted list of included instruction PCs.  Binary search gives the correct
//! sequential slot index regardless of instruction width.

use alloc::vec::Vec;
use yecta::SlotAssigner;

use crate::{CfgDecoder, ReachableSet};

/// A sorted list of included guest instruction PCs.
///
/// The index into the sorted list is the sequential 0-based WASM function slot
/// index — correct by construction for any instruction width.
///
/// ## Constructors
///
/// | Method | Omission | Use when |
/// |--------|----------|----------|
/// | [`all_slots`](PcSlotMap::all_slots) | None | Fixing the latent formula bug; no reachability filter |
/// | [`from_reachable`](PcSlotMap::from_reachable) | Unreachable PCs | True slot omission |
pub struct PcSlotMap {
    /// Sorted list of included PCs.  The index is the local slot index.
    sorted_pcs: Vec<u64>,
}

impl PcSlotMap {
    /// Build a `PcSlotMap` from a sequential decode of raw bytes — no omission.
    ///
    /// Walks `bytes` starting at `base`, using `decoder` to determine instruction
    /// lengths.  Every successfully decoded instruction PC is included.
    ///
    /// This is the correct drop-in replacement for formula-based slot counting
    /// (`count_fns` / `rip_to_func_idx` / `pc_to_func_idx`).
    pub fn all_slots(bytes: &[u8], base: u64, decoder: &dyn CfgDecoder) -> Self {
        let mut sorted_pcs = Vec::new();
        let mut offset: usize = 0;
        while offset < bytes.len() {
            let pc = base + offset as u64;
            match decoder.decode_edges(pc, &bytes[offset..]) {
                Some(edges) => {
                    sorted_pcs.push(pc);
                    let step = edges.insn_len as usize;
                    if step == 0 {
                        break; // guard against infinite loop on decoder bug
                    }
                    offset += step;
                }
                None => break,
            }
        }
        Self { sorted_pcs }
    }

    /// Build a `PcSlotMap` from a [`ReachableSet`] — omits unreachable PCs.
    ///
    /// Only PCs present in `set` receive a slot.  PCs absent from `set` are
    /// completely omitted: no WASM function is allocated for them at all.
    pub fn from_reachable(set: &ReachableSet) -> Self {
        // ReachableSet.pcs is a BTreeSet so iteration is already in sorted order.
        let sorted_pcs = set.pcs.iter().copied().collect();
        Self { sorted_pcs }
    }

    /// Total number of instruction slots (= length of the included PC list).
    pub fn len(&self) -> usize {
        self.sorted_pcs.len()
    }

    /// Returns `true` if no PCs are included.
    pub fn is_empty(&self) -> bool {
        self.sorted_pcs.is_empty()
    }
}

impl SlotAssigner for PcSlotMap {
    fn total_slots(&self) -> u32 {
        self.sorted_pcs.len() as u32
    }

    fn slot_for_pc(&self, pc: u64) -> Option<u32> {
        self.sorted_pcs
            .binary_search(&pc)
            .ok()
            .map(|idx| idx as u32)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CfgEdges, ReachabilitySpec, compute_reachable};
    use alloc::vec;
    use yecta::SlotAssigner;

    /// Fixed-4-byte decoder (simulates MIPS or standard RISC-V).
    struct Fixed4Decoder;
    impl CfgDecoder for Fixed4Decoder {
        fn decode_edges(&self, _pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
            if bytes.len() < 4 {
                return None;
            }
            Some(CfgEdges {
                static_successors: vec![],
                fallthrough: true,
                insn_len: 4,
                has_indirect: false,
            })
        }
    }

    #[test]
    fn all_slots_assigns_sequential_indices() {
        // 3 × 4-byte instructions at 0x1000, 0x1004, 0x1008.
        // Old formula: (0x1000 - 0x1000) / 4 = 0, (0x1004 - 0x1000) / 4 = 1, etc.
        // PcSlotMap: 0, 1, 2 — same for fixed-4 but proven correct for variable-width.
        let bytes = [0u8; 12];
        let map = PcSlotMap::all_slots(&bytes, 0x1000, &Fixed4Decoder);
        assert_eq!(map.total_slots(), 3);
        assert_eq!(map.slot_for_pc(0x1000), Some(0));
        assert_eq!(map.slot_for_pc(0x1004), Some(1));
        assert_eq!(map.slot_for_pc(0x1008), Some(2));
    }

    #[test]
    fn slot_for_pc_mid_instruction_returns_none() {
        // Bytes between instruction starts should not be in the map.
        let bytes = [0u8; 8];
        let map = PcSlotMap::all_slots(&bytes, 0x1000, &Fixed4Decoder);
        assert_eq!(map.slot_for_pc(0x1001), None);
        assert_eq!(map.slot_for_pc(0x1002), None);
        assert_eq!(map.slot_for_pc(0x1003), None);
    }

    #[test]
    fn from_reachable_omits_unreachable() {
        // 3 instructions; mark only 0x1000 and 0x1008 as reachable (omit 0x1004).
        let bytes = [0u8; 12];
        let spec = ReachabilitySpec::new([0x1000, 0x1008]);
        let reachable = compute_reachable(&spec, &bytes, 0x1000, &Fixed4Decoder);
        // BFS from 0x1000 follows fallthrough to 0x1004, 0x1008 — all three reachable.
        // Use manual from_reachable test with a custom set instead.
        let _ = reachable; // just verify it doesn't panic

        // Build a two-element map manually: slot 0 → 0x1000, slot 1 → 0x1008.
        let mut set_pcs = alloc::collections::BTreeSet::new();
        set_pcs.insert(0x1000u64);
        set_pcs.insert(0x1008u64);
        let set = ReachableSet { pcs: set_pcs };
        let map = PcSlotMap::from_reachable(&set);

        assert_eq!(map.total_slots(), 2);
        assert_eq!(map.slot_for_pc(0x1000), Some(0));
        assert_eq!(map.slot_for_pc(0x1004), None); // omitted
        assert_eq!(map.slot_for_pc(0x1008), Some(1));
    }

    #[test]
    fn all_slots_fixes_formula_for_variable_width() {
        // Simulate a 2-byte-then-4-byte sequence starting at 0x1000.
        // With the old RISC-V formula (stride=2): 0x1000→0, 0x1002→1.
        // With PcSlotMap: 0x1000→0, 0x1002→1 — same here, but the key point is
        // a 4-byte instruction at 0x1004 would get slot 2, NOT slot 2 via the formula.
        struct Mixed2And4Decoder;
        impl CfgDecoder for Mixed2And4Decoder {
            fn decode_edges(&self, pc: u64, bytes: &[u8]) -> Option<CfgEdges> {
                if bytes.is_empty() {
                    return None;
                }
                // First instruction is 2 bytes; remaining are 4 bytes.
                let len = if pc == 0x1000 { 2u32 } else { 4u32 };
                if bytes.len() < len as usize {
                    return None;
                }
                Some(CfgEdges {
                    static_successors: vec![],
                    fallthrough: true,
                    insn_len: len,
                    has_indirect: false,
                })
            }
        }
        // 10 bytes: 2-byte insn at 0x1000, 4-byte at 0x1002, 4-byte at 0x1006.
        let bytes = [0u8; 10];
        let map = PcSlotMap::all_slots(&bytes, 0x1000, &Mixed2And4Decoder);
        assert_eq!(map.total_slots(), 3);
        assert_eq!(map.slot_for_pc(0x1000), Some(0));
        assert_eq!(map.slot_for_pc(0x1002), Some(1));
        assert_eq!(map.slot_for_pc(0x1006), Some(2));
        // The old stride-2 formula would say (0x1006 - 0x1000) / 2 = 3 — wrong!
    }
}
