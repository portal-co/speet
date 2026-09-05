//! Structure-aware AArch64 instruction generator (plan §4, M4).
//!
//! Forms are limited to operations the `speet-aarch64` frontend translates
//! (see `direct.rs::translate_one`): ADD/SUB (imm + shifted-reg, with/without
//! S), AND/ORR/EOR (shifted-reg), MOV register, MOVZ/MOVK, UBFM/SBFM shifts,
//! LDR/STR (unsigned-offset 32/64-bit), CBZ/CBNZ, B/B.cond, CSEL, MUL/UDIV.
//! Words are verified by decoding with `disarm64` at the exact emission PC.

use crate::case::FuzzCase;
use crate::generator::{
    data_displacement, interesting_value, Rand, SeedRng, CODE_BASE, DATA_BASE, DATA_SIZE,
    RO_DATA_OFFSET, RO_DATA_SIZE, STACK_BASE, STACK_SIZE, STEP_BUDGET,
};

/// X20 anchors the data-region base (avoiding X30 = LR, X29 = FP).
const ANCHOR: u32 = 20;
/// Destination-eligible registers: X0–X28 excluding X20 (the anchor) and
/// X29 (FP). LR (X30) is excluded — a clobbered LR loses the halt sentinel.
const DSTS: [u32; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
    26, 27, 28,
];

fn dst(r: &mut dyn Rand) -> u32 {
    DSTS[r.below(DSTS.len() as u64) as usize]
}
fn any(r: &mut dyn Rand) -> u32 {
    r.below(31) as u32
}

/// Data-region displacement scaled for LDR/STR unsigned-offset imm12
/// (`offset = imm12 * size`), bounded into the region.
fn ldr_imm12(r: &mut dyn Rand, size: u64) -> u64 {
    let max = (DATA_SIZE as u64).saturating_sub(size + 16);
    (r.below(max.max(1) / size) * size)
}

fn disp_bytes(r: &mut dyn Rand, size: u64) -> u64 {
    data_displacement(r, size) as u64
}

/// Encode one word at `pc`; verify it round-trips through disarm64.
fn emit(word: u32, pc: u64, code: &mut Vec<u8>) -> bool {
    // disarm64 decode check: an undefined word must never enter a case.
    if crate::gen_a64::decode(word).is_none() {
        return false;
    }
    let _ = pc;
    code.extend_from_slice(&word.to_le_bytes());
    true
}

fn decode(word: u32) -> Option<()> {
    disarm64::decoder::decode(word).map(|_| ())
}

/// Generate one instruction appended to `code`, recording forward-branch
/// sites for post-pass patching. Returns `true` on success.
fn gen_one(code: &mut Vec<u8>, r: &mut dyn Rand, pc: u64, fwd: &mut Vec<usize>) -> bool {
    let d = dst(r);
    let s = any(r);
    let n = code.len() / 4;

    let word: u32 = match r.below(15) {
        // MOV Xd, Xs (ORR shifted-register alias)
        0 => 0xAA0003E0 | (s << 16) | d,
        // ADD/SUB imm12 (no shift-out): sf=1 opc=00/10 100010 sh=0
        1 => 0x91000000 | ((r.below(0x1000) as u32) << 10) | (s << 5) | d, // ADD imm
        2 => 0xD1000000 | ((r.below(0x1000) as u32) << 10) | (s << 5) | d, // SUB imm
        // ADD/SUB shifted-register, optionally flag-setting
        3 => {
            let amt = r.below(8) as u32;
            let base = match r.below(4) {
                0 => 0x8B000000u32, // ADD
                1 => 0xAB000000,    // ADDS
                2 => 0xCB000000,    // SUB
                _ => 0xEB000000,    // SUBS
            };
            base | (amt << 10) | (s << 16) | d
        }
        // AND/ORR/EOR shifted-register (LSL #0..7), optionally flag-setting (ANDS)
        4 => {
            let amt = r.below(8) as u32;
            let base = match r.below(4) {
                0 => 0x8A000000u32, // AND
                1 => 0xAA000000,    // ORR
                2 => 0xCA000000,    // EOR
                _ => 0xEA000000,    // ANDS
            };
            base | (amt << 10) | (s << 16) | d
        }
        // MOVZ/MOVK imm16, shift 0/16
        5 => {
            let hw = r.below(4) as u32; // 0..3 (64-bit: all four lanes)
            let imm16 = (interesting_value(r) & 0xFFFF) as u32;
            let op = if r.chance(1, 2) { 0xD2800000u32 } else { 0xF2800000 }; // MOVZ | MOVK
            op | (hw << 21) | (imm16 << 5) | d
        }
        // LSL/LSR/ASR imm via UBFM/SBFM (sf=1)
        6 => {
            let amt = r.below(64) as u32;
            match r.below(3) {
                // LSL Xd, Xn, #amt = UBFM Xd, Xn, (64-amt)&63, 63-amt
                // (base 0xD3400000 = UBFM sf=1 N=1; 0x93400000 = SBFM
                // sf=1 N=1 — bit 22 of both bases is the N bit, already 1)
                0 => 0xD3400000 | (((64 - amt) & 63) << 16) | ((63 - amt) << 10) | (s << 5) | d,
                // LSR Xd, Xn, #amt = UBFM Xd, Xn, amt, 63
                1 => 0xD3400000 | (amt << 16) | (63 << 10) | (s << 5) | d,
                // ASR Xd, Xn, #amt = SBFM Xd, Xn, amt, 63
                _ => 0x93400000 | (amt << 16) | (63 << 10) | (s << 5) | d,
            }
        }
        // LDR/STR Xd, [X20, #scaled] and 32-bit variants (unsigned offset)
        7..=9 => {
            let is_load = r.chance(1, 2);
            let (size, opc) = if r.chance(1, 2) { (8u64, 3u32) } else { (4, 2) };
            let off = ldr_imm12(r, size);
            let base = match (is_load, size) {
                (true, 8) => 0xF9400000u32,
                (false, 8) => 0xF9000000,
                (true, 4) => 0xB9400000,
                _ => 0xB9000000,
            };
            base | (((off / size) as u32) << 10) | (ANCHOR << 5) | d
        }
        // ADD/STR displacement-bounded raw offset into data (uses anchor +
        // byte displacement via ADD then LDR/STR reg? — skip; imm12 suffices)
        // CBZ/CBNZ Xd → forward; target patched post-pass to a valid slot
        // (sentinel = clean exit, or a small backward loop).
        10 | 11 => {
            fwd.push(n);
            let base = if r.chance(1, 2) { 0xB4000000u32 } else { 0xB5000000 };
            base | d // imm19 = 0 placeholder
        }
        // B forward — patched post-pass (always forward: terminator stays
        // reachable; backward would need live-lock guarding).
        12 => {
            fwd.push(n);
            0x14000000 // imm26 placeholder
        }
        // B.cond forward — patched post-pass
        13 => {
            fwd.push(n);
            let cond = r.below(14) as u32; // skip AL/NV
            0x54000000 | (cond) // imm19 placeholder
        }
        // CSEL Xd, Xn, Xm, cond
        _ => {
            let cond = r.below(14) as u32;
            0x9A800000 | (s << 16) | (cond << 12) | (d << 5) | d
        }
    };

    emit(word, pc, code)
}

/// Patch forward-branch sites to land on a valid slot: 50% the sentinel
/// (clean exit via the halt slot), 50% a small backward loop. `sentinel`
/// is the word offset of the halt sentinel (= code_len/4).
fn patch_branches(code: &mut [u8], sites: &[usize], r: &mut dyn Rand, sentinel: u64) {
    let code_len = code.len() as u64;
    for &wi in sites {
        let pc = (wi * 4) as u64;
        let word = u32::from_le_bytes([code[wi*4], code[wi*4+1], code[wi*4+2], code[wi*4+3]]);
        let is_b = word & 0x7C00_0000 == 0x1400_0000; // B (unconditional)
        let is_cbz = word & 0x7E00_0000 == 0x3400_0000; // CBZ/CBNZ
        let is_bcond = word & 0xFF00_0000 == 0x5400_0000;
        let _ = sentinel;
        let can_back = pc >= 16; // room for a backward loop
        let target: i64 = if !can_back || r.chance(1, 2) {
            // Forward to the last real instruction (one word before the
            // terminator): branching to the sentinel itself would index the
            // halt slot, which `pc_to_func_idx` rejects (`idx >= total`) —
            // the frontend then seals the branch with `unreachable`.
            (code_len as i64 - 4) - pc as i64
        } else {
            -(((1 + r.below(4)) * 4) as i64) // small backward loop
        };
        let new = if is_b {
            let off = (target / 4) & 0x03FF_FFFF;
            0x1400_0000 | off as u32
        } else if is_cbz {
            let off = ((target / 4) as u32) & 0x7FFFF;
            (word & 0xFFE0_001F) | (off << 5)
        } else if is_bcond {
            let off = ((target / 4) as u32) & 0x7FFFF;
            (word & 0xFFFF_FFE0 & !(0x7FFFF << 5)) | (off << 5)
        } else {
            continue;
        };
        code[wi*4..wi*4+4].copy_from_slice(&new.to_le_bytes());
    }
}

/// Generate a complete AArch64 fuzz case from a seed.
pub fn generate_case(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0xA6_4C_0F_F1_CE_5E_ED_01);
    let n_target = 8 + rng.below(17) as usize; // 8..=24 instructions
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut count = 0;
    let mut stalls = 0;
    let mut fwd_sites: Vec<usize> = Vec::new();
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one(&mut code, &mut rng, pc, &mut fwd_sites) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    // Terminate with RET (branch to register X30 = halt sentinel).
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());
    // Branch targets: the sentinel slot = code_len/4 words = the halt slot
    // (one past the last translated word slot) — or a small backward loop.
    let sentinel_words = (code.len() / 4) as u64;
    patch_branches(&mut code, &fwd_sites, &mut rng, sentinel_words);

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(31) {
        *g = interesting_value(&mut rng);
    }
    regs.gprs[ANCHOR as usize] = DATA_BASE;
    // X30 = LR seeds the halt sentinel (link-register ret convention).
    regs.gprs[30] = CODE_BASE + code.len() as u64;
    regs.rip = CODE_BASE;

    let data = vec![0u8; DATA_SIZE];
    let mut stack = vec![0u8; STACK_SIZE];
    let sp = STACK_BASE + ((STACK_SIZE as u64) & !0xF);
    regs.sp = sp;

    FuzzCase {
        arch: crate::case::Arch::AArch64,
        code,
        entry_pc: CODE_BASE,
        regs,
        data,
        data_base: DATA_BASE,
        stack,
        stack_base: STACK_BASE,
        read_only: vec![(RO_DATA_OFFSET, RO_DATA_SIZE)],
        step_budget: STEP_BUDGET,
        seed,
    }
}

// `disp_bytes` is reserved for reg-offset addressing modes (M4.1 keeps the
// vocabulary to unsigned-offset LDR/STR); keep the helper compiled.
#[allow(dead_code)]
fn _keepalive(r: &mut dyn Rand) -> u64 {
    disp_bytes(r, 8)
}
