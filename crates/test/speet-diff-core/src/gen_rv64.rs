//! Structure-aware RV64 instruction generator (plan §4, M4).
//!
//! Integer-only (FP is a later milestone), forms limited to what the
//! `speet-riscv` frontend translates: ADDI/ADD/SUB/AND/OR/XOR (reg+imm),
//! SLLI/SRLI/SRAI/SLL/SRL/SRA, LUI, LD/SD (bounded), LUI-based addressing,
//! BEQ/BNE forward, JALR (ret). Words are verified with `rv-asm`'s decoder
//! (`Inst::decode`) at RV64.

use crate::case::FuzzCase;
use crate::generator::{
    interesting_value, Rand, SeedRng, CODE_BASE, DATA_BASE, DATA_SIZE, RO_DATA_OFFSET,
    RO_DATA_SIZE, STACK_BASE, STACK_SIZE, STEP_BUDGET,
};
use rv_asm::{Inst, IsCompressed, Xlen};

/// x5 (t0) anchors the data-region base — avoiding x1 (ra), x2 (sp).
pub(crate) const ANCHOR: u32 = 5;
/// Destination-eligible registers: x3–x31 minus the anchor (x5). x0 is
/// hardwired zero (writes are ignored architecturally); x1 = ra is excluded
/// (it carries the halt sentinel — clobbering it makes the final `ret`
/// jump outside the table); x2 = SP is excluded (stack boundedness at
/// generation time).
const DSTS: [u32; 27] = [
    3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31,
];

fn dst(r: &mut dyn Rand) -> u32 {
    DSTS[r.below(DSTS.len() as u64) as usize]
}
fn any(r: &mut dyn Rand) -> u32 {
    r.below(32) as u32
}

pub(crate) fn r_type_pub(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    r_type(funct7, rs2, rs1, funct3, rd, opcode)
}

fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
pub(crate) fn i_type_pub(imm12: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    i_type(imm12, rs1, funct3, rd, opcode)
}

fn i_type(imm12: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// Sign a 12-bit immediate for encoding (the field is stored as-is; the
/// hardware sign-extends).
fn simm12(v: i64) -> u32 {
    (v & 0xFFF) as u32
}

fn decode(word: u32) -> bool {
    Inst::decode(word, Xlen::Rv64).is_ok()
}

/// Generate one instruction appended to `code`. Returns `true` on success.
fn gen_one(
    code: &mut Vec<u8>,
    r: &mut dyn Rand,
    _pc: u64,
    branch_sites: &mut Vec<usize>,
) -> bool {
    let d = dst(r);
    let s = any(r);
    let imm: i64 = interesting_value(r) as i64;

    let word: u32 = match r.below(13) {
        // ADDI rd, rs, imm12 (biased to interesting values, clamped to imm12)
        0 => i_type(simm12(imm), s, 0, d, 0x13),
        // ADD / SUB / AND / OR / XOR rd, rs1, rs2
        1 => match r.below(5) {
            0 => r_type(0, s, d, 0, d, 0x33),
            1 => r_type(0x20, s, d, 0, d, 0x33),
            2 => r_type(0, s, d, 7, d, 0x33),
            3 => r_type(0, s, d, 6, d, 0x33),
            _ => r_type(0, s, d, 4, d, 0x33),
        },
        // ANDI/ORI/XORI rd, rs, imm12
        2 => match r.below(3) {
            0 => i_type(simm12(imm), s, 7, d, 0x13),
            1 => i_type(simm12(imm), s, 6, d, 0x13),
            _ => i_type(simm12(imm), s, 4, d, 0x13),
        },
        // SLLI/SRLI/SRAI rd, rs, shamt(0..63)
        3 => {
            let sh = r.below(64) as u32;
            match r.below(3) {
                0 => i_type(sh, s, 1, d, 0x13),
                1 => i_type(sh, s, 5, d, 0x13),
                _ => i_type(0x400 | sh, s, 5, d, 0x13),
            }
        }
        // SLL/SRL/SRA rd, rs1, rs2
        4 => match r.below(3) {
            0 => r_type(0, s, d, 1, d, 0x33),
            1 => r_type(0, s, d, 5, d, 0x33),
            _ => r_type(0x20, s, d, 5, d, 0x33),
        },
        // LUI rd, imm20
        5 => 0x37 | (((imm >> 12) as u32 & 0xFFFFF) << 12) | (d << 7),
        // LD rd, imm12(x5) — displacement bounded into the data region
        6 => {
            // LD's imm12 is sign-extended: keep the offset below 0x800 so
            // the encoded field stays positive (0x7F8 < DATA_SIZE).
            let off = r.below(0x7F8) as i64;
            i_type(simm12(off), ANCHOR, 3, d, 0x03)
        }
        // SD rs2=any, imm12(x5) — bounded
        7 => {
            // SD's imm12 is sign-extended: same 0x800 bound as LD.
            let off = r.below(0x7F8) as i64;
            let hi = (((off as u32) >> 5) & 0x7F) as u32;
            let lo = ((off as u32) & 0x1F) as u32;
            (hi << 25) | (s << 20) | (ANCHOR << 15) | (3 << 12) | (lo << 7) | 0x23
        }
        // ADDIW rd, rs, imm12 (W-suffix: 32-bit add, sign-extended)
        8 => i_type(simm12(imm), s, 0, d, 0x1B),
        // SLLIW/SRLIW rd, rs, shamt(0..31)
        9 => {
            let sh = r.below(32) as u32;
            if r.chance(1, 2) {
                i_type(sh, s, 1, d, 0x1B)
            } else {
                i_type(sh, s, 5, d, 0x1B)
            }
        }
        // BEQ/BNE rs1=d, rs2=any → target patched post-pass to a valid
        // slot (sentinel = clean exit, or a small backward loop)
        10 | 11 => {
            branch_sites.push(code.len() / 4);
            let funct3 = if r.chance(1, 2) { 0 } else { 1 };
            (s << 20) | (d << 15) | (funct3 << 12) | 0x63 // imm=0 placeholder
        }
        // XORI bias — replaced with AUIPC (rd, imm20) — PC-relative; both
        // engines compute identically. Keep it rare.
        _ => 0x17 | (((imm >> 12) as u32 & 0xFFFFF) << 12) | (d << 7),
    };

    if !decode(word) {
        return false;
    }
    code.extend_from_slice(&word.to_le_bytes());
    true
}

/// Patch branch sites: 50% the sentinel (clean exit), 50% a small backward
/// loop. B-type immediate: imm[12|10:5] | imm[4:1|11].
pub(crate) fn patch_branches_pub(code: &mut [u8], sites: &[usize], r: &mut dyn Rand, sentinel: u64) {
    patch_branches(code, sites, r, sentinel)
}

fn patch_branches(code: &mut [u8], sites: &[usize], r: &mut dyn Rand, sentinel: u64) {
    let code_len = code.len() as u64;
    for &wi in sites {
        let pc = (wi * 4) as u64;
        let _ = sentinel;
        let can_back = pc >= 16; // room for a backward loop
        let target: i64 = if !can_back || r.chance(1, 2) {
            // Forward to the last real instruction (see gen_a64 note).
            (code_len as i64 - 4) - pc as i64
        } else {
            -(((1 + r.below(4)) * 4) as i64)
        };
        let off = target as u32;
        let b12 = (off >> 12) & 1;
        let b11 = (off >> 11) & 1;
        let b10_5 = (off >> 5) & 0x3F;
        let b4_1 = (off >> 1) & 0xF;
        let word = u32::from_le_bytes([code[wi*4], code[wi*4+1], code[wi*4+2], code[wi*4+3]]);
        let new = ((b12 as u32) << 31)
            | ((b10_5 as u32) << 25)
            | (word & 0x01FF_FFFF) // keep rs2/rs1/funct3/opcode
            | ((b4_1 as u32) << 8)
            | ((b11 as u32) << 7);
        code[wi*4..wi*4+4].copy_from_slice(&new.to_le_bytes());
    }
}

/// Generate a complete RV64 fuzz case from a seed.
pub fn generate_case(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0x51_7C_B1_55_C0_DE_00_64);
    let n_target = 8 + rng.below(17) as usize; // 8..=24 instructions
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut count = 0;
    let mut stalls = 0;
    let mut branch_sites: Vec<usize> = Vec::new();
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one(&mut code, &mut rng, pc, &mut branch_sites) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    // Terminate with JALR x0, x1, 0 (ret through the link register = halt
    // sentinel).
    code.extend_from_slice(&0x0000_8067u32.to_le_bytes());
    // Branch targets: the sentinel slot (code_len/2 halfwords = halt slot)
    // or a small backward loop.
    let sentinel_halfwords = (code.len() / 2) as u64;
    patch_branches(&mut code, &branch_sites, &mut rng, sentinel_halfwords);

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(32) {
        *g = interesting_value(&mut rng);
    }
    regs.gprs[0] = 0; // x0 hardwired zero — both engines enforce this
    regs.gprs[ANCHOR as usize] = DATA_BASE;
    // x1 (ra) seeds the halt sentinel (link-register ret convention); x2
    // (sp) the stack top.
    regs.gprs[1] = CODE_BASE + code.len() as u64;
    regs.gprs[2] = STACK_BASE + ((STACK_SIZE as u64) & !0xF);
    regs.rip = CODE_BASE;

    FuzzCase {
        arch: crate::case::Arch::RiscV64,
        code,
        entry_pc: CODE_BASE,
        regs,
        data: vec![0u8; DATA_SIZE],
        data_base: DATA_BASE,
        stack: vec![0u8; STACK_SIZE],
        stack_base: STACK_BASE,
        read_only: vec![(RO_DATA_OFFSET, RO_DATA_SIZE)],
        step_budget: STEP_BUDGET,
        seed,
    }
}
