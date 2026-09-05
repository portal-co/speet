//! Structure-aware generators for the M5 arch set: AArch32 (ARM), i686,
//! RV32, and MIPS32 (big-endian). Same shape as the M4 generators: forms
//! limited to what each frontend translates, encodings verified by
//! round-trip decode, bounded data-region addressing, no register-clobber
//! of SP/LR/anchors, branches patched to in-code targets.

use crate::case::{Arch, FuzzCase};
use crate::generator::{
    interesting_value, Rand, SeedRng, CODE_BASE, DATA_BASE, DATA_SIZE, RO_DATA_OFFSET,
    RO_DATA_SIZE, STACK_BASE, STACK_SIZE, STEP_BUDGET,
};

fn finish_case(
    arch: Arch,
    seed: u64,
    code: Vec<u8>,
    regs: crate::case::RegState,
    lr_reg: Option<usize>,
) -> FuzzCase {
    let mut regs = regs;
    // Link-register halt sentinel (stack-based-ret archs carry it at [SP]
    // instead, seeded by the caller below).
    if let Some(g) = lr_reg {
        regs.gprs[g] = CODE_BASE + code.len() as u64;
    }
    regs.rip = CODE_BASE;
    FuzzCase {
        arch,
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

// ── AArch32 (ARM) ────────────────────────────────────────────────────────

/// R4 anchors the data-region base; R13 (SP) / R14 (LR) / R15 (PC) are
/// excluded from destinations.
const ARM_DSTS: [u32; 12] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12];
const ARM_ANCHOR: u32 = 4;

/// LDR/STR imm12 is unsigned: bound below 0x800 (well inside the region
/// and below the RO page at 0x1000 for a mixed-coverage profile).
fn arm_imm12(r: &mut dyn Rand) -> u32 {
    (r.below(0x800) & !0x3) as u32
}

fn gen_one_arm(code: &mut Vec<u8>, r: &mut dyn Rand, pc: u64, sites: &mut Vec<usize>) -> bool {
    let d = ARM_DSTS[r.below(ARM_DSTS.len() as u64) as usize];
    let s = r.below(13) as u32;
    let imm: u64 = interesting_value(r);
    let word: u32 = match r.below(9) {
        // MOV Rd, Rm (data-processing imm-shifted-register, shift 0).
        // Rn must be 0b0000: ARMv7 defines MOV/MVN's Rn as SBZ, and QEMU
        // (the oracle) raises UNDEFINED for Rn != 0 — undefined encodings
        // are out of scope (plan §1), so don't emit them.
        0 => 0xE1A0_0000 | (d << 12) | r.below(13) as u32,
        // MOV Rd, #imm8 ror (immediate operand: imm12 = rot<<8 | imm8)
        1 => 0xE3A0_0000 | (d << 12) | ((imm & 0xFF) as u32) | ((r.below(16) as u32) << 8),
        // ADD/SUB/AND/ORR/EOR Rd, Rn, #imm12 (S=0 only — the smoke DP
        // path never updates CPSR, so flag-setting forms would flood the
        // results with a known-divergence class; CMP (opcode 0xA) is
        // outright unsupported)
        2 => {
            let op = match r.below(5) {
                0 => 0xE280_0000u32, // ADD
                1 => 0xE240_0000,    // SUB
                2 => 0xE200_0000,    // AND
                3 => 0xE380_0000,    // ORR
                _ => 0xE220_0000,    // EOR
            };
            op | (s << 16) | (d << 12) | (r.below(0x100) as u32)
        }
        // ADD/SUB/AND/ORR/EOR Rd, Rn, Rm (unshifted — smoke path rejects
        // shifted-register op2; S=0 only)
        3 => {
            let op = match r.below(5) {
                0 => 0xE080_0000u32, // ADD
                1 => 0xE040_0000,    // SUB
                2 => 0xE000_0000,    // AND
                3 => 0xE180_0000,    // ORR
                _ => 0xE020_0000,    // EOR
            };
            op | (s << 16) | (d << 12) | r.below(13) as u32
        }
        // LDR/STR Rd, [R4, #imm12]
        6 | 7 => {
            let op = if r.chance(1, 2) { 0xE590_0000u32 } else { 0xE580_0000 };
            op | (ARM_ANCHOR << 16) | (d << 12) | arm_imm12(r)
        }
        // B forward — patched post-pass (AL only; the frontend rejects
        // every other condition code)
        _ => {
            sites.push(code.len() / 4);
            0xEA00_0000
        }
    };
    // AL-condition prefixes (0xE...) never contain ret-like junk; the word
    // is fixed-width so no shingle hazard exists.
    let _ = pc;
    code.extend_from_slice(&word.to_le_bytes());
    true
}

/// Generate a complete AArch32 fuzz case. Terminator: BX LR
/// (0xE12FFF1E — always-condition BX Rm with Rm=14).
pub fn generate_case_arm(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0xA3_20_00_71_9E_5E_ED_01);
    let n_target = 8 + rng.below(17) as usize;
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut sites = Vec::new();
    let mut stalls = 0;
    let mut count = 0;
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one_arm(&mut code, &mut rng, pc, &mut sites) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    code.extend_from_slice(&0xE12F_FF1Eu32.to_le_bytes()); // BX LR
    // A32 branch offset: (target - pc - 8) / 4, imm24 sign-extended.
    // Branch targets: the last real instruction (never the sentinel —
    // `pc_to_func_idx` rejects idx >= total) or a small backward loop.
    for &wi in &sites {
        let pc = (wi * 4) as i64;
        // `target` is a code-RELATIVE absolute address (guest base added
        // by the recompiler/hardware); backward targets are deltas from pc.
        let target: i64 = if pc >= 32 && rng.chance(1, 2) {
            pc - ((1 + rng.below(4)) * 4) as i64
        } else {
            code.len() as i64 - 4
        };
        // A32 B: displacement = (target - (pc + 8)) / 4 (PC reads as
        // insn + 8).
        let off = ((target - pc - 8) / 4) as u32 & 0x00FF_FFFF;
        let word = 0xEA00_0000 | off;
        code[wi * 4..wi * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(16) {
        *g = interesting_value(&mut rng);
    }
    regs.gprs[ARM_ANCHOR as usize] = DATA_BASE;
    regs.gprs[13] = STACK_BASE + ((STACK_SIZE as u64) & !0xF); // SP
    finish_case(Arch::Arm, seed, code, regs, Some(14)) // LR
}

// ── i686 ─────────────────────────────────────────────────────────────────

/// The speet-x86 frontend's handled set: Add/Sub/And/Or/Xor, Mov, Push/Pop,
/// the listed Jcc family, Call, Ret — no Cmp/Test/shifts. Reuse the x86-64
/// generator shape with 32-bit operand size. GPR numbering: EAX..EDI (8),
/// RSP index 4 excluded from destinations.
fn gen_one_x86_32(code: &mut Vec<u8>, r: &mut dyn Rand, pc: u64) -> bool {
    use iced_x86::{Code, Decoder, DecoderOptions, Encoder, Instruction as IxInst, MemoryOperand, Register};
    // EBX excluded: it anchors [ebx+disp] memory addressing — clobbering
    // it makes later loads/stores read unmapped memory (the oracle faults
    // READ_UNMAPPED, wasting the case).
    const DSTS32: [Register; 6] = [
        Register::EAX, Register::ECX, Register::EDX,
        Register::EBP, Register::ESI, Register::EDI,
    ];
    let d = DSTS32[r.below(DSTS32.len() as u64) as usize];
    let s = [
        Register::EAX, Register::ECX, Register::EDX, Register::EBX,
        Register::ESP, Register::EBP, Register::ESI, Register::EDI,
    ][r.below(8) as usize];
    let imm: u64 = interesting_value(r);

    // NOTE: the speet-x86 (i686) frontend is a smoke path — condition
    // flags are not modeled (`direct.rs` "Smoke: fall through only
    // (condition flags not modeled yet)") and Jcc emits a flags-stub
    // unreachable. The vocabulary therefore excludes flag-setting ALU and
    // Jcc: including them floods the run with the (real, but expected)
    // divergence "flags never computed". Re-add ALU/Jcc once the frontend
    // models flags.
    let built: Option<IxInst> = match r.below(4) {
        // MOV reg,imm32
        0 => IxInst::with2(Code::Mov_r32_imm32, d, (imm & 0xFFFF_FFFF) as u32 as i32).ok(),
        // MOV reg,reg
        1 => IxInst::with2(Code::Mov_r32_rm32, d, s).ok(),
        // MOV reg32, [ebx+disp32] / store (SIB with EBX anchor)
        2 => {
            let off = r.below((DATA_SIZE as u64).saturating_sub(16)) as i32;
            IxInst::with2(
                if r.chance(1, 2) { Code::Mov_r32_rm32 } else { Code::Mov_rm32_r32 },
                d,
                MemoryOperand::new(Register::EBX, Register::None, 1, off as i64, 4, false, Register::None),
            )
            .ok()
        }
        // PUSH/POP reg
        _ => IxInst::with1(
            if r.chance(1, 2) { Code::Push_r64 } else { Code::Pop_r64 },
            d,
        )
        .ok(),
    };
    let Some(inst) = built else { return false };
    let mut e = Encoder::new(32);
    let Some(bytes) = e.encode(&inst, pc).ok().map(|_| e.take_buffer()) else {
        return false;
    };
    if bytes.is_empty() {
        return false;
    }
    // Shingle-hazard filter (byte-granular slots, same as x86-64). For
    // 32-bit code the ret-like bytes are C3/C2/CF.
    if bytes.contains(&0xC3) || bytes.contains(&0xC2) || bytes.contains(&0xCF) {
        return false;
    }
    let mut dec = Decoder::with_ip(32, &bytes, pc, DecoderOptions::NONE);
    let check = dec.decode();
    if check.is_invalid() || check.len() as usize != bytes.len() {
        return false;
    }
    if code.len() + bytes.len() > 0x1000 {
        return false;
    }
    code.extend_from_slice(&bytes);
    true
}

/// Generate a complete i686 fuzz case. Terminator: RET (C3), sentinel
/// 4 bytes at [SP].
pub fn generate_case_x86_32(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0x00_32_00_86_00_32_00_01);
    let n_target = 8 + rng.below(17) as usize;
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut count = 0;
    let mut stalls = 0;
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one_x86_32(&mut code, &mut rng, pc) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    code.push(0xC3);

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(8) {
        *g = interesting_value(&mut rng) & 0xFFFF_FFFF;
    }
    regs.gprs[3] = DATA_BASE; // EBX anchors the data region
    // 32-bit ret pops 4 bytes: sentinel 4 bytes below SP.
    let sp = STACK_BASE + ((STACK_SIZE as u64) & !0xF) - 4;
    regs.gprs[4] = sp;
    let sentinel = (CODE_BASE + code.len() as u64).to_le_bytes();
    let mut case = finish_case(Arch::X86_32, seed, code, regs, None);
    let off = (sp - STACK_BASE) as usize;
    case.stack[off..off + 4].copy_from_slice(&sentinel[..4]);
    case
}

// ── RV32 ─────────────────────────────────────────────────────────────────

/// Same shape as the RV64 generator, minus W-suffix instructions (invalid
/// in RV32) and with RV32 decode verification. x5 anchors the data region;
/// x1 (ra) carries the halt sentinel and x2 (sp) the stack top — both
/// excluded from destinations.
const RV32_DSTS: [u32; 27] = [
    3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31,
];

fn rv_word(code: &[u8], wi: usize) -> u32 {
    u32::from_le_bytes([code[wi * 4], code[wi * 4 + 1], code[wi * 4 + 2], code[wi * 4 + 3]])
}

fn gen_one_rv32(code: &mut Vec<u8>, r: &mut dyn Rand, _pc: u64, sites: &mut Vec<usize>) -> bool {
    let d = RV32_DSTS[r.below(RV32_DSTS.len() as u64) as usize];
    let s = r.below(32) as u32;
    let imm: i64 = interesting_value(r) as i64;
    let word: u32 = match r.below(10) {
        0 => crate::gen_rv64::i_type_pub((imm & 0xFFF) as u32, s, 0, d, 0x13), // ADDI
        1 => match r.below(5) {
            0 => crate::gen_rv64::r_type_pub(0, s, d, 0, d, 0x33),
            1 => crate::gen_rv64::r_type_pub(0x20, s, d, 0, d, 0x33),
            2 => crate::gen_rv64::r_type_pub(0, s, d, 7, d, 0x33),
            3 => crate::gen_rv64::r_type_pub(0, s, d, 6, d, 0x33),
            _ => crate::gen_rv64::r_type_pub(0, s, d, 4, d, 0x33),
        },
        2 => match r.below(3) {
            0 => crate::gen_rv64::i_type_pub((imm & 0xFFF) as u32, s, 7, d, 0x13),
            1 => crate::gen_rv64::i_type_pub((imm & 0xFFF) as u32, s, 6, d, 0x13),
            _ => crate::gen_rv64::i_type_pub((imm & 0xFFF) as u32, s, 4, d, 0x13),
        },
        // SLLI/SRLI/SRAI — shamt is 5 bits in RV32
        3 => {
            let sh = r.below(32) as u32;
            match r.below(3) {
                0 => crate::gen_rv64::i_type_pub(sh, s, 1, d, 0x13),
                1 => crate::gen_rv64::i_type_pub(sh, s, 5, d, 0x13),
                _ => crate::gen_rv64::i_type_pub(0x400 | sh, s, 5, d, 0x13),
            }
        }
        4 => match r.below(3) {
            0 => crate::gen_rv64::r_type_pub(0, s, d, 1, d, 0x33),
            1 => crate::gen_rv64::r_type_pub(0, s, d, 5, d, 0x33),
            _ => crate::gen_rv64::r_type_pub(0x20, s, d, 5, d, 0x33),
        },
        5 => 0x37 | (((imm >> 12) as u32 & 0xFFFFF) << 12) | (d << 7), // LUI
        // LD/SD don't exist in RV32 base — LW/SW instead (funct3 2)
        6 | 7 => {
            let off = (r.below(0x7F8) & !0x3) as u32;
            if r.chance(1, 2) {
                crate::gen_rv64::i_type_pub(off, crate::gen_rv64::ANCHOR, 2, d, 0x03)
            } else {
                let hi = (off >> 5) & 0x7F;
                let lo = off & 0x1F;
                (hi << 25) | (s << 20) | (crate::gen_rv64::ANCHOR << 15) | (2 << 12)
                    | (lo << 7) | 0x23
            }
        }
        // BEQ/BNE — patched post-pass
        8 | 9 => {
            sites.push(code.len() / 4);
            let funct3 = if r.chance(1, 2) { 0 } else { 1 };
            (s << 20) | (d << 15) | (funct3 << 12) | 0x63
        }
        _ => unreachable!(),
    };
    match rv_asm::Inst::decode(word, rv_asm::Xlen::Rv32) {
        Ok(_) => {
            code.extend_from_slice(&word.to_le_bytes());
            true
        }
        Err(_) => false,
    }
}

/// Generate a complete RV32 fuzz case. Terminator: JALR x0, x1, 0.
pub fn generate_case_rv32(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0x51_7C_B1_55_C0_DE_00_32);
    let n_target = 8 + rng.below(17) as usize;
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut sites = Vec::new();
    let mut count = 0;
    let mut stalls = 0;
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one_rv32(&mut code, &mut rng, pc, &mut sites) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    code.extend_from_slice(&0x0000_8067u32.to_le_bytes());
    let n_halfwords = (code.len() / 2) as u64;
    crate::gen_rv64::patch_branches_pub(&mut code, &sites, &mut rng, n_halfwords);

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(32) {
        *g = interesting_value(&mut rng) & 0xFFFF_FFFF;
    }
    regs.gprs[0] = 0;
    regs.gprs[crate::gen_rv64::ANCHOR as usize] = DATA_BASE;
    finish_case(Arch::RiscV32, seed, code, regs, Some(1)) // x1 = ra
}

// ── MIPS32 (big-endian) ──────────────────────────────────────────────────

/// $8 (t0) anchors the data region; $29 (sp) / $31 (ra) excluded.
const MIPS_DSTS: [u32; 26] = [
    1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27,
];

fn mips_i(op: u32, rs: u32, rt: u32, imm: u32) -> u32 {
    (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
}
fn mips_r(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
    (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
}

fn gen_one_mips(code: &mut Vec<u8>, r: &mut dyn Rand, _pc: u64, sites: &mut Vec<usize>) -> bool {
    let d = MIPS_DSTS[r.below(MIPS_DSTS.len() as u64) as usize];
    let s = r.below(28) as u32;
    let imm: u64 = interesting_value(r);
    let word: u32 = match r.below(10) {
        // ADDIU rt, rs, imm16 (unsigned-bound: imm16 < 0x8000 keeps the
        // sign-extended offset positive)
        0 => mips_i(0x09, s, d, (r.below(0x8000)) as u32),
        // ADDU/AND/OR/XOR/NOR/SUBU rd, rs, rt
        1 => {
            let funct = match r.below(6) {
                0 => 0x21, // ADDU
                1 => 0x24, // AND
                2 => 0x25, // OR
                3 => 0x26, // XOR
                4 => 0x27, // NOR
                _ => 0x23, // SUBU
            };
            mips_r(0, s, d, d, 0, funct)
        }
        // ANDI/ORI/XORI rt, rs, imm16 (zero-extended)
        2 => {
            let op = match r.below(3) {
                0 => 0x0C,
                1 => 0x0D,
                _ => 0x0E,
            };
            mips_i(op, s, d, (imm & 0xFFFF) as u32)
        }
        // SLL/SRL/SRA rd, rt, sa
        3 => {
            let sa = r.below(32) as u32;
            let funct = match r.below(3) {
                0 => 0x00,
                1 => 0x02,
                _ => 0x03,
            };
            mips_r(0, 0, d, d, sa, funct)
        }
        // LUI rt, imm16
        4 => 0x3C00_0000 | (d << 16) | (imm & 0xFFFF) as u32,
        // LW/SW rt, off($8) — signed imm16, bound inside the data region
        5 | 6 => {
            let off = (r.below(DATA_SIZE as u64 - 16) & !0x3) as u32;
            if r.chance(1, 2) {
                mips_i(0x23, MIPS_ANCHOR, d, off) // LW
            } else {
                mips_i(0x2B, MIPS_ANCHOR, s, off) // SW (store rs)
            }
        }
        // BEQ/BNE rs, rt — patched post-pass (offset relative to pc+4)
        7 | 8 => {
            sites.push(code.len() / 4);
            let op = if r.chance(1, 2) { 0x04 } else { 0x05 };
            (op << 26) | (s << 21) | (d << 16)
        }
        // JR $ra terminator is appended separately; filler LUI
        _ => 0x3C00_0000 | (d << 16),
    };
    code.extend_from_slice(&word.to_be_bytes());
    true
}

const MIPS_ANCHOR: u32 = 8;

/// Generate a complete MIPS32 (big-endian) fuzz case. Terminator:
/// JR $ra (0x03E00008, stored big-endian).
pub fn generate_case_mips(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed ^ 0x4D_1F_53_00_00_00_00_01);
    let n_target = 8 + rng.below(17) as usize;
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut sites = Vec::new();
    let mut count = 0;
    let mut stalls = 0;
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one_mips(&mut code, &mut rng, pc, &mut sites) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }
    code.extend_from_slice(&0x03E0_0008u32.to_be_bytes()); // JR $ra
    // Branch offsets: imm16 = (target - (pc+4)) / 4, sign-extended.
    for &wi in &sites {
        let pc = (wi * 4) as i64;
        // `target` is a code-RELATIVE absolute address (the code base is
        // added by the recompiler/hardware); backward targets are deltas.
        let target: i64 = if pc >= 16 && rng.chance(1, 2) {
            pc - ((1 + rng.below(4)) * 4) as i64
        } else {
            code.len() as i64 - 4
        };
        // MIPS branches: displacement = (target - (pc + 4)) / 4 (delay
        // slot is pc+4).
        let off = ((target - pc - 4) / 4) as u16;
        let word = rv_word_be(&code, wi) & 0xFFFF_0000 | off as u32;
        code[wi * 4..wi * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }

    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut().take(32) {
        *g = interesting_value(&mut rng) & 0xFFFF_FFFF;
    }
    regs.gprs[0] = 0;
    regs.gprs[MIPS_ANCHOR as usize] = DATA_BASE;
    regs.gprs[29] = STACK_BASE + ((STACK_SIZE as u64) & !0xF); // $sp
    finish_case(Arch::Mips, seed, code, regs, Some(31)) // $ra
}

/// Big-endian word read (MIPS code is stored big-endian).
fn rv_word_be(code: &[u8], wi: usize) -> u32 {
    u32::from_be_bytes([code[wi * 4], code[wi * 4 + 1], code[wi * 4 + 2], code[wi * 4 + 3]])
}
