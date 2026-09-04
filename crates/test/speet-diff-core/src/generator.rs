//! Structure-aware x86-64 instruction generator (plan §4).
//!
//! Raw random bytes are almost useless — most decode as unsupported or trap.
//! The generator produces *forms* the frontend's decode tables claim to
//! support (ADD/SUB/AND/OR/XOR, SHL/SHR/SAR, MOV/MOVZX/MOVSX/LEA/XCHG,
//! INC/DEC/NOT/NEG, PUSH/POP, Jcc, CMP/TEST), with memory operands
//! displacement-bounded into the mapped data region. Encodings are verified
//! by decoding with iced at the exact emission PC before inclusion.

use crate::case::FuzzCase;
use iced_x86::{
    Code, Decoder, DecoderOptions, Encoder, Instruction as IxInst, MemoryOperand, Register,
};

/// Layout knobs shared with the oracle and the recompiled runner — values
/// must match across all three (the generator just tags the case).
///
/// All regions sit inside the recompiled module's default linear memory
/// (64 pages = 4 MiB, OwnedLinear identity mapping: guest_va == wasm
/// offset), so no memory growth or remapping is needed on the speet side.
pub const CODE_BASE: u64 = 0x0010_0000; // 1 MiB
pub const DATA_BASE: u64 = 0x0020_0000; // 2 MiB
pub const DATA_SIZE: usize = 0x2000;
pub const STACK_BASE: u64 = 0x0030_0000; // 3 MiB
pub const STACK_SIZE: usize = 0x8000;
/// The read-only page lives inside the data region so displacement-bounded
/// stores can reach it: data offsets `[RO_DATA_OFFSET, RO_DATA_OFFSET+RO_DATA_SIZE)`.
/// Unicorn's `mem_protect` requires 4 KiB-aligned address AND size.
pub const RO_DATA_OFFSET: u64 = 0x1000;
pub const RO_DATA_SIZE: u64 = 0x1000;
/// Default per-case instruction budget.
pub const STEP_BUDGET: u64 = 4_000;

/// RNG abstraction so proptest and libfuzzer drivers can both drive the
/// generator without this crate depending on either.
pub trait Rand {
    fn next_u64(&mut self) -> u64;
    /// Uniform in `[0, n)`.
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den as u64) < num as u64
    }
}

/// xorshift64* — tiny, deterministic, replayable from the u64 seed recorded
/// in every artifact.
pub struct SeedRng(pub u64);

impl Rand for SeedRng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Draw an "interesting" 64-bit value: small, bit-boundary, i64::MIN/MAX —
/// biasing toward flag edges (carry, overflow) rather than uniform noise.
fn interesting_value(r: &mut dyn Rand) -> u64 {
    match r.below(8) {
        0 => r.below(16),
        1 => r.below(0x1_0000),
        2 => u64::MAX,
        3 => 0x7FFF_FFFF_FFFF_FFFF, // i64::MAX — overflow edge
        4 => 0x8000_0000_0000_0000, // i64::MIN — overflow edge
        5 => 1u64 << r.below(64),
        6 => 1u64 << r.below(32),
        _ => r.next_u64(),
    }
}

/// The 16 x86-64 GPRs across widths (indices follow the RAX..R15 numbering
/// used everywhere in speet). RSP (index 4) is excluded from destinations so
/// generated cases never clobber the stack pointer mid-sequence —
/// stack-boundedness is enforced at generation time (plan §4.2).
struct Gpr {
    r64: Register,
    r32: Register,
    r16: Register,
    r8: Register,
}

const GPRS: [Gpr; 16] = [
    Gpr { r64: Register::RAX, r32: Register::EAX, r16: Register::AX, r8: Register::AL },
    Gpr { r64: Register::RCX, r32: Register::ECX, r16: Register::CX, r8: Register::CL },
    Gpr { r64: Register::RDX, r32: Register::EDX, r16: Register::DX, r8: Register::DL },
    Gpr { r64: Register::RBX, r32: Register::EBX, r16: Register::BX, r8: Register::BL },
    Gpr { r64: Register::RSP, r32: Register::ESP, r16: Register::SP, r8: Register::SPL },
    Gpr { r64: Register::RBP, r32: Register::EBP, r16: Register::BP, r8: Register::BPL },
    Gpr { r64: Register::RSI, r32: Register::ESI, r16: Register::SI, r8: Register::SIL },
    Gpr { r64: Register::RDI, r32: Register::EDI, r16: Register::DI, r8: Register::DIL },
    Gpr { r64: Register::R8, r32: Register::R8D, r16: Register::R8W, r8: Register::R8L },
    Gpr { r64: Register::R9, r32: Register::R9D, r16: Register::R9W, r8: Register::R9L },
    Gpr { r64: Register::R10, r32: Register::R10D, r16: Register::R10W, r8: Register::R10L },
    Gpr { r64: Register::R11, r32: Register::R11D, r16: Register::R11W, r8: Register::R11L },
    Gpr { r64: Register::R12, r32: Register::R12D, r16: Register::R12W, r8: Register::R12L },
    Gpr { r64: Register::R13, r32: Register::R13D, r16: Register::R13W, r8: Register::R13L },
    Gpr { r64: Register::R14, r32: Register::R14D, r16: Register::R14W, r8: Register::R14L },
    Gpr { r64: Register::R15, r32: Register::R15D, r16: Register::R15W, r8: Register::R15L },
];

/// Destination-eligible registers (all but RSP; RBP included — the frontend
/// models it like any other GPR).
const DST_INDICES: [usize; 15] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn dst(r: &mut dyn Rand) -> &'static Gpr {
    &GPRS[DST_INDICES[r.below(DST_INDICES.len() as u64) as usize]]
}

fn any(r: &mut dyn Rand) -> &'static Gpr {
    &GPRS[r.below(16) as usize]
}

/// Widths the frontend models for ALU/mov. Each width picks the Code value
/// and subregister per operation site.
#[derive(Clone, Copy)]
enum W { B, W, D, Q }

fn width(r: &mut dyn Rand) -> W {
    match r.below(4) {
        0 => W::B,
        1 => W::W,
        2 => W::D,
        _ => W::Q,
    }
}

/// Displacement into the data region, bounded so `disp + size` stays inside
/// the region (plan §4.2). RBX is pre-seeded to `DATA_BASE` so
/// `[rbx + disp]` addressing is safe by construction.
fn data_displacement(r: &mut dyn Rand, size: u64) -> i64 {
    let max = (DATA_SIZE as u64).saturating_sub(size + 16);
    (r.below(max.max(1))) as i64
}

fn mem_operand(r: &mut dyn Rand, access_size: u64) -> MemoryOperand {
    MemoryOperand::new(Register::RBX, Register::None, 1, data_displacement(r, access_size), 1, false, Register::None)
}

/// Encode one instruction at `pc`; verify it round-trips through the decoder
/// at that PC with the same length (plan §4.1).
fn emit(inst: &IxInst, pc: u64) -> Option<Vec<u8>> {
    let mut e = Encoder::new(64);
    e.encode(inst, pc).ok()?;
    let bytes = e.take_buffer();
    if bytes.is_empty() {
        return None;
    }
    // Shingle-hazard filter: the no-gate byte-granular layout creates a slot
    // at every byte offset; an encoding containing a `ret`-like opcode byte
    // (C3/C2/CB) can decode as a *garbage* `ret` at a misaligned slot, whose
    // seal merges an indirect return into the middle of the real
    // instruction stream — executing semantics the oracle never runs. Reject
    // such encodings (plan §4.1's "verified decode" extended to bytes).
    if bytes.contains(&0xC3) || bytes.contains(&0xC2) || bytes.contains(&0xCB) {
        return None;
    }
    let mut dec = Decoder::with_ip(64, &bytes, pc, DecoderOptions::NONE);
    let check = dec.decode();
    if check.is_invalid() || check.len() as usize != bytes.len() {
        return None;
    }
    Some(bytes)
}

/// ALU `op dst, src` code values per width. (reg,rm) forms: dst is the reg
/// operand, src the rm operand — both registers here, so order is direct.
fn alu_code(m: AluOp, w: W) -> Option<Code> {
    use AluOp::*;
    Some(match (m, w) {
        (Add, W::Q) => Code::Add_r64_rm64,
        (Add, W::D) => Code::Add_r32_rm32,
        (Add, W::W) => Code::Add_r16_rm16,
        (Add, W::B) => Code::Add_r8_rm8,
        (Sub, W::Q) => Code::Sub_r64_rm64,
        (Sub, W::D) => Code::Sub_r32_rm32,
        (Sub, W::W) => Code::Sub_r16_rm16,
        (Sub, W::B) => Code::Sub_r8_rm8,
        (And, W::Q) => Code::And_r64_rm64,
        (And, W::D) => Code::And_r32_rm32,
        (And, W::W) => Code::And_r16_rm16,
        (And, W::B) => Code::And_r8_rm8,
        (Or, W::Q) => Code::Or_r64_rm64,
        (Or, W::D) => Code::Or_r32_rm32,
        (Or, W::W) => Code::Or_r16_rm16,
        (Or, W::B) => Code::Or_r8_rm8,
        (Xor, W::Q) => Code::Xor_r64_rm64,
        (Xor, W::D) => Code::Xor_r32_rm32,
        (Xor, W::W) => Code::Xor_r16_rm16,
        (Xor, W::B) => Code::Xor_r8_rm8,
    })
}

#[derive(Clone, Copy)]
enum AluOp { Add, Sub, And, Or, Xor }

fn alu(r: &mut dyn Rand) -> AluOp {
    match r.below(5) {
        0 => AluOp::Add,
        1 => AluOp::Sub,
        2 => AluOp::And,
        3 => AluOp::Or,
        _ => AluOp::Xor,
    }
}

fn mov_code(w: W) -> Code {
    match w {
        W::Q => Code::Mov_r64_rm64,
        W::D => Code::Mov_r32_rm32,
        W::W => Code::Mov_r16_rm16,
        W::B => Code::Mov_r8_rm8,
    }
}

/// Jcc rel8/rel32-to-64 code values, index-matched to `CONDS`.
const JCC_CODES: [Code; 16] = [
    Code::Je_rel8_64, Code::Jne_rel8_64, Code::Jl_rel8_64, Code::Jle_rel8_64,
    Code::Jg_rel8_64, Code::Jge_rel8_64, Code::Jb_rel8_64, Code::Jbe_rel8_64,
    Code::Ja_rel8_64, Code::Jae_rel8_64, Code::Js_rel8_64, Code::Jns_rel8_64,
    Code::Jo_rel8_64, Code::Jno_rel8_64, Code::Jp_rel8_64, Code::Jnp_rel8_64,
];

/// Generate one instruction appended to `code`. Returns `true` on success.
fn gen_one(code: &mut Vec<u8>, r: &mut dyn Rand, pc: u64) -> bool {
    let pick = r.below(400);
    let d = dst(r);
    let s = any(r);

    let built: Option<IxInst> = match pick {
        // ALU reg,reg (all widths — subreg masking is part of the modeled surface)
        0..=14 => alu_code(alu(r), width(r)).map(|c| IxInst::with2(c, d.r64, s.r64).ok()).flatten(),
        // ALU reg,imm8/imm32 — immediates biased to flag edges.
        15..=22 => {
            let op = alu(r);
            let w = width(r);
            let imm: u64 = interesting_value(r);
            let code = match (op, w) {
                (AluOp::Add, W::Q) => Some(Code::Add_rm64_imm8),
                (AluOp::Sub, W::Q) => Some(Code::Sub_rm64_imm8),
                (AluOp::And, W::Q) => Some(Code::And_rm64_imm8),
                (AluOp::Or, W::Q) => Some(Code::Or_rm64_imm8),
                (AluOp::Xor, W::Q) => Some(Code::Xor_rm64_imm8),
                _ => None,
            };
            code.map(|c| {
                let imm8 = (imm & 0xFF) as u32;
                IxInst::with2(c, d.r64, imm8).ok()
            }).flatten()
        }
        // ALU reg,[rbx+disp] — displacement-bounded into the data region.
        // Only ADD/SUB have translated reg,[mem] forms today (AND/OR/XOR
        // reg,[mem] fall through to `unsupported` and would seal the merged
        // straight-line group with `unreachable`, wasting the whole case).
        23..=32 => {
            let op = match r.below(2) {
                0 => AluOp::Add,
                _ => AluOp::Sub,
            };
            alu_code(op, W::Q).and_then(|c| IxInst::with2(c, d.r64, mem_operand(r, 8)).ok())
        }
        // SHL/SHR/SAR r64, imm8
        33..=38 => {
            let c = match r.below(3) {
                0 => Code::Shl_rm64_imm8,
                1 => Code::Shr_rm64_imm8,
                _ => Code::Sar_rm64_imm8,
            };
            IxInst::with2(c, d.r64, (r.below(64) + 1) as u32).ok()
        }
        // NOT r/m64 (the only unary the frontend translates today —
        // INC/DEC/NEG would skip at execution).
        39..=44 => IxInst::with1(Code::Not_rm64, d.r64).ok(),
        // MOV r64, imm64
        45..=51 => IxInst::with2(Code::Mov_r64_imm64, d.r64, interesting_value(r) as i64).ok(),
        // MOV reg,reg (mixed widths — subreg writes)
        52..=58 => {
            let w = width(r);
            let (dc, sc) = match w {
                W::Q => (d.r64, s.r64),
                W::D => (d.r32, s.r32),
                W::W => (d.r16, s.r16),
                W::B => (d.r8, s.r8),
            };
            IxInst::with2(mov_code(w), dc, sc).ok()
        }
        // MOV [rbx+disp],r64 / MOV r64,[rbx+disp]
        59..=66 => {
            let mem = mem_operand(r, 8);
            if r.chance(1, 2) {
                IxInst::with2(Code::Mov_rm64_r64, mem, d.r64).ok()
            } else {
                IxInst::with2(Code::Mov_r64_rm64, d.r64, mem).ok()
            }
        }
        // MOVZX/MOVSX r64, subreg (within-register-file moves)
        67..=69 => {
            let (sub_w, c) = match r.below(4) {
                0 => (W::B, if r.chance(1, 2) { Code::Movzx_r64_rm8 } else { Code::Movsx_r64_rm8 }),
                1 => (W::W, if r.chance(1, 2) { Code::Movzx_r64_rm16 } else { Code::Movsx_r64_rm16 }),
                2 => (W::B, Code::Movsx_r32_rm8),
                _ => (W::W, Code::Movsx_r32_rm16),
            };
            let src = match sub_w {
                W::B => s.r8,
                _ => s.r16,
            };
            let dreg = if c == Code::Movsx_r32_rm8 || c == Code::Movsx_r32_rm16 { d.r32 } else { d.r64 };
            IxInst::with2(c, dreg, src).ok()
        }
        // CMP/TEST r64,r64
        70..=76 => {
            let c = if r.chance(1, 2) { Code::Cmp_rm64_r64 } else { Code::Test_rm64_r64 };
            IxInst::with2(c, d.r64, s.r64).ok()
        }
        // LEA r64,[rbx+disp]
        77 => IxInst::with2(Code::Lea_r64_m, d.r64, mem_operand(r, 8)).ok(),
        // XCHG r64,r64
        78 => IxInst::with2(Code::Xchg_rm64_r64, d.r64, s.r64).ok(),
        // PUSH/POP r64 (bounded: RSP itself is never written by other forms)
        79 => IxInst::with1(Code::Push_r64, d.r64).ok(),
        80 => IxInst::with1(Code::Pop_r64, d.r64).ok(),
        // NOP
        81 => Some(IxInst::with(Code::Nopd)),
        // Jcc — target patched after we know the encoding length below.
        82..=88 => IxInst::with1(JCC_CODES[r.below(16) as usize], CODE_BASE as u32 as i32).ok(),
        // DELIBERATE unsupported candidate: x87 FNOP (0xD9 0xD0). Rare
        // frontend overtranslates — this translates through and only skips
        // if actually executed (plan §1 scope rule).
        300 => {
            if code.len() + 2 <= 0x1000 {
                code.extend_from_slice(&[0xD9, 0xD0]);
                return true;
            }
            return false;
        }
        // Default: MOV r64,r64 filler.
        _ => IxInst::with2(Code::Mov_r64_rm64, d.r64, s.r64).ok(),
    };

    let Some(mut inst) = built else { return false };

    // Patch Jcc target: backward to sequence start (small bounded loop) or
    // forward past the end (clean exit) — the step budget guards loops.
    if matches!(inst.mnemonic(), iced_x86::Mnemonic::Je | iced_x86::Mnemonic::Jne
        | iced_x86::Mnemonic::Jl | iced_x86::Mnemonic::Jle | iced_x86::Mnemonic::Jg
        | iced_x86::Mnemonic::Jge | iced_x86::Mnemonic::Jb | iced_x86::Mnemonic::Jbe
        | iced_x86::Mnemonic::Ja | iced_x86::Mnemonic::Jae | iced_x86::Mnemonic::Js
        | iced_x86::Mnemonic::Jns | iced_x86::Mnemonic::Jo | iced_x86::Mnemonic::Jno
        | iced_x86::Mnemonic::Jp | iced_x86::Mnemonic::Jnp)
    {
        // Encode once to learn the length, then set a real target.
        let probe = emit(&inst, pc).map_or(6, |b| b.len());
        let target = if r.chance(1, 2) { CODE_BASE } else { pc + probe as u64 };
        inst.set_near_branch64(target);
    }

    match emit(&inst, pc) {
        Some(bytes) => {
            if code.len() + bytes.len() > 0x1000 {
                return false;
            }
            code.extend_from_slice(&bytes);
            true
        }
        None => false,
    }
}

/// Generate a complete fuzz case from a seed. Deterministic: same seed,
/// same case — every artifact records its seed for replay.
pub fn generate_case(seed: u64) -> FuzzCase {
    let mut rng = SeedRng(seed);
    let n_target = 8 + rng.below(17) as usize; // 8..=24 instructions
    let mut code: Vec<u8> = Vec::with_capacity(n_target * 4);
    let mut count = 0;
    let mut stalls = 0;
    while count < n_target && stalls < 64 {
        let pc = CODE_BASE + code.len() as u64;
        if gen_one(&mut code, &mut rng, pc) {
            count += 1;
            stalls = 0;
        } else {
            stalls += 1;
        }
    }

    // Terminate every case with `ret` (C3) so control flow reaches the
    // halt sentinel cleanly. Without it, execution falls off the end of the
    // translated set — the reactor seals the trailing slot with a bare
    // `unreachable` (there is no next slot to fall through to), which would
    // misreport every case as executed-unsupported.
    code.push(0xC3);

    // Initial register state.
    let mut regs = crate::case::RegState::default();
    for g in regs.gprs.iter_mut() {
        *g = interesting_value(&mut rng);
    }
    // RBX anchors the data-region base — displacement-bounded addressing
    // depends on this (see `data_displacement`).
    regs.gprs[3] = DATA_BASE;
    regs.rip = CODE_BASE;

    let data = vec![0u8; DATA_SIZE];
    let mut stack = vec![0u8; STACK_SIZE];
    // Seed the halt sentinel 8 bytes below SP (see `FuzzCase::halt_sentinel`)
    // so the final `ret` — the normal sequence end for these crt0-less
    // cases — lands past the translated set and surfaces the register file.
    let sp = STACK_BASE + ((STACK_SIZE as u64) & !0xF) - 8;
    let off = (sp - STACK_BASE) as usize;
    stack[off..off + 8].copy_from_slice(&(CODE_BASE + code.len() as u64).to_le_bytes());
    // RSP is param 4 in the register file (see `X86Recompiler::SP_PARAM_INDEX`).
    regs.gprs[4] = sp;

    FuzzCase {
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
