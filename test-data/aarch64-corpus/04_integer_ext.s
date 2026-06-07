// AArch64 Integer Extension Instructions Test
//
// Covers: UBFM/SBFM aliases (LSL/LSR/ASR/SXTB/SXTH/SXTW/UXTB/UXTH),
//         register shifts (LSLV/LSRV/ASRV/RORV), UDIV/SDIV,
//         MUL (MADD), SMADDL/UMADDL, ADD with UXTW/SXTW,
//         AND/ORR/EOR with bitmask immediate, CSEL/CSINC/CSINV/CSNEG,
//         ADR, BRK.
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── Shifts via UBFM/SBFM aliases ──────────────────────────────────────────
    movz x0, #0xff
    lsl  x1, x0, #4                  // x1 = 0xff0  (UBFM)
    lsr  x2, x1, #2                  // x2 = 0x3fc  (UBFM)
    asr  x3, x1, #3                  // x3 = sign-extending right shift (SBFM)

    // ── Sign/zero-extend aliases ──────────────────────────────────────────────
    movz x4, #0xffff
    sxtb x5, w4                       // x5 = -1 (sign-extend byte from 0xff)
    sxth x6, w4                       // x6 = -1 (sign-extend halfword from 0xffff)
    movz x7, #0x1234
    uxtb x8, w7                       // x8 = 0x34
    uxth x9, w7                       // x9 = 0x1234

    // ── Register shifts ───────────────────────────────────────────────────────
    movz x10, #1
    movz x11, #8
    lsl  x12, x10, x11                // x12 = 1 << 8 = 256 (LSLV)
    lsr  x13, x12, x11                // x13 = 1 (LSRV)
    asr  x14, x3,  x10               // arithmetic right shift by 1 (ASRV)
    ror  x15, x12, x11               // rotate right (RORV)

    // ── Integer division ──────────────────────────────────────────────────────
    movz x16, #100
    movz x17, #7
    udiv x18, x16, x17               // x18 = 14  (unsigned)
    sdiv x19, x16, x17               // x19 = 14  (signed)

    // ── MUL (MADD with XZR as accumulator) ───────────────────────────────────
    movz x20, #6
    movz x21, #7
    mul  x22, x20, x21               // x22 = 42

    // ── SMADDL/UMADDL ─────────────────────────────────────────────────────────
    movz x23, #3
    smaddl x24, w23, w20, x22        // x24 = 42 + 3*6 = 60  (signed 32-bit mul, 64-bit acc)
    umaddl x25, w23, w20, xzr        // x25 = 3*6 = 18  (unsigned)

    // ── ADD with extended register (UXTW) ─────────────────────────────────────
    movz x26, #1, lsl #16             // x26 = 0x10000
    add  x27, x26, w25, uxtw         // x27 = 0x10000 + 18

    // ── Logical with bitmask immediate ───────────────────────────────────────
    movz x0, #0xffff
    and  x1, x0, #0xff               // x1 = 0xff  (bitmask immediate)
    orr  x2, x0, #0xff00             // x2 = 0xffff | 0xff00 = 0xffff
    eor  x3, x0, #0x00ff             // x3 = 0xffff ^ 0x00ff = 0xff00  (valid bitmask)

    // ── CSEL / CSINC / CSINV / CSNEG ─────────────────────────────────────────
    movz x4, #1
    movz x5, #2
    subs xzr, x4, x5                 // set flags: N=1, Z=0 (1 < 2)
    csel  x6, x4, x5, lt             // x6 = x4=1  (N≠V → LT is true → pick x4)
    csinc x7, x4, x5, lt             // x7 = x4=1  (LT true → x4)
    csinv x8, x4, x5, ge             // x8 = ~x5 = ~2  (GE false → invert x5)
    csneg x9, x4, x5, ge             // x9 = -x5 = -2  (GE false → negate x5)

    // ── ADR ───────────────────────────────────────────────────────────────────
    adr  x10, _start                 // x10 = address of _start

    // ── BRK (always traps, so put it last) ───────────────────────────────────
    // (not executed due to ret before it)
    ret
    brk  #0
