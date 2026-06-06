// AArch64 Integer Computational Instructions Test
//
// Covers: ADD (immediate), SUB (immediate), ADD (register), SUB (register),
//         AND, ORR, EOR (shifted register), MOVZ, MOVK, MOVN.
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── MOVZ / MOVK — load 64-bit constants ───────────────────────────────────
    movz x0, #42                   // x0 = 42
    movz x1, #1000                 // x1 = 1000
    movz x2, #0                    // x2 = 0
    movn x3, #0                    // x3 = -1 (all ones)

    // ── ADD immediate ─────────────────────────────────────────────────────────
    movz x4, #10
    add  x5, x4, #5                // x5 = 15
    add  x6, x5, #100              // x6 = 115
    sub  x7, x6, #15               // x7 = 100

    // ── ADD / SUB register (shifted by #0) ────────────────────────────────────
    movz x8, #20
    movz x9, #30
    add  x10, x8, x9               // x10 = 50
    sub  x11, x9, x8               // x11 = 10

    // ── AND / ORR / EOR shifted register ──────────────────────────────────────
    movz x12, #0xff
    movz x13, #0x0f
    and  x14, x12, x13             // x14 = 0x0f
    orr  x15, x12, x13             // x15 = 0xff
    eor  x16, x12, x13             // x16 = 0xf0

    // ── SUBS sets N/Z flags (CMP alias) ───────────────────────────────────────
    movz x17, #5
    movz x18, #5
    subs x19, x17, x18             // x19 = 0, Z=1

    // ── ADDS ──────────────────────────────────────────────────────────────────
    // Load 0x7fff into x20 (a positive number)
    movz x20, #0x7fff
    movz x21, #1
    adds x22, x20, x21             // x22 = 0x8000, N=0 for this range

    ret
