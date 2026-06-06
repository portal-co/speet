// AArch64 Load/Store Instructions Test
//
// Covers: STR, STRB, STRH, LDR, LDRB, LDRH, LDRSW.
// Uses memory at address 0x200 (well within the 64-page wasmi heap).
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── Setup scratch base at 0x200 ───────────────────────────────────────────
    movz x0, #0x200

    // ── 64-bit store / load ───────────────────────────────────────────────────
    movz x1, #0xabcd
    str  x1, [x0, #0]              // [0x200] = 0xabcd   (imm12=0, scale=8)
    ldr  x2, [x0, #0]              // x2 = 0xabcd

    // ── 8-bit store / load ────────────────────────────────────────────────────
    movz x3, #0x42
    strb w3, [x0, #8]              // [0x208] = 0x42      (imm12=8, scale=1)
    ldrb w4, [x0, #8]              // x4 = 0x42

    // ── 16-bit store / load ───────────────────────────────────────────────────
    movz x5, #0x1234
    strh w5, [x0, #10]             // [0x20a] = 0x1234    (imm12=5, scale=2)
    ldrh w6, [x0, #10]             // x6 = 0x1234

    // ── 32-bit signed load (LDRSW) ────────────────────────────────────────────
    movz x7, #0x5678
    str  x7, [x0, #16]             // [0x210] = 0x5678    (imm12=2, scale=8)
    ldrsw x8, [x0, #16]            // x8 = 0x5678 sign-extended (imm12=4, scale=4)

    ret
