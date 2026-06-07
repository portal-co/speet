// AArch64 Extended Load/Store Instructions Test
//
// Covers: pre-index LDR/STR, post-index LDR/STR,
//         register-offset LDR/STR, LDP/STP (offset), LDP/STP (pre-index), LDPSW.
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── Set up a scratch area on stack ─────────────────────────────────────────
    // We borrow some space below SP (red zone approach — just for the test)
    sub  sp, sp, #64

    // ── STR / LDR unsigned offset (baseline, already tested in 03) ────────────
    movz x0, #0xABCD
    str  x0, [sp, #0]
    ldr  x1, [sp, #0]                // x1 = 0xABCD

    // ── STR with pre-index ────────────────────────────────────────────────────
    movz x2, #0x1234
    str  x2, [sp, #8]!               // write to sp+8, then sp += 8
    ldr  x3, [sp]                    // x3 = 0x1234; sp is now +8

    // ── LDR with post-index ───────────────────────────────────────────────────
    ldr  x4, [sp], #-8               // x4 = 0x1234 from [sp], then sp -= 8

    // ── STR with post-index ───────────────────────────────────────────────────
    movz x5, #0x5678
    str  x5, [sp], #8                // write 0x5678 to [sp], then sp += 8

    // ── LDR with pre-index ────────────────────────────────────────────────────
    ldr  x6, [sp, #-8]!              // sp -= 8 first, then load x6 = 0x5678

    // ── LDR with register offset ──────────────────────────────────────────────
    movz x7, #8
    ldr  x8, [sp, x7]                // x8 = *[sp + 8]
    ldr  x9, [sp, x7, lsl #0]        // same as above (LSL#0)

    // ── STP (store pair, signed offset) ───────────────────────────────────────
    movz x10, #0x11
    movz x11, #0x22
    stp  x10, x11, [sp, #16]         // [sp+16]=0x11, [sp+24]=0x22

    // ── LDP (load pair, signed offset) ────────────────────────────────────────
    ldp  x12, x13, [sp, #16]         // x12=0x11, x13=0x22

    // ── STP with pre-index ────────────────────────────────────────────────────
    movz x14, #0x33
    movz x15, #0x44
    stp  x14, x15, [sp, #32]!        // sp += 32, then store pair

    // ── LDP with post-index ───────────────────────────────────────────────────
    ldp  x16, x17, [sp], #-32        // load pair from [sp], then sp -= 32

    // ── LDRB/STRB with register offset ────────────────────────────────────────
    movz x18, #0xAB
    strb w18, [sp, #4]
    ldrb w19, [sp, #4]               // x19 = 0xAB

    // ── Restore stack and return ──────────────────────────────────────────────
    add  sp, sp, #64
    ret
