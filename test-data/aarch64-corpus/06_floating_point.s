// AArch64 Floating-Point Instructions Test
//
// Covers: FMOV (GPR↔FP, immediate), FADD/FSUB/FMUL/FDIV,
//         FMADD/FMSUB, FABS/FNEG/FSQRT, SCVTF/FCVTZS, FCMP, FCSEL.
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── FMOV immediate ─────────────────────────────────────────────────────────
    fmov d0, #1.0                    // d0 = 1.0
    fmov d1, #2.0                    // d1 = 2.0

    // ── FMOV GPR → FP (bitcast) ───────────────────────────────────────────────
    movz x0, #0x3ff0, lsl #48        // x0 = bits of 1.0 in double
    fmov d2, x0                      // d2 = 1.0 (bitcast from GPR)

    // ── FMOV FP → GPR (bitcast) ───────────────────────────────────────────────
    fmov x1, d1                      // x1 = raw bits of 2.0

    // ── Basic FP arithmetic ────────────────────────────────────────────────────
    fadd d3, d0, d1                  // d3 = 3.0
    fsub d4, d1, d0                  // d4 = 1.0
    fmul d5, d0, d1                  // d5 = 2.0
    fdiv d6, d1, d0                  // d6 = 2.0

    // ── FMADD / FMSUB ─────────────────────────────────────────────────────────
    fmov d7, #3.0
    fmadd d8, d0, d1, d7             // d8 = d7 + d0*d1 = 3.0 + 1.0*2.0 = 5.0
    fmsub d9, d0, d1, d7             // d9 = d7 - d0*d1 = 3.0 - 2.0 = 1.0

    // ── FABS / FNEG / FSQRT ───────────────────────────────────────────────────
    fneg d10, d0                     // d10 = -1.0
    fabs d11, d10                    // d11 = 1.0
    fmov d12, #4.0
    fsqrt d13, d12                   // d13 = 2.0

    // ── SCVTF / FCVTZS ────────────────────────────────────────────────────────
    movz x2, #42
    scvtf d14, x2                    // d14 = 42.0
    fcvtzs x3, d14                   // x3  = 42

    // ── FCMP and FCSEL ────────────────────────────────────────────────────────
    fcmp d0, d1                      // compare 1.0 vs 2.0 → N=1 (less than)
    fcsel d15, d0, d1, lt            // d15 = d0=1.0 (LT is true)

    // ── FCMP with zero ────────────────────────────────────────────────────────
    fcmp d10, #0.0                   // compare -1.0 vs 0.0 → N=1

    ret
