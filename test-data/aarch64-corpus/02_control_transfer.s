// AArch64 Control Transfer Instructions Test
//
// Covers: B (unconditional), CBZ, CBNZ, B.EQ, B.NE, B.LT, B.GE, BL, RET.
// Assembled with:
//   llvm-mc --triple=aarch64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    // ── B: unconditional forward jump ─────────────────────────────────────────
    movz x0, #1
    b    after_skip
    movz x0, #99                   // skipped
after_skip:

    // ── CBZ / CBNZ ────────────────────────────────────────────────────────────
    movz x1, #0
    cbz  x1, zero_taken            // x1 == 0 → taken
    movz x0, #99                   // skipped
zero_taken:
    movz x2, #5
    cbnz x2, nonzero_taken         // x2 != 0 → taken
    movz x0, #99                   // skipped
nonzero_taken:

    // ── B.EQ / B.NE (CMP then branch) ────────────────────────────────────────
    movz x3, #10
    movz x4, #10
    subs xzr, x3, x4               // sets Z=1
    b.eq eq_taken                  // Z==1 → taken
    movz x0, #99                   // skipped
eq_taken:
    movz x5, #7
    movz x6, #3
    subs xzr, x5, x6               // sets Z=0, N=0
    b.ne ne_taken                  // Z==0 → taken
    movz x0, #99                   // skipped
ne_taken:

    // ── B.LT / B.GE ──────────────────────────────────────────────────────────
    movz x7, #2
    movz x8, #5
    subs xzr, x7, x8               // 2-5=-3, N=1, V=0 → N!=V → LT
    b.lt lt_taken
    movz x0, #99                   // skipped
lt_taken:
    movz x9, #10
    movz x10, #3
    subs xzr, x9, x10              // 10-3=7, N=0, V=0 → N==V → GE
    b.ge ge_taken
    movz x0, #99                   // skipped
ge_taken:

    // ── BL / RET ─────────────────────────────────────────────────────────────
    // BL clobbers the link register (x30), so preserve the caller's LR in a
    // spare callee-saved register across the call.  The harness enters with
    // x30 = 0; restoring it before the final RET makes that RET return to the
    // caller (index 0 → unpopulated funcref → clean trap), instead of looping
    // on the stale subroutine return address.
    mov  x19, x30                  // save caller LR
    bl   subroutine                // x30 = &(b done); call subroutine
    b    done                      // subroutine returns here, then jump to done

subroutine:
    movz x11, #42
    ret                            // return to caller via x30 (→ b done)

done:
    mov  x30, x19                  // restore caller LR
    ret                            // return to caller
