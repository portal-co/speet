# MIPS32 Multiply / Divide Instructions Test
#
# Covers: MULT, MULTU, DIV, DIVU, MFHI, MFLO, MTHI, MTLO,
#         and the pseudo MUL instruction (three-register form).
#
# Assembled with:
#   llvm-mc --triple=mips-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # MULT: signed 32×32 → 64-bit result in HI:LO
    # ------------------------------------------------------------------
    addi    $t0, $zero, 7
    addi    $t1, $zero, 6
    mult    $t0, $t1            # HI:LO = 42
    mflo    $t2                 # t2 = 42
    mfhi    $t3                 # t3 = 0

    # Large multiplication
    lui     $t4, 0x0001
    lui     $t5, 0x0001
    mult    $t4, $t5            # HI:LO = 0x0000000100000000
    mfhi    $t6                 # t6 = 1
    mflo    $t7                 # t7 = 0

    # Signed negative × positive
    addi    $s0, $zero, -3
    addi    $s1, $zero, 10
    mult    $s0, $s1            # HI:LO = -30 (signed)
    mflo    $s2                 # s2 = 0xffffffe2 (-30 in 32-bit)
    mfhi    $s3                 # s3 = 0xffffffff (sign-extension word)

    # ------------------------------------------------------------------
    # MULTU: unsigned 32×32 → 64-bit result
    # ------------------------------------------------------------------
    lui     $t0, 0x8000         # t0 = 0x80000000
    lui     $t1, 0x0002         # t1 = 0x00020000
    multu   $t0, $t1            # HI:LO = 0x0001000000000000
    mfhi    $t2                 # t2 = 0x00010000
    mflo    $t3                 # t3 = 0x00000000

    # ------------------------------------------------------------------
    # MUL pseudo (result only in rd, no HI/LO update)
    # ------------------------------------------------------------------
    addi    $t4, $zero, 9
    addi    $t5, $zero, 5
    mul     $t6, $t4, $t5       # t6 = 45

    # ------------------------------------------------------------------
    # DIV: signed 32-bit quotient in LO, remainder in HI
    # ------------------------------------------------------------------
    addi    $t0, $zero, 17
    addi    $t1, $zero, 5
    div     $zero, $t0, $t1     # LO = 3, HI = 2
    mflo    $t2                 # t2 = 3
    mfhi    $t3                 # t3 = 2

    # Negative dividend
    addi    $t4, $zero, -17
    div     $zero, $t4, $t1     # LO = -3, HI = -2
    mflo    $t5
    mfhi    $t6

    # ------------------------------------------------------------------
    # DIVU: unsigned division
    # ------------------------------------------------------------------
    lui     $s0, 0x8000         # s0 = 0x80000000 (large unsigned)
    addi    $s1, $zero, 4
    divu    $zero, $s0, $s1     # LO = 0x20000000, HI = 0
    mflo    $s2
    mfhi    $s3

    # ------------------------------------------------------------------
    # MTHI / MTLO: write HI/LO directly
    # ------------------------------------------------------------------
    addi    $t0, $zero, 0xaa
    mthi    $t0                 # HI = 0xaa
    mfhi    $t1                 # t1 = 0xaa
    addi    $t2, $zero, 0xbb
    mtlo    $t2                 # LO = 0xbb
    mflo    $t3                 # t3 = 0xbb

    jr      $ra
    nop
