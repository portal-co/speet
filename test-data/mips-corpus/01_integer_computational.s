# MIPS32 Integer Computational Instructions Test
#
# Covers: ADD, ADDU, ADDI, ADDIU, SUB, SUBU, AND, OR, XOR, NOR,
#         ANDI, ORI, XORI, LUI, SLL, SRL, SRA, SLLV, SRLV, SRAV,
#         SLT, SLTU, SLTI, SLTIU.
#
# Assembled with:
#   llvm-mc --triple=mips-unknown-elf --filetype=obj
# Note: MIPS is big-endian; instruction words are read as big-endian u32.

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # ADD / ADDU / ADDI / ADDIU
    # ------------------------------------------------------------------
    addi  $t0, $zero, 10        # t0 = 10
    addi  $t1, $zero, 20        # t1 = 20
    add   $t2, $t0,  $t1        # t2 = 30
    addu  $t3, $t0,  $t1        # t3 = 30 (unsigned, no overflow trap)
    addiu $t4, $t0,  -3         # t4 = 7
    addi  $t5, $zero, -1        # t5 = 0xffffffff (-1)

    # ------------------------------------------------------------------
    # SUB / SUBU
    # ------------------------------------------------------------------
    sub   $t6, $t1, $t0         # t6 = 10
    subu  $t7, $t1, $t0         # t7 = 10

    # ------------------------------------------------------------------
    # AND / OR / XOR / NOR
    # ------------------------------------------------------------------
    addi  $s0, $zero, 0x0f
    addi  $s1, $zero, 0xff
    and   $s2, $s0,  $s1        # s2 = 0x0f
    or    $s3, $s0,  $s1        # s3 = 0xff
    xor   $s4, $s0,  $s1        # s4 = 0xf0
    nor   $s5, $s0,  $s1        # s5 = ~0xff = 0xffffff00

    # ------------------------------------------------------------------
    # ANDI / ORI / XORI (zero-extended 16-bit immediate)
    # ------------------------------------------------------------------
    ori   $t0, $zero, 0x00ff    # t0 = 0x00ff
    andi  $t1, $t0,  0x0f0f    # t1 = 0x000f (AND with 16-bit zero-ext)
    xori  $t2, $t0,  0x00ff    # t2 = 0x0000

    # ------------------------------------------------------------------
    # LUI: load upper immediate
    # ------------------------------------------------------------------
    lui   $t3, 0xdead           # t3 = 0xdead0000
    ori   $t3, $t3, 0xbeef      # t3 = 0xdeadbeef

    # ------------------------------------------------------------------
    # Shifts: SLL, SRL, SRA (constant shift amount)
    # ------------------------------------------------------------------
    addi  $t4, $zero, 1         # note: 0x0f0f0f0f doesn't fit ADDI; use LUI+ORI below
    sll   $t5, $t4,  4          # t5 = 16
    srl   $t6, $t5,  1          # t6 = 8
    sra   $t7, $t5,  2          # t7 = 4

    # Arithmetic shift on negative value
    lui   $s0, 0x8000           # s0 = 0x80000000
    sra   $s1, $s0,  1          # s1 = 0xc0000000 (sign-extends)

    # ------------------------------------------------------------------
    # Variable shifts: SLLV, SRLV, SRAV
    # ------------------------------------------------------------------
    addi  $s2, $zero, 3
    addi  $s3, $zero, 0x0f
    sllv  $s4, $s3,  $s2        # s4 = 0x78
    srlv  $s5, $s3,  $s2        # s5 = 0x01
    srav  $s5, $s3,  $s2        # s5 = 0x01 (positive, same as srlv)

    # ------------------------------------------------------------------
    # SLT / SLTU / SLTI / SLTIU
    # ------------------------------------------------------------------
    addi  $t0, $zero, 5
    addi  $t1, $zero, 10
    slt   $t2, $t0,  $t1        # t2 = 1 (5 < 10 signed)
    sltu  $t3, $t0,  $t1        # t3 = 1 (5 < 10 unsigned)
    slt   $t4, $t1,  $t0        # t4 = 0 (10 >= 5)
    slti  $t5, $t0,  7          # t5 = 1 (5 < 7)
    sltiu $t6, $t0,  3          # t6 = 0 (5 >= 3 unsigned)

    # NOR as NOT: nor $rd, $rs, $zero → rd = ~rs
    lui   $t0, 0x0f0f
    ori   $t0, $t0, 0x0f0f      # t0 = 0x0f0f0f0f
    nor   $t1, $t0,  $zero      # t1 = 0xf0f0f0f0

    jr    $ra
    nop
