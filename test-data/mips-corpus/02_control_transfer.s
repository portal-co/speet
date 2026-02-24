# MIPS32 Control Transfer Instructions Test
#
# Covers: J, JAL, JR, JALR, BEQ, BNE, BLEZ, BGTZ, BLTZ, BGEZ,
#         BLTZAL, BGEZAL, and delay-slot NOPs.
#
# Assembled with:
#   llvm-mc --triple=mips-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # J: unconditional jump (26-bit target, same 256MB region)
    # ------------------------------------------------------------------
    j       .Lafter_j
    nop                         # delay slot
.Lafter_j:

    # ------------------------------------------------------------------
    # JAL: jump and link ($ra = PC+8)
    # ------------------------------------------------------------------
    jal     .Lmy_func
    nop                         # delay slot
    j       .Lafter_call
    nop
.Lmy_func:
    addi    $v0, $zero, 42
    jr      $ra
    nop
.Lafter_call:

    # ------------------------------------------------------------------
    # JR: jump register (indirect)
    # ------------------------------------------------------------------
    la      $t0, .Ljr_target    # pseudo: lui+ori
    jr      $t0
    nop
.Ljr_target:

    # ------------------------------------------------------------------
    # JALR: jump and link register
    # ------------------------------------------------------------------
    la      $t1, .Ljalr_func
    jalr    $t1
    nop
    j       .Lafter_jalr
    nop
.Ljalr_func:
    jr      $ra
    nop
.Lafter_jalr:

    # ------------------------------------------------------------------
    # BEQ: branch if equal
    # ------------------------------------------------------------------
    addi    $t0, $zero, 5
    addi    $t1, $zero, 5
    beq     $t0, $t1, .Lbeq_taken
    nop
    j       .Lbeq_skip
    nop
.Lbeq_taken:
.Lbeq_skip:

    # ------------------------------------------------------------------
    # BNE: branch if not equal
    # ------------------------------------------------------------------
    addi    $t2, $zero, 3
    addi    $t3, $zero, 7
    bne     $t2, $t3, .Lbne_taken
    nop
    j       .Lbne_skip
    nop
.Lbne_taken:
.Lbne_skip:

    # ------------------------------------------------------------------
    # BLEZ: branch if less than or equal to zero
    # ------------------------------------------------------------------
    addi    $t4, $zero, -1
    blez    $t4, .Lblez_taken
    nop
    j       .Lblez_skip
    nop
.Lblez_taken:
.Lblez_skip:
    addi    $t4, $zero, 0
    blez    $t4, .Lblez2_taken
    nop
    j       .Lblez2_skip
    nop
.Lblez2_taken:
.Lblez2_skip:

    # ------------------------------------------------------------------
    # BGTZ: branch if greater than zero
    # ------------------------------------------------------------------
    addi    $t5, $zero, 1
    bgtz    $t5, .Lbgtz_taken
    nop
    j       .Lbgtz_skip
    nop
.Lbgtz_taken:
.Lbgtz_skip:

    # ------------------------------------------------------------------
    # BLTZ: branch if less than zero
    # ------------------------------------------------------------------
    addi    $t6, $zero, -5
    bltz    $t6, .Lbltz_taken
    nop
    j       .Lbltz_skip
    nop
.Lbltz_taken:
.Lbltz_skip:

    # ------------------------------------------------------------------
    # BGEZ: branch if greater than or equal to zero
    # ------------------------------------------------------------------
    addi    $t7, $zero, 0
    bgez    $t7, .Lbgez_taken
    nop
    j       .Lbgez_skip
    nop
.Lbgez_taken:
.Lbgez_skip:

    # ------------------------------------------------------------------
    # BLTZAL / BGEZAL: branch and link (save return address in $ra)
    # ------------------------------------------------------------------
    addi    $s0, $zero, -1
    bltzal  $s0, .Lbltzal_target
    nop
    j       .Lafter_bltzal
    nop
.Lbltzal_target:
    jr      $ra
    nop
.Lafter_bltzal:

    addi    $s1, $zero, 1
    bgezal  $s1, .Lbgezal_target
    nop
    j       .Lafter_bgezal
    nop
.Lbgezal_target:
    jr      $ra
    nop
.Lafter_bgezal:

    jr      $ra
    nop
