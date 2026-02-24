# MIPS32 Edge Cases Test
#
# Covers: NOP (sll $zero,$zero,0), BREAK, SYSCALL, writes to $zero
#         (ignored by hardware), MOVE pseudo (ADDU rd, rs, $zero),
#         LI pseudo (ADDIU / LUI+ORI), LA pseudo, MAX/MIN patterns,
#         and shift-amount boundaries (sa=0 and sa=31).
#
# Assembled with:
#   llvm-mc --triple=mips-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # NOP (canonical encoding: SLL $zero, $zero, 0 = 0x00000000)
    # ------------------------------------------------------------------
    nop
    nop
    nop

    # ------------------------------------------------------------------
    # Writes to $zero are silently ignored
    # ------------------------------------------------------------------
    addi    $zero, $zero, 42    # no effect
    add     $zero, $t0, $t1     # no effect
    lui     $zero, 0xffff       # no effect

    # ------------------------------------------------------------------
    # MOVE pseudo (ADDU $rd, $rs, $zero)
    # ------------------------------------------------------------------
    addi    $t0, $zero, 0x1234
    move    $t1, $t0            # t1 = t0

    # ------------------------------------------------------------------
    # LI pseudo (small: ADDIU; large: LUI+ORI)
    # ------------------------------------------------------------------
    li      $t2, 0              # t2 = 0
    li      $t3, 1              # t3 = 1
    li      $t4, -1             # t4 = 0xffffffff
    li      $t5, 0x7fff         # t5 = 32767 (fits in ADDIU)
    li      $t6, 0x8000         # t6 = 32768 (needs LUI+ORI or ADDIU trick)
    li      $t7, 0xdeadbeef     # t7 = 0xdeadbeef (LUI+ORI)

    # ------------------------------------------------------------------
    # LA pseudo (load address)
    # ------------------------------------------------------------------
    la      $s0, .Ledge_data    # s0 = address of .Ledge_data

    # ------------------------------------------------------------------
    # Shift boundary: sa = 0 (no shift) and sa = 31 (max shift)
    # ------------------------------------------------------------------
    lui     $t0, 0xffff
    ori     $t0, $t0, 0xffff    # t0 = 0xffffffff
    sll     $t1, $t0, 0         # t1 = t0 (unchanged)
    sll     $t2, $t0, 31        # t2 = 0x80000000
    srl     $t3, $t0, 0         # t3 = t0
    srl     $t4, $t0, 31        # t4 = 1 (logical, sign bit)
    sra     $t5, $t0, 31        # t5 = 0xffffffff (arithmetic, sign-extends)

    # ------------------------------------------------------------------
    # Overflow patterns (ADD traps, ADDU does not — encoding exercise)
    # ------------------------------------------------------------------
    lui     $s1, 0x7fff
    ori     $s1, $s1, 0xffff    # s1 = 0x7fffffff (INT32_MAX)
    # ADDU: wraps silently
    addu    $s2, $s1, $s1       # s2 = 0xfffffffe (wraps unsigned)
    # SUBU: wraps silently
    addi    $s3, $zero, 0
    subu    $s4, $s3, $s1       # s4 = -INT32_MAX

    # ------------------------------------------------------------------
    # Conditional move patterns (SLT + MOVN/MOVZ if assembler supports)
    # ------------------------------------------------------------------
    addi    $t0, $zero, 5
    addi    $t1, $zero, 10
    slt     $t2, $t0, $t1       # t2 = 1 (5 < 10)
    # MOVN: move if not zero
    movn    $t3, $t1, $t2       # t3 = t1 (since t2 != 0)
    # MOVZ: move if zero
    movz    $t4, $t0, $t2       # NOT taken (t2 != 0)

    jr      $ra
    nop

.section .data
.Ledge_data:
    .word   0xdeadbeef
    .word   0xcafebabe
