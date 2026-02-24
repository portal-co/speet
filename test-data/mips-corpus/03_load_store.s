# MIPS32 Load / Store Instructions Test
#
# Covers: LB, LBU, LH, LHU, LW, SB, SH, SW and various offset forms.
#
# Assembled with:
#   llvm-mc --triple=mips-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # Set up a base pointer into the .data section
    # ------------------------------------------------------------------
    lui     $s0, %hi(.Ltest_data)
    addiu   $s0, $s0, %lo(.Ltest_data)

    # ------------------------------------------------------------------
    # LW / SW: 32-bit word
    # ------------------------------------------------------------------
    lw      $t0, 0($s0)         # load first word (0xdeadbeef)
    lw      $t1, 4($s0)         # load second word (0x01020304)
    sw      $t0, 8($s0)         # store t0 → third slot
    lw      $t2, 8($s0)         # reload it

    # ------------------------------------------------------------------
    # LH / LHU: 16-bit halfword (signed / unsigned)
    # ------------------------------------------------------------------
    lh      $t3, 0($s0)         # sign-extend upper halfword of 0xdeadbeef: 0xffffdead
    lhu     $t4, 0($s0)         # zero-extend: 0x0000dead
    lh      $t5, 2($s0)         # lower half of first word: 0xffffbeef
    lhu     $t6, 2($s0)         # zero-extend: 0x0000beef

    # ------------------------------------------------------------------
    # LB / LBU: 8-bit byte (signed / unsigned)
    # ------------------------------------------------------------------
    lb      $t7, 0($s0)         # sign-extend first byte 0xde: 0xffffffde
    lbu     $v0, 0($s0)         # zero-extend: 0x000000de
    lb      $v1, 3($s0)         # last byte of word: 0xef → 0xffffffef
    lbu     $a0, 3($s0)         # 0x000000ef

    # ------------------------------------------------------------------
    # SH: 16-bit store
    # ------------------------------------------------------------------
    addi    $a1, $zero, 0x1234
    sh      $a1, 12($s0)        # store halfword
    lhu     $a2, 12($s0)        # reload

    # ------------------------------------------------------------------
    # SB: 8-bit store
    # ------------------------------------------------------------------
    addi    $a3, $zero, 0x55
    sb      $a3, 14($s0)        # store byte
    lbu     $t0, 14($s0)        # reload

    # ------------------------------------------------------------------
    # Negative offsets
    # ------------------------------------------------------------------
    addiu   $s1, $s0, 16
    lw      $t1, -16($s1)       # same as 0($s0)
    sw      $t1, -4($s1)

    # ------------------------------------------------------------------
    # Max positive/negative 16-bit offsets (boundary values)
    # ------------------------------------------------------------------
    # addiu so we don't fault; just test encoding
    addiu   $s2, $zero, 0
    lw      $t2, 32767($s2)     # max positive offset  (will fault at runtime, tests encoding)
    lw      $t3, -32768($s2)    # max negative offset

    jr      $ra
    nop

.section .data
.Ltest_data:
    .word   0xdeadbeef
    .word   0x01020304
    .word   0x00000000          # slot for store test
    .word   0x00000000
    .word   0x00000000
    .word   0x00000000
    .word   0x00000000
    .word   0x00000000
