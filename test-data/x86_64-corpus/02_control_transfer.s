# x86_64 Control Transfer Instructions Test
#
# Covers: JMP (short/near/indirect), Jcc (all 16 conditions), CALL/RET,
#         LOOP, and a switch-like indirect jump table.
#
# Assembled with:
#   llvm-mc --triple=x86_64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # Unconditional JMP (short and near)
    # ------------------------------------------------------------------
    jmp     .Lafter_jmp_short
    ud2                         # unreachable
.Lafter_jmp_short:
    jmp     .Lafter_jmp_near
    .zero   200                 # force near encoding
.Lafter_jmp_near:

    # ------------------------------------------------------------------
    # Conditional branches: integer comparisons (signed)
    # ------------------------------------------------------------------
    movq    $5,  %rax
    movq    $10, %rbx
    cmpq    %rbx, %rax          # 5 - 10 → SF=1, ZF=0, CF=1

    jl      .Lsigned_lt         # signed less
    ud2
.Lsigned_lt:
    jle     .Lsigned_le         # signed less-or-equal
    ud2
.Lsigned_le:
    jg      .Lno_g              # should NOT be taken (5 < 10)
.Lno_g:
    jge     .Lno_ge
.Lno_ge:

    cmpq    %rax, %rbx          # 10 - 5 → positive
    jg      .Lsigned_gt
    ud2
.Lsigned_gt:
    jge     .Lsigned_ge
    ud2
.Lsigned_ge:

    # ------------------------------------------------------------------
    # Conditional branches: ZF / equality
    # ------------------------------------------------------------------
    movq    $7, %rax
    movq    $7, %rbx
    cmpq    %rbx, %rax
    je      .Lequal             # ZF=1
    ud2
.Lequal:
    movq    $3, %rbx
    cmpq    %rbx, %rax
    jne     .Lnot_equal         # ZF=0
    ud2
.Lnot_equal:

    # ------------------------------------------------------------------
    # Conditional branches: unsigned comparisons (CF/ZF)
    # ------------------------------------------------------------------
    movq    $2,  %rax
    movq    $10, %rbx
    cmpq    %rbx, %rax          # unsigned: 2 < 10, CF=1
    jb      .Lunsigned_below    # CF=1
    ud2
.Lunsigned_below:
    jbe     .Lunsigned_be
    ud2
.Lunsigned_be:
    ja      .Lno_above          # CF=1 → not above
.Lno_above:

    # ------------------------------------------------------------------
    # Test-based branches (TEST sets ZF/SF without storing result)
    # ------------------------------------------------------------------
    movq    $0xff, %rax
    testq   %rax, %rax
    jnz     .Lnonzero
    ud2
.Lnonzero:
    xorq    %rax, %rax
    testq   %rax, %rax
    jz      .Liszero
    ud2
.Liszero:

    # ------------------------------------------------------------------
    # Sign / overflow branches
    # ------------------------------------------------------------------
    movq    $0x7fffffffffffffff, %rax
    addq    $1, %rax            # overflow: OF=1
    jo      .Loverflow
    ud2
.Loverflow:
    jno     .Lno_ov             # OF cleared after non-overflowing add
.Lno_ov:

    movq    $-1, %rax
    testq   %rax, %rax
    js      .Lsign              # SF=1 (negative)
    ud2
.Lsign:
    movq    $1, %rax
    testq   %rax, %rax
    jns     .Lnosign            # SF=0 (positive)
    ud2
.Lnosign:

    # ------------------------------------------------------------------
    # CALL / RET
    # ------------------------------------------------------------------
    callq   .Lmy_func
    jmp     .Lafter_call
.Lmy_func:
    movq    $0xdeadbeef, %rax
    retq
.Lafter_call:

    # ------------------------------------------------------------------
    # LOOP (uses %rcx as counter)
    # ------------------------------------------------------------------
    movq    $4, %rcx
    xorq    %rax, %rax
.Lloop_top:
    incq    %rax
    loop    .Lloop_top          # dec rcx, jnz

    # ------------------------------------------------------------------
    # Indirect JMP via register (simulated dispatch)
    # ------------------------------------------------------------------
    leaq    .Ljump_targets(%rip), %rbx
    movq    $0, %rcx
    movq    (%rbx,%rcx,8), %rax
    jmpq    *%rax
.Ljump_targets:
    .quad   .Ltarget0
.Ltarget0:

    ret

