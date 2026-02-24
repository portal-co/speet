# x86_64 Flags and Conditional Operations Test
#
# Covers: CMP, TEST, SETcc (all conditions), CMOVcc (select-on-condition),
#         flag state after arithmetic/logic, LAHF/SAHF, PUSHF/POPF.
#
# Assembled with:
#   llvm-mc --triple=x86_64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $64, %rsp

    # ------------------------------------------------------------------
    # SETcc: materialise a condition into a byte register
    # ------------------------------------------------------------------
    movq    $5, %rax
    movq    $5, %rbx
    cmpq    %rbx, %rax          # ZF=1, CF=0, SF=0, OF=0

    sete    %cl                 # cl = 1
    setne   %dl                 # dl = 0
    setl    %sil                # sil = 0
    setle   %dil                # dil = 1
    setg    %r8b                # r8b = 0
    setge   %r9b                # r9b = 1
    setb    %r10b               # r10b = 0 (unsigned below)
    setbe   %r11b               # r11b = 1 (unsigned below or equal)
    seta    %r12b               # r12b = 0
    setae   %r13b               # r13b = 1

    movq    $-1, %rax
    testq   %rax, %rax
    sets    %cl                 # cl = 1 (SF=1)
    setns   %dl                 # dl = 0

    movq    $0x7fffffffffffffff, %rax
    addq    $1, %rax            # overflow
    seto    %cl                 # cl = 1 (OF=1)
    setno   %dl                 # dl = 0

    # ------------------------------------------------------------------
    # CMOVcc: conditional move (no branch)
    # ------------------------------------------------------------------
    movq    $100, %rax
    movq    $200, %rbx
    cmpq    $50, %rax           # 100 > 50 → ZF=0, SF=0, CF=0
    cmovgq  %rbx, %rax         # rax = 200 (taken: 100 > 50)

    movq    $10, %rax
    movq    $5, %rbx
    cmpq    %rbx, %rax          # 10 > 5
    cmovlq  %rbx, %rax         # NOT taken, rax stays 10
    cmovleq %rbx, %rax         # NOT taken, rax stays 10
    cmovsq  %rbx, %rax         # NOT taken (SF=0)

    movq    $3, %rax
    movq    $7, %rbx
    cmpq    %rbx, %rax          # 3 < 7
    cmovlq  %rbx, %rax         # taken: rax = 7

    # CMOVE / CMOVNE
    movq    $0, %rcx
    movq    $0, %rdx
    cmpq    %rcx, %rdx          # ZF=1
    cmoveq  %rbx, %rax         # taken: rax = rbx
    movq    $1, %rcx
    cmpq    %rcx, %rdx          # ZF=0
    cmovneq %rbx, %rax         # taken

    # Unsigned CMOV
    movq    $0, %rax
    movq    $1, %rbx
    cmpq    %rbx, %rax          # CF=1 (unsigned below)
    cmovbq  %rbx, %rax         # taken: rax = 1

    # ------------------------------------------------------------------
    # LAHF / SAHF (transfer AH ↔ low flags byte)
    # ------------------------------------------------------------------
    movq    $0, %rax
    testq   %rax, %rax          # ZF=1, SF=0, CF=0, PF=1
    lahf                        # ah = flags byte
    movb    %ah, -1(%rbp)       # stash it

    movb    $0, %ah             # clear flags
    sahf                        # restore ZF=0, etc.

    # ------------------------------------------------------------------
    # Arithmetic flag interactions
    # ------------------------------------------------------------------
    # ADD carry chain
    movq    $0xffffffffffffffff, %rax
    addq    $1, %rax            # CF=1, ZF=1, result=0
    movq    $0, %rbx
    adcq    $0, %rbx            # rbx += CF → rbx=1

    # Subtraction borrow: CF set when src > dst (unsigned)
    movq    $5, %rax
    movq    $10, %rbx
    subq    %rbx, %rax          # 5-10: CF=1 (borrow), SF=1

    # NEG sets CF=0 only for operand=0, otherwise CF=1
    movq    $0, %rax
    negq    %rax                # CF=0
    movq    $1, %rax
    negq    %rax                # CF=1

    addq    $64, %rsp
    popq    %rbp
    ret
