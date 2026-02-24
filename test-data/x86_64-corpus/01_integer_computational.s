# x86_64 Integer Computational Instructions Test
#
# Covers: MOV, ADD, SUB, IMUL, AND, OR, XOR, NOT, NEG, INC, DEC,
#         SHL, SHR, SAR, LEA with 64-bit and 32-bit operand sizes.
#
# Assembled with:
#   llvm-mc --triple=x86_64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # MOV: register ← immediate, register ← register
    # ------------------------------------------------------------------
    movq    $0, %rax
    movq    $1, %rbx
    movq    $0x7fffffffffffffff, %rcx
    movq    %rax, %rdx
    movl    $42, %esi
    movl    %esi, %edi

    # ------------------------------------------------------------------
    # ADD / SUB
    # ------------------------------------------------------------------
    movq    $10, %rax
    movq    $20, %rbx
    addq    %rbx, %rax          # rax = 30
    addq    $5,   %rax          # rax = 35
    subq    %rbx, %rax          # rax = 15
    subq    $3,   %rax          # rax = 12

    # 32-bit forms (zero-extend into 64-bit dest)
    movl    $0xffffffff, %eax
    addl    $1, %eax            # eax = 0 (wraps), rax zero-extended

    # ------------------------------------------------------------------
    # IMUL (two-operand and three-operand)
    # ------------------------------------------------------------------
    movq    $7, %rax
    movq    $6, %rbx
    imulq   %rbx, %rax          # rax = 42
    imulq   $3, %rcx, %rdx      # rdx = rcx * 3

    # ------------------------------------------------------------------
    # AND / OR / XOR / NOT / NEG
    # ------------------------------------------------------------------
    movq    $0xf0f0f0f0f0f0f0f0, %rax
    movq    $0x0f0f0f0f0f0f0f0f, %rbx
    andq    %rbx, %rax          # rax = 0
    movq    $0xf0f0f0f0f0f0f0f0, %rax
    orq     %rbx, %rax          # rax = 0xffffffffffffffff
    xorq    %rax, %rax          # rax = 0 (canonical zeroing)
    movq    $0x00ff00ff00ff00ff, %rax
    notq    %rax                # rax = ~0x00ff00ff00ff00ff
    movq    $1, %rax
    negq    %rax                # rax = -1

    # ------------------------------------------------------------------
    # INC / DEC
    # ------------------------------------------------------------------
    movq    $0, %rcx
    incq    %rcx                # rcx = 1
    incq    %rcx                # rcx = 2
    decq    %rcx                # rcx = 1

    # ------------------------------------------------------------------
    # Shifts: SHL, SHR, SAR (immediate and %cl forms)
    # ------------------------------------------------------------------
    movq    $1, %rax
    shlq    $4, %rax            # rax = 16
    shrq    $1, %rax            # rax = 8
    sarq    $1, %rax            # rax = 4
    movb    $3, %cl
    shlq    %cl, %rax           # rax = 32
    shrq    %cl, %rax           # rax = 4
    sarq    %cl, %rax           # rax = 0 (arithmetic, sign = 0)

    # ------------------------------------------------------------------
    # LEA: load effective address arithmetic
    # ------------------------------------------------------------------
    movq    $100, %rbx
    leaq    8(%rbx), %rax       # rax = 108
    leaq    (%rbx,%rbx,2), %rax # rax = 300
    leaq    -4(%rip), %rax      # rip-relative (common in PIC code)

    ret
