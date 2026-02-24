# x86_64 Edge Cases Test
#
# Covers: REX prefixes, 32-bit sub-register implicit zero-extension,
#         NOP variants, two-byte NOPs, XCHG-as-NOP (xchg eax,eax),
#         CDQ/CQO, BSWAP, BSF/BSR, MOVABS, RIP-relative addressing,
#         and zero-operand instructions.
#
# Assembled with:
#   llvm-mc --triple=x86_64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # 32-bit writes zero-extend into 64-bit register
    # ------------------------------------------------------------------
    movq    $0xdeadbeefcafe1234, %rax
    movl    $0x00000042, %eax   # upper 32 bits of rax become 0
    # rax is now 0x0000000000000042

    # ------------------------------------------------------------------
    # REX.W vs no-REX on otherwise identical opcodes
    # ------------------------------------------------------------------
    movl    %eax, %ecx          # 32-bit copy
    movq    %rax, %rcx          # 64-bit copy

    # ------------------------------------------------------------------
    # NOP variants
    # ------------------------------------------------------------------
    nop                         # 1-byte NOP (0x90)
    xchgl   %eax, %eax          # also NOP (2-byte encoding via xchg eax,eax)
    .byte   0x66, 0x90          # 66 NOP (data-size prefix NOP)

    # ------------------------------------------------------------------
    # CDQ / CQO: sign-extend rax into rdx:rax
    # ------------------------------------------------------------------
    movl    $-1, %eax
    cdq                         # edx:eax = sign_extend(eax) → edx = 0xffffffff
    movq    $-1, %rax
    cqo                         # rdx:rax = sign_extend(rax) → rdx = -1

    # ------------------------------------------------------------------
    # BSWAP: byte-reverse a register
    # ------------------------------------------------------------------
    movq    $0x0102030405060708, %rax
    bswapq  %rax                # rax = 0x0807060504030201
    movl    $0x01020304, %ecx
    bswapl  %ecx                # ecx = 0x04030201

    # ------------------------------------------------------------------
    # BSF / BSR: bit scan forward / reverse
    # ------------------------------------------------------------------
    movq    $0x0000000000000080, %rax
    bsfq    %rax, %rbx          # rbx = 7  (lowest set bit index)
    bsrq    %rax, %rcx          # rcx = 7  (highest set bit index)
    movq    $0x0000000000008100, %rax
    bsfq    %rax, %rbx          # rbx = 8
    bsrq    %rax, %rcx          # rcx = 15

    # ------------------------------------------------------------------
    # MOVABS: 64-bit immediate to register (REX.W + B8+rd)
    # ------------------------------------------------------------------
    movabsq $0x123456789abcdef0, %rax
    movabsq $0xffffffffffffffff, %rbx

    # ------------------------------------------------------------------
    # RIP-relative data access
    # ------------------------------------------------------------------
    leaq    .Ldata(%rip), %rsi
    movq    (%rsi), %rax        # load from RIP-relative label
    addq    $8, %rsi
    movq    (%rsi), %rbx

    # ------------------------------------------------------------------
    # IDIV / DIV (64-bit and 32-bit)
    # ------------------------------------------------------------------
    movq    $100, %rax
    cqo                         # sign-extend rax into rdx:rax
    movq    $7, %rcx
    idivq   %rcx                # rax = quotient=14, rdx = remainder=2

    movl    $255, %eax
    cdq
    movl    $10, %ecx
    idivl   %ecx                # eax = 25, edx = 5

    # ------------------------------------------------------------------
    # IMUL one-operand form (rdx:rax ← rax * src)
    # ------------------------------------------------------------------
    movq    $0x100000000, %rax
    movq    $0x100000000, %rcx
    imulq   %rcx                # result in rdx:rax

    jmp     .Ldone

.Ldata:
    .quad   0xdeadbeefcafebabe
    .quad   0x0102030405060708

.Ldone:
    ret
