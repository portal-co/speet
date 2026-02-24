# x86_64 Load / Store Instructions Test
#
# Covers: MOV (all widths), MOVSX/MOVZX, PUSH/POP, XCHG, CMPXCHG,
#         LEA, stack frame setup/teardown, and basic addressing modes.
#
# Assembled with:
#   llvm-mc --triple=x86_64-unknown-elf --filetype=obj

.globl _start
.section .text

_start:
    # ------------------------------------------------------------------
    # Stack-relative loads/stores (standard function prologue/epilogue)
    # ------------------------------------------------------------------
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $64, %rsp           # 64-byte local frame

    # Store all widths onto the stack
    movb    $0x11,  -1(%rbp)    # byte
    movw    $0x2233, -4(%rbp)   # word
    movl    $0x44556677, -8(%rbp)  # dword
    movq    $0x8899aabbccddeeff, %rax
    movq    %rax, -16(%rbp)     # qword

    # Reload and verify widths
    movb    -1(%rbp),  %al      # byte load
    movw    -4(%rbp),  %ax      # word load
    movl    -8(%rbp),  %eax     # dword load (zero-extends to rax)
    movq    -16(%rbp), %rax     # qword load

    # ------------------------------------------------------------------
    # MOVZX / MOVSX: zero- and sign-extension
    # ------------------------------------------------------------------
    movb    $0x80, -17(%rbp)
    movzbl  -17(%rbp), %eax     # zero-extend byte→dword: eax = 0x00000080
    movsbq  -17(%rbp), %rax     # sign-extend byte→qword: rax = 0xffffff80...
    movw    $0x8001, -20(%rbp)
    movzwl  -20(%rbp), %ecx     # zero-extend word→dword
    movswq  -20(%rbp), %rcx     # sign-extend word→qword
    movl    $0x80000001, -24(%rbp)
    movslq  -24(%rbp), %rdx     # sign-extend dword→qword

    # ------------------------------------------------------------------
    # PUSH / POP
    # ------------------------------------------------------------------
    pushq   $0xcafe
    pushq   %rbx
    pushq   %rcx
    popq    %rcx
    popq    %rbx
    popq    %rax                # rax = 0xcafe

    # ------------------------------------------------------------------
    # Addressing modes: base, base+disp, base+index*scale+disp
    # ------------------------------------------------------------------
    leaq    -64(%rbp), %rsi     # base of local array
    movq    $1, 0(%rsi)
    movq    $2, 8(%rsi)
    movq    $3, 16(%rsi)
    movq    0(%rsi), %rax       # base
    movq    8(%rsi), %rbx       # base + disp
    movq    $1, %rcx
    movq    (%rsi,%rcx,8), %rdx # base + index*8

    # SIB with all components
    leaq    -64(%rbp), %rbx
    movq    $0, %rdi
    movq    (%rbx,%rdi,8), %rax

    # ------------------------------------------------------------------
    # XCHG: atomic-by-default exchange
    # ------------------------------------------------------------------
    movq    $10, %rax
    movq    $20, %rbx
    xchgq   %rax, %rbx         # rax=20, rbx=10

    # ------------------------------------------------------------------
    # CMPXCHG: compare and exchange (non-lock variant for recompiler test)
    # ------------------------------------------------------------------
    movq    $42, %rax           # expected value
    movq    $99, %rbx           # new value
    movq    $42, -32(%rbp)
    cmpxchgq %rbx, -32(%rbp)   # ZF=1 if [mem]==rax; then [mem]=rbx

    # ------------------------------------------------------------------
    # Epilogue
    # ------------------------------------------------------------------
    addq    $64, %rsp
    popq    %rbp
    ret
