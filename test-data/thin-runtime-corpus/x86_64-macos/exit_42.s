# x86_64 macOS: exit(42) via syscall
.global _start
_start:
    mov $42, %rdi
    mov $0x2000001, %rax
    syscall
