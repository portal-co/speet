# x86_64 Linux: exit(42) via syscall (arch-native guest; syscall lowering future)
.global _start
_start:
    mov $42, %rdi
    mov $60, %rax
    syscall
