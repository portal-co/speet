# AArch64 Linux: exit(42) via svc
.global _start
_start:
    mov x0, #42
    mov x8, #93
    svc #0
