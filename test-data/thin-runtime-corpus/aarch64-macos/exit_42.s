# AArch64 macOS: exit(42) via svc
.global _start
_start:
    mov x0, #42
    mov x16, #1
    svc #0
