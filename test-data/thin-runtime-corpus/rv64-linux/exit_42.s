# RV64 Linux: exit(42) via ecall
# a0=42, a7=93 (exit)
.global _start
_start:
    li a0, 42
    li a7, 93
    ecall
