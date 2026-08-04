# x86-corpus

Minimal i686 (IA-32) instruction sequences for `speet-x86` smoke tests.

## Format

Phase 4 does **not** require full ELF objects. Tests embed raw instruction
bytes directly (see `speet-e2e` `x86_flag_smoke` and `speet-x86` unit tests).

Optional raw dumps may be added here as `.bin` files (contiguous i686 machine
code, no ELF header). Entry EIP is chosen by the test harness (typically
`0x1000`).

## Smoke sequences

| Name | Bytes | Meaning |
|------|-------|---------|
| `mov_ret` | `b8 01 00 00 00 c3` | `mov eax, 1; ret` |
| `call_ret` | `e8 00 00 00 00 c3` | `call +0; ret` |

No REX prefixes — this is 32-bit mode only (`speet-x86`, not `speet-x86_64`).
