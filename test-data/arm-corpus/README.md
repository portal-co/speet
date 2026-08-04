# arm-corpus

Minimal AArch32 (ARMv7-A / A32) instruction sequences for `speet-arm` smoke tests.

## Format

Phase 4 does **not** require full ELF objects. Tests embed little-endian A32
instruction words directly (see `speet-e2e` `arm_flag_smoke` and
`speet-arm` unit tests).

Optional raw dumps may be added here as `.bin` files (contiguous A32
instructions, no ELF header). Entry PC is chosen by the test harness
(typically `0x1000`).

## Smoke sequences

| Name | Bytes (LE words) | Meaning |
|------|------------------|---------|
| `mov_bx` | `e3a00001 e12fff1e` | `mov r0, #1; bx lr` |
| `bl_self` | `eb000000` | `bl +0` (link + call self+8) |

## Thumb-2

Thumb decode is stubbed in `speet-arm` (A32 primary). Thumb corpora can land
once the Thumb path is wired.
