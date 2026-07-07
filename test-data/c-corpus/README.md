# C compiler-output corpus (`test-data/c-corpus/`)

Clang-generated guests at **`-O1`** with vectorization disabled. Multi-TU programs link portable `lib/common/` helpers; per-triple artifacts are indexed in `manifest.toml`.

## Layout

```
lib/common/           # portable C (checksum, buffer, io helpers)
programs/<name>/      # main.c + expected.toml (semantic assertions)
manifest.toml         # committed (program × triple) artifact index
<triple>/             # build outputs per program
```

| Artifact | Use |
|----------|-----|
| `{triple}/{program}.linked.elf` / `.macho` | Run **original** guest (native or emulated) |
| `{triple}/{program}.text.elf` | Recompiler input (`.text` extract) |
| `{triple}/{program}.entry` | `main` offset within `.text` |
| `programs/{name}/expected.toml` | Expected `exit_code`, `main_return`, optional `stdout_sha256` |

Programs: `arith`, `frame`, `pairs` (aarch64), `exit42`, `hello`, `checksum`.

## Rebuild

```bash
./compile_corpus.sh
```

Requires LLVM `clang`, `llvm-objcopy`, `llvm-nm`, `llvm-link`. macOS Mach-O guests need `xcrun --show-sdk-path`. Linux linked guests may be skipped on macOS without a Linux sysroot.

## Recompiler + semantic tests (WASM lane)

Translate every slot, instrument unreachable traps, run under wasmi, assert trap-free **and** expected return/exit/stdout per `expected.toml`:

```bash
cargo test -p speet-x86_64 --test c_corpus_tests
cargo test -p speet-aarch64 --test c_corpus_tests
cargo test -p speet-riscv --test rv_c_corpus_tests   # uses shared lib/programs
```

Harness: `crates/test/speet-corpus-harness/`.

## Original vs recompiled equivalence

Original guests run via **nestable runner paths** (`speet-guest-runner`); recompiled guests run natively on the host via thin runtime. Tests **fail only when all paths are exhausted** (not on first incompatible path).

Path priority (first step): `[Native]` → `[NestedVm, QemuUser]` → `[Blink]` → `[QemuUser]` → `[Blink, QemuUser]` (VM hosts without nested KVM).

```bash
# Optional: provision Blink / qemu-user into ~/.cache/speet/emulators/
scripts/install-corpus-emulators.sh all

cargo test -p speet-runtime --test c_program_equiv_e2e
```

Environment:

| Variable | Effect |
|----------|--------|
| `SPEET_AUTO_INSTALL_EMULATORS=1` | Auto-install missing emulators (default on when `CI=true`) |
| `SPEET_EMULATOR_CACHE` | Override emulator cache root |
| `BLINK` / `SPEET_QEMU_RISCV64` etc. | Pin explicit binaries |

## Thin runtime (libc tunnel)

Only guest `.text` is recompiled; libc bodies are tunnelled at link.

```bash
cargo test -p speet-runtime --test c_corpus_e2e
```
