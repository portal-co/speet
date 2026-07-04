# C compiler-output corpus (`test-data/c-corpus/`)

Clang-generated guests at **`-O1`** with vectorization disabled. Exercises instruction forms hand-written asm corpuses avoid (compiler-chosen `stp`/`ldp`, stack prologues, etc.).

## Layout

| Path | Role |
|------|------|
| `*/arith.text.elf` | Recompiler input (`.text` only, freestanding `-c`) |
| `*/frame.text.elf` | Stack-heavy recompiler input |
| `aarch64-linux/pairs.text.elf` | Struct copy / pair-return (`ldp`/`stp` pressure) |
| `*/arith.entry` etc. | Hex offset of `main` within `.text` (for test entry PC) |
| `*/exit.elf` / `*/exit.macho` | Full libc-linked guests for thin-runtime tunnel e2e |

## Rebuild

```bash
./compile_corpus.sh
```

Requires LLVM `clang`, `llvm-objcopy`, `llvm-nm`. macOS Mach-O guests need `xcrun --show-sdk-path`. Linux `exit.elf` link may be skipped on macOS without a Linux sysroot.

### Flags (documented in script)

```
-O1 -fno-vectorize -fno-slp-vectorize -fno-stack-protector -fno-ident -fno-asynchronous-unwind-tables
```

`.text.elf` objects also use `-ffreestanding -nostdlib` (guest code only). Linked `exit` guests use the platform libc / libSystem.

## Recompiler tests (runtime unreachable-trap policy)

Integration tests translate **every 2/4-byte slot**, instrument WASM so each `unreachable` calls `env.__speet_unreachable_trap(func_idx)`, run under wasmi from `main`, and assert **no trap hits**. There is no `unsupported_insns()` or static reachability scan.

```bash
cargo test -p speet-x86_64 --test c_corpus_tests
cargo test -p speet-aarch64 --test c_corpus_tests
```

Harness: `crates/test/speet-corpus-harness/`.

## Thin runtime (libc tunnel boundary)

Only guest `.text` is recompiled; libc/libSystem bodies are **not** ingested — unresolved symbols are tunnelled at link via `TunneledHostApi` / `default_host_api()`.

```bash
cargo test -p speet-runtime --test c_corpus_e2e
```

## vs asm corpuses

| Corpus | Source | Location |
|--------|--------|----------|
| `x86_64-corpus` / hand asm | Maintainer `.s` | `test-data/x86_64-corpus/` |
| **`c-corpus`** | **Clang `-O1` C** | **`test-data/c-corpus/`** (this tree) |
