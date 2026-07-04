# RISC-V C compiler-output corpus (`test-data/rv-c-corpus/`)

Clang-generated RV32/RV64 guests in the **main repo** (not a submodule). Sibling to the hand-written asm submodule at `test-data/rv-corpus/`.

## Layout

```
rv32/   arith.text.elf, frame.text.elf, *.entry
rv64/   arith.text.elf, frame.text.elf, exit.c (+ exit.elf when link succeeds)
```

## Rebuild

```bash
./compile_corpus.sh
```

Targets:

- rv32: `--target=riscv32-unknown-elf -march=rv32im -mabi=ilp32`
- rv64: `--target=riscv64-unknown-elf -march=rv64im -mabi=lp64`

Same `-O1` / anti-vectorization profile as `test-data/c-corpus/`.

## Tests

```bash
cargo test -p speet-riscv --test rv_c_corpus_tests
```

Uses the shared runtime unreachable-trap harness (`speet-corpus-harness`): translate full slot model → instrument → wasmi → assert no `__speet_unreachable_trap` hits.

RV tests enable **memory64** in the harness to match the corpus WASM memory model.

## vs `rv-corpus` submodule

| | `rv-corpus` | `rv-c-corpus` |
|---|-------------|---------------|
| Source | Hand-written asm (submodule) | Clang C (in-repo) |
| Tests | `rv_corpus_tests` (translate-only) | `rv_c_corpus_tests` (runtime trap) |
| Purpose | ISA extension coverage | Compiler-selected encodings |
