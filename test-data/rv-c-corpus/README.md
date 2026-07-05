# RISC-V C compiler-output corpus (`test-data/rv-c-corpus/`)

Mirrors `test-data/c-corpus/` layout: shared `lib/common/` + `programs/` sources, RV-specific build outputs under `rv32/` and `rv64/`.

## Rebuild

```bash
./compile_corpus.sh
```

Sources live in `../c-corpus/lib/` and `../c-corpus/programs/`; this tree holds `rv32/*.text.elf`, `rv64/*.text.elf`, and `manifest.toml`.

## Tests

```bash
cargo test -p speet-riscv --test rv_c_corpus_tests
```

Equivalence (when `rv64/exit42.linked.elf` is committed):

```bash
cargo test -p speet-runtime --test c_program_equiv_e2e
```

RV Linux guests use the same `PathPlanner` as native c-corpus: `[Native]` on riscv hosts, `[Blink, QemuUser(rv64)]` on VM dev machines.
