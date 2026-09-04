# Testing Guide

---

## Running tests

```
cargo test                  # all Rust unit and integration tests
cargo test -p <crate>       # tests for a single crate
```

There are no JS projects in this repository.

---

## Test crates

- `crates/test/speet-e2e` — end-to-end test harness. Tests live in `tests/` as Rust integration tests; the `lib.rs` is a minimal stub.

---

## Writing tests

- Unit tests: place in the same file under `#[cfg(test)]` or in `tests/` inside the crate.
- Integration tests: add `.rs` files under `crates/test/speet-e2e/tests/` or a dedicated crate.
- For tests that require compiling or running guest binaries, use the `harness/` directory for tooling (build scripts, helper binaries, small runtimes).

---

## harness/

The `harness/` directory holds tooling used by tests:

- `harness/build.sh` — install compilers and build helper binaries (add as needed)
- `harness/run.sh` — execute the harness (add as needed)
- `harness/docker/` — Docker images for test environments (add as needed)

Currently sparse; add to it as new test infrastructure is needed.

---

## What to test

- Instruction correctness: for each architecture frontend, compile a small test binary and compare WASM output against expected values.
- Comparison fuzzing: differential testing of random instructions against Unicorn — see [docs/comparison-fuzzing-plan.md](docs/comparison-fuzzing-plan.md) (planned; scope: RX memory, unsupported instructions/traps/RO-writes ignored).
- Reachability: verify `speet-reach` marks the right slots as reachable given a known CFG.
- Linker two-pass: verify `FuncSchedule` panics on count mismatch and produces correct indices.
- Trap state survival: verify `RopDetectTrap` counter survives across `return_call` chains.
