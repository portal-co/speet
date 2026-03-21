# speet

A static binary recompiler written in Rust that translates native machine code and managed bytecode into WebAssembly. The project is in active early development with several architecture frontends in varying states of completion.

## What it does

Speet takes binary code — ELF sections, DEX files, or raw instruction streams — and produces WebAssembly functions, one per input instruction. The translated WASM can then be assembled into a complete module. The core strategy is: map each guest instruction to a separate WASM function, represent guest registers as WASM function parameters (so they survive `return_call` boundaries), and use tail calls for straight-line and backward control flow.

The project is not a JIT. It performs all translation ahead of time and has no runtime code generation step.

## Repository structure

```
crates/
  helper/
    yecta/              # WASM control flow reactor (core library)
    wasm-layout/        # LocalLayout: maps named register groups to wasm locals
    speet-memory/       # Memory emitter, page-table mapper codegen
    speet-ordering/     # Memory ordering helpers (strong vs. relaxed)
    speet-traps/        # Pluggable instruction-level and jump-level trap hooks
    speet-wasm-helpers/ # Pure helpers (e.g. 64x64->128-bit mulh sequences)
  native/
    speet-riscv/        # RISC-V → WASM (most complete frontend)
    speet-x86_64/       # x86-64 → WASM (integer subset)
    speet-mips/         # MIPS → WASM (32-bit and 64-bit)
    speet-powerpc/      # PowerPC → WASM (stub, no translation logic)
  managed/
    speet-dex/          # Dalvik (DEX) → WASM
    dex-bytecode/       # Standalone no_std DEX bytecode parser
    speet-wasm/         # WASM → WASM (address-space remapping frontend)
    speet-object/       # Object model for managed runtimes (heap layout, type hashes)
  os/
    speet-link/         # Multi-binary linker (merges BinaryUnits into one WASM module)
    osctx/              # OS interface traits (syscall dispatch, register/memory access)
```

## Core library: yecta

`yecta` is the control flow reactor that all architecture frontends build on. It manages a control-flow graph where each basic block becomes a WASM function. Key behaviours:

- **Cycle detection**: back edges become tail calls rather than loops.
- **Exception-based non-local returns**: calls are wrapped in `try_table` blocks; callees "return" by throwing a tagged exception, which the caller catches. This allows coroutine-like and longjmp-like patterns to be expressed without restructuring the CFG.
- **Conditional branches**: yecta emits WebAssembly `if`/`else`/`end` blocks and tracks nesting depth.
- **Composable code fragments**: the `Snippet` trait lets instruction emitters inject custom WASM at jump sites (e.g. to compute a table index for an indirect branch).

The library is `no_std` and depends only on `wasm-encoder` and `alloc`.

## Architecture frontends

### speet-riscv (most complete)

Translates RISC-V to WASM. Supported ISA extensions:

- **RV32I / RV64I**: full base integer instruction sets, including runtime-switchable 64-bit mode
- **M**: integer multiply/divide (including `MULH`, `MULHSU`, `MULHU`)
- **A**: atomics — `LR.W`/`SC.W` implemented; AMO instructions stubbed for runtime support
- **F**: single-precision floating-point (all arithmetic, fused multiply-add, sign injection, conversions)
- **D**: double-precision floating-point (same coverage as F)
- **Zicsr**: CSR instructions stubbed — return zero, require runtime CSR management
- **C (compressed)**: handled transparently by the `rv-asm` decoder before translation

Register mapping: x0–x31 → wasm locals 0–31, f0–f31 → locals 32–63, PC → local 64.

Optional features: memory64 addressing (i64 addresses for >4 GB WASM memories), HINT instruction tracking (for the rv-corpus test suite), and callbacks for ECALL, EBREAK, and HINT instructions with access to the instruction sink.

The decoder is `rv-asm` (from `github.com/portal-co/rv-utils`), which is described as panic-free and exhaustively tested.

### speet-x86_64

Translates a subset of x86-64 to WASM using `iced-x86` for decoding. Supported:

- Integer ALU: `ADD`, `SUB`, `IMUL`, `AND`, `OR`, `XOR`, `SHL`, `SHR`, `SAR`, `NOT`, `NEG`, `INC`, `DEC`, `CMP`, `TEST`
- Data movement: `MOV`, `MOVZX`, `MOVSX`, `MOVSXD`, `LEA`, `XCHG`, `PUSH`, `POP`
- Control flow: `Jcc`, `JMP`, `CALL`, `RET` (with optional speculative-call optimisation)
- Memory string operations

Floating-point and SIMD instructions are recognised and classified but emitted as `unreachable` placeholders.

Register mapping: 16 GPRs → wasm locals 0–15 (i64), RIP → local 16, flags (ZF/SF/CF/OF/PF) → locals 17–21 (i32), scratch → locals 22–24.

Speculative calls: when enabled, ABI-conformant `CALL` instructions lower to direct WASM `call` inside a `try_table`; `RET` checks a shadow-stack entry for the fast path.

### speet-mips

Translates MIPS32 and MIPS64 using `rabbitizer` for decoding. Covers the base integer instruction set, load/store, branches/jumps, HI/LO registers, and `SYSCALL`/`BREAK`. Includes load-reserved/store-conditional for atomics.

Register mapping: $0–$31 → locals 0–31, HI/LO → locals 32–33, PC → local 34.

### speet-powerpc

Placeholder crate. No translation logic is implemented. The package exists to reserve the crate name in the workspace.

## Managed-runtime frontends

### speet-dex

Translates Dalvik (Android DEX) bytecode to WASM. Each DEX instruction in each method becomes a separate WASM function. DEX registers are WASM parameters. Uses `dex-parser` for DEX file parsing.

### dex-bytecode

A standalone `no_std`, `forbid(unsafe_code)` Dalvik bytecode parser. Every standard opcode maps to a typed enum variant. Intended to be independently usable outside of speet.

### speet-wasm

A WASM-to-WASM transforming frontend. Parses an existing WASM module and re-emits each function with optional address-space remapping, global-index offsetting, and call-index offsetting. Does not use `yecta` — WASM functions already have structured control flow.

## Cross-cutting infrastructure

### speet-traps

Pluggable hook system installable into any architecture recompiler:

- **InstructionTrap**: fires before each translated instruction. Use cases: context-switch preemption checks, per-instruction counters, coverage bitmaps, instruction filtering.
- **JumpTrap**: fires before each control-flow transfer. Use cases: ROP detection (`RopDetectTrap`), CFI return validation (`CfiReturnTrap`), control-flow logging (`TraceLogTrap`).

Traps can return `TrapAction::Skip` to suppress the instruction or jump and substitute a custom snippet.

### speet-memory

Architecture-agnostic memory emission helpers: address computation, optional virtual-to-physical mapper callbacks, and page-table code generators (single-level and multi-level, 32-bit and 64-bit).

### speet-ordering

Controls how memory instructions are emitted relative to WASM's memory model. `MemOrder::Strong` emits plain eager instructions; `MemOrder::Relaxed` defers stores to control-flow boundaries so the reactor can sink and deduplicate them. `AtomicOpts` independently switches to WASM atomic load/store instructions for shared-memory scenarios.

### speet-link

Links multiple translated binaries (potentially from different architecture frontends) into a single WASM module. Deduplicates function types and accumulates code/type/export sections via `MegabinaryBuilder`. `FuncSchedule` handles two-pass translation: a registration phase that allocates function index ranges before any code is emitted, and an emit phase that fills them in.

### osctx

Traits (`Ctx`, `OS`) for the boundary between generated WASM and a host OS emulation layer. `OS::syscall` and `OS::osfuncall` handle guest system calls and PLT stub calls respectively.

## Status

| Component | Status |
|-----------|--------|
| yecta (reactor) | Functional, tested |
| speet-riscv | Most complete; RV32IMAFD + RV64 + Zicsr stubs |
| speet-x86_64 | Integer subset functional; FP/SIMD stubbed |
| speet-mips | Basic integer and load/store coverage |
| speet-powerpc | Stub — no translation logic |
| speet-dex | In progress |
| dex-bytecode | Standalone parser, `no_std` |
| speet-wasm | WASM-to-WASM remapping functional |
| speet-link | Multi-binary linker infrastructure in place |
| osctx | Trait definitions only |

There is no top-level binary or CLI. The project is a library workspace; consumers integrate individual crates into their own tooling.

## Testing

Tests are spread across individual crates. The RISC-V and MIPS crates have corpus tests that feed pre-assembled ELF sections through the recompiler. The x86-64 crate similarly uses pre-built ELF objects from `test-data/x86_64-corpus/`. An rv-corpus test suite is included as a git submodule (`test-data/rv-corpus`).

```
cargo test -p speet-riscv
cargo test -p speet-x86_64
cargo test -p speet-mips
cargo test -p yecta
```

## Dependencies (notable)

| Crate | Purpose |
|-------|---------|
| `wasm-encoder` / `wasmparser` | WASM code generation and parsing |
| `iced-x86` | x86-64 instruction decoding |
| `rv-asm` (portal-co/rv-utils) | RISC-V instruction decoding |
| `rabbitizer` | MIPS instruction decoding |
| `powerpc` | PowerPC (unused in translation as yet) |
| `dex` (letmutx/dex-parser) | DEX file parsing |
| `wax-core` (portal-co/wax) | Instruction sink abstraction |
| `portal-solutions-asm-semantics` / `asim-core` | Assembly semantics (portal-co) |
| `disarm64` | AArch64 decoding (declared as workspace dep, not yet used by any crate) |

Several dependencies are git-sourced from `github.com/portal-co/` and are not on crates.io.

## License

Dual-licensed: [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) for open source use, or a proprietary license from Portal Solutions LLC for commercial use. Contributors grant Portal Solutions LLC the right to issue proprietary licenses alongside the AGPLv3. See `COPYING.md`.
