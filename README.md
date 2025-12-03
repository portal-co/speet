# Speet - Fast Static Recompiler

Speet is a fast static recompiler that translates machine code to WebAssembly. It provides architecture-specific translators and a powerful control flow management library.

## Overview

Speet enables efficient translation of native machine code to WebAssembly, supporting multiple architectures through a modular design. The project consists of:

- **yecta** - A WebAssembly control flow reactor library for managing complex control flow during code generation
- **speet-powerpc** - PowerPC to WebAssembly translator
- **speet-riscv** - RISC-V to WebAssembly translator

## Architecture

The recompiler is built on three core components:

### Yecta Control Flow Library

Yecta provides the foundation for generating WebAssembly functions with complex control flow patterns. It manages:

- Control flow graphs with predecessor/successor tracking
- Exception-based non-local control flow
- Conditional and unconditional jumps
- Direct and indirect function calls
- Automatic closure of nested control structures

See the [yecta crate documentation](crates/yecta/README.md) for more details.

### Architecture Translators

Architecture-specific translators convert native instruction sequences to WebAssembly:

- **speet-powerpc** - Translates PowerPC instructions
- **speet-riscv** - Translates RISC-V instructions

Each translator leverages yecta for control flow management while providing architecture-specific instruction semantics.

## Building

Build all crates:

```bash
cargo build
```

Build with optimizations:

```bash
cargo build --release
```

Run tests:

```bash
cargo test
```

## Usage

Speet is designed to be used as a library. Add the appropriate crate to your `Cargo.toml`:

```toml
[dependencies]
yecta = { path = "path/to/speet/crates/yecta" }
# or
speet-powerpc = { path = "path/to/speet/crates/speet-powerpc" }
# or
speet-riscv = { path = "path/to/speet/crates/speet-riscv" }
```

## License

This project is dual-licensed under:

- [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) for open source use
- Proprietary licenses available from Portal Solutions LLC for commercial use

By contributing to this repository, you grant Portal Solutions LLC the right to issue proprietary licenses to your contributions alongside the rights granted by the GNU AGPLv3.

See [COPYING.md](COPYING.md) for full details.

## Contributing

Contributions are welcome! Please ensure your code:

- Builds without warnings (`cargo build`)
- Passes all tests (`cargo test`)
- Follows existing code style and conventions
- Includes appropriate documentation

## Project Structure

```
speet/
├── crates/
│   ├── yecta/           # Control flow reactor library
│   ├── speet-powerpc/   # PowerPC translator
│   └── speet-riscv/     # RISC-V translator
├── Cargo.toml           # Workspace configuration
├── COPYING.md           # License information
└── README.md            # This file
```
