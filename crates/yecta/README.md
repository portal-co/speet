# Yecta - WebAssembly Control Flow Reactor

Yecta is a `no_std` library for generating WebAssembly functions with complex control flow patterns. It provides a reactive approach to function generation where control flow edges and nested structures are automatically managed.

## Features

- **No standard library dependency** - Works in embedded and constrained environments
- **Type-safe indices** - Uses newtype wrappers for function, tag, table, type, and local indices
- **Control flow graph management** - Automatically tracks predecessors and successors
- **Exception-based escapes** - Implements non-local control flow using WebAssembly exceptions
- **Cycle detection** - Automatically converts cyclic control flow to tail calls
- **Conditional jumps** - Support for conditional branches with automatic if-statement closure
- **Dynamic code generation** - `Snippet` trait for composable code fragments

## Core Types

### Index Types

Yecta uses newtype structs to provide type safety for different kinds of indices:

- `FuncIdx` - WebAssembly function index
- `TagIdx` - WebAssembly exception tag index
- `TableIdx` - WebAssembly table index
- `TypeIdx` - WebAssembly type (function signature) index
- `LocalIdx` - Local variable index within a function

### Reactor

The `Reactor` is the main interface for generating WebAssembly functions:

```rust
use yecta::{Reactor, FuncIdx};
use wasm_encoder::ValType;

let mut reactor = Reactor::default();

// Create a function with 2 i32 locals at control flow distance 0
reactor.next([(2, ValType::I32)].into_iter(), 0);
```

### Target

Specifies where a jump or call should go:

- `Target::Static { func: FuncIdx }` - Direct call to a known function
- `Target::Dynamic { idx: &dyn Snippet }` - Indirect call through a table

### EscapeTag

Configuration for exception-based control flow:

```rust
use yecta::{EscapeTag, TagIdx, TypeIdx};

let escape_tag = EscapeTag {
    tag: TagIdx(0),  // Exception tag index
    ty: TypeIdx(1),  // Function type for the exception signature
};
```

### Pool

Configuration for indirect function calls:

```rust
use yecta::{Pool, TableIdx, TypeIdx};

let pool = Pool {
    table: TableIdx(0),  // Table index for indirect calls
    ty: TypeIdx(1),      // Function type index
};
```

### Snippet

A trait for code fragments that can emit WebAssembly instructions:

```rust
use yecta::Snippet;
use wasm_encoder::Instruction;

struct ConstantSnippet(i32);

impl Snippet for ConstantSnippet {
    fn emit(&self, go: &mut (dyn FnMut(&Instruction<'_>) + '_)) {
        go(&Instruction::I32Const(self.0));
    }
}
```

## Operations

### Creating Functions

Use `next()` to create a new function:

```rust
// Create function with locals: 2 i32s, 1 i64
reactor.next(
    [(2, ValType::I32), (1, ValType::I64)].into_iter(),
    0  // control flow distance
);
```

### Emitting Instructions

Use `feed()` to emit instructions to all active functions:

```rust
use wasm_encoder::Instruction;

reactor.feed(&Instruction::I32Const(42));
reactor.feed(&Instruction::LocalSet(0));
```

### Jumping

Use `jmp()` for unconditional jumps:

```rust
reactor.jmp(FuncIdx(0), 2);  // Jump to function 0 with 2 parameters
```

### Conditional Operations

Use `ji()` for complex control flow with conditions:

```rust
use alloc::collections::BTreeMap;

let fixups = BTreeMap::new();  // No parameter modifications
let condition = Some(&my_condition_snippet as &dyn Snippet);

reactor.ji(
    2,           // 2 parameters
    &fixups,     // Parameter modifications
    Target::Static { func: FuncIdx(1) },
    None,        // Not a call (just a jump)
    pool,
    condition,   // Conditional
);
```

### Calls with Exception Handling

Use `call()` for calls wrapped in try-catch:

```rust
reactor.call(
    Target::Static { func: FuncIdx(2) },
    escape_tag,
    pool,
);
```

### Returns via Exceptions

Use `ret()` to return via exception throw:

```rust
reactor.ret(2, escape_tag);  // Return 2 parameters via exception
```

### Sealing Functions

Use `seal()` to finalize a function:

```rust
use wasm_encoder::Instruction;

reactor.seal(&Instruction::Unreachable);
```

## Control Flow Patterns

### Simple Function

```rust
use yecta::{Reactor, FuncIdx};
use wasm_encoder::{ValType, Instruction};

let mut reactor = Reactor::default();

// Create function with 1 i32 local
reactor.next([(1, ValType::I32)].into_iter(), 0);

// Load parameter and add 1
reactor.feed(&Instruction::LocalGet(0));
reactor.feed(&Instruction::I32Const(1));
reactor.feed(&Instruction::I32Add);

// Return
reactor.seal(&Instruction::Return);
```

### Conditional Jump

```rust
use yecta::{Reactor, Target, FuncIdx};
use wasm_encoder::{ValType, Instruction};

struct ConditionSnippet;
impl yecta::Snippet for ConditionSnippet {
    fn emit(&self, go: &mut (dyn FnMut(&Instruction<'_>) + '_)) {
        go(&Instruction::LocalGet(0));  // Use local 0 as condition
    }
}

let mut reactor = Reactor::default();
reactor.next([(1, ValType::I32)].into_iter(), 0);

let condition = ConditionSnippet;
reactor.ji(
    1,
    &Default::default(),
    Target::Static { func: FuncIdx(1) },
    None,
    Pool { table: TableIdx(0), ty: TypeIdx(0) },
    Some(&condition),
);

reactor.feed(&Instruction::Else);
reactor.feed(&Instruction::End);
```

## Implementation Details

### Control Flow Graph

The reactor maintains a control flow graph where:

- Each function has a set of predecessor functions
- Jumps create edges in this graph
- Cycles are detected and converted to tail calls

### Nested If Statements

The reactor tracks nested if statements to ensure they're properly closed:

- Conditional operations increment the if-statement counter
- The `seal()` operation closes all nested if statements
- All predecessors are traversed to close their if statements

### Exception-Based Returns

Non-local returns are implemented using WebAssembly exceptions:

1. Calls are wrapped in `try_table` blocks
2. The callee can "return" by throwing an exception with the tag
3. The caller catches the exception and continues

This enables efficient implementation of complex control flow patterns like coroutines, generators, and state machines.

## Building

```bash
cargo build
```

Run tests:

```bash
cargo test
```

Generate documentation:

```bash
cargo doc --no-deps --open
```

## License

This crate is part of the Speet project and is dual-licensed under:

- [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)
- Proprietary licenses available from Portal Solutions LLC

See the [main repository](../../COPYING.md) for details.

## Dependencies

- `wasm-encoder` - For WebAssembly code generation
- `alloc` - For heap allocations (no_std compatible)

No standard library required - works in embedded environments.
