Recompilation Strategy & Toolchain

Goals
- Translate entire container filesystems into a single, standalone WASM megabinary.
- Ensure the translation is deterministic and optimized for both microservices and agentic AI.
- Map every execution-capable file (ELF, script) to a hash-addressed entry point in the megabinary.

Megabinary Construction
Instead of separate WASM modules, the toolchain performs "global container linking":
1. **Per-Binary Lifting**: Each ELF binary is lifted to WASM IR.
2. **Library Deduplication**: Shared libraries (libc, libssl) are identified across all binaries in the container. They are lifted once and shared within the megabinary's internal module structure.
3. **Internal Routing**: A global dispatcher is generated. It maps binary hashes to the lifted entry points of the original applications.
4. **Megabinary Bundling**: All lifted code, shared components, and the dispatcher are compiled into a single `.wasm` file.

Content-Addressable Mapping
- For every executable in the original image, a cryptographic hash (e.g., BLAKE3 or SHA-256) is calculated.
- The toolchain generates a `mapping.json` (part of the container manifest) that links `hash -> megabinary_export`.
- If a script (e.g., a shell script) is executed, the interpreter (e.g., `/bin/sh`) is resolved via its hash, and the script is passed as an argument within the WASM environment.

AI Agent Optimizations
- **Pre-warmed Tooling**: Common AI tools (Python, grep, sed) are pre-compiled into the megabinary.
- **Shared Polyfills**: A single WASM-based Python interpreter can be shared by multiple concurrent agent tasks, with isolated memory spaces managed by the vkernel.
- **Rapid Instantiation**: The vkernel can AOT-compile the megabinary once at container start. Launching a "process" (a new WASM instance) becomes a near-zero-latency operation.

Handling Non-Recompilable Binaries
- **Polyfill Injection**: Replace problematic binaries (like Node.js with JIT) with pre-built WASM polyfills (like QuickJS or a JIT-less Node build) directly in the megabinary.
- **Hash-Failure Policy**: Any binary that cannot be recompiled or polyfilled is omitted from the mapping. Attempting to run it results in an immediate hash-lookup failure at the vkernel level, ensuring no "raw" ELF execution.

Toolchain Workflow
1. `scan`: Identify all ELFs and scripts in the source image.
2. `lift`: Convert each ELF to WASM IR.
3. `merge`: Combine lifted IR, resolving shared dependencies and adding the dispatcher.
4. `compile`: AOT-compile the megabinary for the target host architecture.
5. `sign`: Sign the megabinary and the hash-mapping manifest.

Next Steps
- Research WASM "multi-module" or "component model" approaches for efficient megabinary construction.
- Implement the hash-lookup and dispatcher generator.
- Integrate with AI agent runtimes to test tool-invocation latency.
