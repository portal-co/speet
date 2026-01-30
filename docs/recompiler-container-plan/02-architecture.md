Top-level Architecture

Goals
- Define modular components to enable incremental implementation.
- Ensure clear separation between build-time artifacts and runtime enforcement.
- Provide a design that supports auditability, hash-based integrity, and strict policy enforcement.
- Optimize for agentic AI: ultra-low overhead, rapid instantiation, and secure multi-tenancy.

Components
1. Build/Analysis Pipeline
   - Image extractor: unpacks container layers and enumerates executables.
   - Static analyzer: extracts symbols, syscalls, and dependency graph.
   - Recompiler: translates binaries to a single, standalone WASM "megabinary".
   - Content Indexer: calculates cryptographic hashes of original binaries and maps them to offsets/entries in the WASM megabinary.
   - Polyfill packager: bundles shims for non-recompilable runtimes into the megabinary.
   - Image reassembler: rebuilds container image with a single WASM artifact and a hash-mapping manifest.

2. Shared Virtual Kernel (vkernel)
   - Host-side process exposing a Linux-compatible syscall surface.
   - Mediates filesystem, network, and resource access.
   - Enforces seccomp-like policies and prohibits executable memory allocation.
   - Directly loads and executes WASM megabinaries, bypassing traditional ELF loading.

3. Hash-Based Execution Controller (replacing Loader Stubs)
   - Integrated into the vkernel or a minimal host-side runtime.
   - Intercepts execution attempts, computes the hash of the requested binary, and looks up the corresponding WASM entry point in the megabinary.
   - Fails immediately if the hash is unknown or the mapping is invalid.

4. WASM Megabinary
   - A single AOT-compiled artifact containing all necessary code for the container.
   - No loader stubs or external dependencies; the runtime jumps directly to the pre-compiled code.
   - Supports internal branching based on the entry point hash.

5. Polyfill Layer
   - Replacements for complex runtimes (QuickJS, WASM-built CPython).
   - Compiled into the megabinary as standard WASM modules.

6. Container Runtime & AI Agent Shim
   - Integration for containerd and AI agent frameworks (e.g., AutoGPT, LangChain tool execution).
   - Provides rapid "fork-like" instantiation of agents into pre-warmed WASM environments.

Design Principles
- Megabinary simplicity: One container = one WASM blob. No complex dynamic loading.
- Content-addressable: Execution is tied to file hashes, not just paths.
- No-JIT by design: Execution is strictly AOT or interpreted WASM.
- Agentic focus: Optimized for thousands of short-lived, high-security AI agent tasks.
- Observability: Audit logs tied to binary hashes for precise attribution.

Next steps
- Document megabinary structure and hash-mapping schema (see 14-formats.md)
- Define vkernel syscall mapping (see 03-vkernel.md)
- Prototype megabinary generator (see 15-implementation-notes.md)
