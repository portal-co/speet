Top-level Architecture

Goals
- Define modular components and responsibilities to enable incremental implementation and testing.
- Ensure clear separation between build-time artifacts and runtime enforcement.
- Provide a design that supports auditability, signature-based integrity, and strict policy enforcement.

Components
1. Build/Analysis Pipeline
   - Image extractor: unpacks container layers and enumerates executables and interpreters.
   - Static analyzer: extracts symbols, syscalls, and dependency graph.
   - Recompiler: translates binaries to WASM (WASI+vkernel) or produces AOT native blobs.
   - WASM appender: embeds signed artifacts into binary or replaces with loader stub.
   - Polyfill packager: bundles shims for non-recompilable runtimes.
   - Image reassembler: rebuilds container image with signed manifest and policy.

2. Shared Virtual Kernel (vkernel)
   - Host-side process exposing a Linux-compatible syscall surface.
   - Mediates filesystem, network, and resource access.
   - Enforces seccomp-like policies and prohibits executable memory allocation.

3. Per-binary Loader Stub
   - Small native loader placed as the executable entrypoint.
   - Validates signatures and connects WASM runtime to vkernel.
   - Negotiates capabilities and enforces runtime restrictions.

4. WASM Runtime
   - Interpreter or AOT-only runtime that does not perform JIT at execution time.
   - Provides extended WASI that maps to the vkernel syscall set.

5. Polyfill Layer
   - Replacements for complex runtimes that cannot be straightforwardly recompiled.
   - Includes QuickJS-based Node shims, WASM-built CPython, and libuv/libc wrappers.

6. Container Runtime Shim / Orchestration Integration
   - containerd plugin or runtime shim that launches loader stubs inside containers.
   - Applies host-level seccomp, cgroup, and LSM policies.

Design Principles
- Defense-in-depth: enforce no-JIT at build, loader, vkernel, and host levels.
- Signed artifacts: all runtime code must be signed and verifiable.
- Minimal syscall surface: default-deny policy with explicit allow lists per container.
- Compatibility focus: start with common microservice workloads and expand.
- Observability: provide audit logs and attestations for what code ran and what syscalls were performed.

Next steps
- Document vkernel syscall subset and mapping to WASI (see 03-vkernel.md)
- Define ELF trailer format and manifest schema (see 14-formats.md)
- Prototype loader stub and minimal vkernel (see 15-implementation-notes.md)