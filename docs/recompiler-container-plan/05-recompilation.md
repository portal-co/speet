Recompilation Strategy & Toolchain

Goals
- Translate existing ELF/binary artifacts into WASM (WASI+vkernel extensions) or build host-specific AOT native blobs at build time.
- Keep the translation deterministic, auditable, and reproducible.
- Cover common platforms (x86_64, aarch64) and common linking models (static and dynamic).

Analysis phase
- Identify executable files and scripts in the image.
- For each binary:
  - Extract architecture, dynamic dependencies, symbol table, and relocation info.
  - Static syscall usage analysis: heuristically determine the set of syscalls used.
  - Flag binaries that use JIT, self-modifying code, or rely on runtime code generation.

Translation targets
- Preferred: WASM (WASI) module
  - Benefits: portability, sandboxing, and well-defined execution model that avoids runtime code generation.
  - Use cases: C/C++ microservices, many command-line tools, statically linked binaries.

- Alternative: AOT native blob
  - Benefits: better performance for CPU-bound workloads.
  - Drawbacks: host specific, larger attack surface if toolchain has bugs. Must be signed and built reproducibly in pipeline.

Toolchain options
- Use existing tools where possible:
  - Binaryen / LLVM-based approaches for C/C++ → WASM
  - Cranelift or wasmtime toolchain for IR translation where suitable
  - Custom binary lifter that converts x86_64/ARM instructions to WASM IR for userland syscalls
- Build deterministic toolchains and pin versions tightly.

Handling dynamic linking and shared libraries
- Strategy A: Recompile shared libraries as separate WASM modules and provide a loader runtime that links them at module instantiation time.
- Strategy B: Statically link libraries at build time into WASM if possible.
- For glibc-specific features, provide compatibility shims in the vkernel/WASI layer.

Fallbacks and qemu-like mode
- For binaries that cannot be recompiled immediately, provide a controlled fallback that runs them via an emulation layer (qemu user mode) inside vkernel while still enforcing no-JIT and syscall policies.
- Mark fallback containers as lower-trust and prioritize them for recompile later.

Performance tuning
- Use AOT for hotspots; profile-run workloads to determine candidates for AOT.
- Inline essential syscalls into optimized host bridges to reduce cross-boundary overhead.

Reproducibility and signing
- Produce deterministic builds and include build metadata in the manifest.
- Sign recompiled artifacts with pipeline keys and ensure hosts verify signatures at runtime.

Developer ergonomics
- Provide a CLI to emulate the translation locally, run tests, and iterate on polyfills.
- Provide mapping reports that show what syscalls and library functions were translated or polyfilled.

Next steps
- Prototype a binary lifter from a simple x86_64 static binary to WASM and verify I/O and signal behavior.