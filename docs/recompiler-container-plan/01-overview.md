Overview

Goals
- Recompile or wrap every program in a container image so containers can run under a single, host-side enforced, Linux-compatible virtual kernel (vkernel).
- Produce deterministic, auditable runtime artifacts for each binary: signed WASM or AOT blobs appended to the binary.
- Provide polyfills for binaries that can\'t be recompiled (JIT-based VMs like V8/Node.js, complex interpreters).
- Enforce a strict no-JIT / no-executable-memory policy using multiple enforcement layers.
- Integrate with container ecosystems (containerd, Kubernetes) while maintaining strong containment and auditability.

Scope
- The plan targets userland containers (no kernel module changes). It assumes a host-side vkernel process and container runtime shim.
- Not all workloads will be immediately supported; some require polyfills or manual intervention (native drivers, FUSE, special device access). The aim is to support common microservice workloads first (NGINX, Redis, Node/Express, Python/Flask).

Deliverables
- Documentation of architecture and components
- File formats for appended artifacts and manifests
- Prototype loader stub and minimal vkernel implementation
- Recompilation toolchain design and reference implementations
- Polyfill library for common runtimes
- Integration shim for container runtimes

Audience
- System architects and engineers building the recompiler pipeline
- Security engineers auditing the no-JIT enforcement and vkernel
- Developers who will port applications or build polyfills for complex runtimes

Related files in this folder
- 02-architecture.md
- 03-vkernel.md
- 04-wasm-appending.md
- 05-recompilation.md
- 06-polyfills.md
- 07-no-jit-policy.md
- 08-pipeline.md
- 09-runtime-integration.md
- 10-testing.md
- 11-migration.md
- 12-threat-model.md
- 13-roadmap.md
- 14-formats.md
- 15-implementation-notes.md