Recompiler for Entire Containers — Plan

This folder contains a set of Markdown documents describing a practical plan to transform entire container images into single, standalone WASM megabinaries. This approach focuses on content-addressable execution, high-density AI agent support, and strict no-JIT enforcement.

Files
- 01-overview.md: High level goals and summary
- 02-architecture.md: Top-level architecture and components
- 03-vkernel.md: Shared virtual kernel (vkernel) design and goals
- 04-wasm-megabinary.md: Single megabinary format and hash-mapping goals
- 05-recompilation.md: Recompilation strategy and toolchain goals
- 06-polyfills.md: Polyfills for unrecompilable binaries (Node, Python, etc.)
- 07-no-jit-policy.md: Strict no-JIT policy and enforcement
- 08-pipeline.md: Image transformation pipeline and build flow
- 09-runtime-integration.md: Runtime/shim and orchestration integration
- 10-testing.md: Testing, validation, compatibility and security tests
- 11-migration.md: Incremental rollout and migration plan
- 12-threat-model.md: Threat model and security properties
- 13-roadmap.md: 30/60/90 day roadmap and next steps
- 14-formats.md: Manifest schema and hash-mapping formats
- 15-implementation-notes.md: Concrete developer API suggestions and files to prototype

Use these docs as the canonical plan for implementation and for tracking next steps and milestones.
