Recompiler for Entire Containers — Plan

This folder contains a set of Markdown documents describing a practical plan to expand a static recompiler to support entire container images. Each file documents one component or strategy and includes explicit goals and implementation notes.

Files
- 01-overview.md: High level goals and summary
- 02-architecture.md: Top-level architecture and components
- 03-vkernel.md: Shared virtual kernel (vkernel) design and goals
- 04-wasm-appending.md: WASM appending format and loader goals
- 05-recompilation.md: Recompilation strategy and toolchain goals
- 06-polyfills.md: Polyfills for unrecompilable binaries (Node, Python, etc.)
- 07-no-jit-policy.md: Strict no-JIT policy and enforcement
- 08-pipeline.md: Image transformation pipeline and build flow
- 09-runtime-integration.md: Runtime/shim and orchestration integration
- 10-testing.md: Testing, validation, compatibility and security tests
- 11-migration.md: Incremental rollout and migration plan
- 12-threat-model.md: Threat model and security properties
- 13-roadmap.md: 30/60/90 day roadmap and next steps
- 14-formats.md: Suggested ELF trailer format and manifest schema
- 15-implementation-notes.md: Concrete developer API suggestions and files to prototype

Use these docs as the canonical plan for implementation and for tracking next steps and milestones.