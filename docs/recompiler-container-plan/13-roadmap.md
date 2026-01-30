30/60/90 Day Roadmap and Next Steps

30-day goals
- Prototype minimal vkernel that supports read/write, sockets, and a small syscall subset.
- Build loader stub that can read an appended WASM trailer and start a WASM runtime.
- Recompile a simple static C "hello world" to WASM, append it to a binary, and run it under the vkernel.

60-day goals
- Expand vkernel to support futex, epoll, signals, and basic thread semantics.
- Implement signing and verification flow for appended artifacts and manifests.
- Prototype a basic recompiler pipeline for dynamic binaries and simple shared libraries.
- Add basic seccomp-like enforcement to deny PROT_EXEC and related syscalls.

90-day goals
- Prototype polyfill for Node (QuickJS-based) and WASM CPython baseline.
- Integrate runtime shim with containerd and provide a local dev CLI.
- Run compatibility tests for NGINX, Redis, and a sample Node app.
- Start a small canary rollout on dedicated nodes and collect metrics.

Longer term
- Harden vkernel, optimize performance, and expand syscall coverage.
- Build robust polyfills for common stacks and provide guidance for porting native modules.
- Integrate attestation into orchestration and SIEM.

Immediate next steps (actionable)
- Create a repository with initial prototype code: loader, vkernel minimal, and a simple lifter.
- Define and publish the ELF trailer format and manifest schema (14-formats.md).
- Start a compatibility test matrix and invite early adopters.

Risks and mitigations
- Compatibility gaps: mitigate by prioritized polyfills and fallbacks.
- Performance: mitigate by selective AOT and hot-path optimization.
- Security vulnerabilities: mitigate with aggressive testing and code review.

Contact and collaboration
- Maintain the docs in this folder as the canonical plan and track progress via issues & PRs for each component.