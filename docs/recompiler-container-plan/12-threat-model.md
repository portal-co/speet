Threat Model and Security Properties

Goals
- Define security objectives for the "Megabinary Hash-Based Execution" model.
- Address specific threats related to agentic AI and multi-tenant sandboxing.

Assumed attacker capabilities
- Attacker can modify the container filesystem (e.g., via a vulnerability in an app or an AI agent "writing" a file).
- Attacker can obtain code execution inside a recompiled WASM instance (userland).
- Attacker can attempt to invoke any binary or script in the container.

Security objectives
- **Integrity of Execution**: Ensure only code that was present at build-time and recompiled into the megabinary can ever be executed.
- **No-JIT Enforcement**: Strictly prevent any runtime code generation (essential for LLM-generated code sandboxing).
- **Tool-Use Isolation**: Prevent an AI agent from "tool-hopping" into unauthorized binaries or scripts.

Controls and mitigations
- **Hash-to-Megabinary Mapping**: Execution is strictly tied to cryptographic file hashes. Even if an attacker overwrites `/usr/bin/ls`, they cannot execute the new version because its hash won't be in the signed `manifest.json`.
- **Signed Megabinary & Manifest**: The entire container execution surface (the WASM blob and the hash-map) is signed by the build pipeline. The vkernel/runtime verifies this before any execution begins.
- **WASM Isolation**: Each executed tool runs in its own isolated WASM memory space, managed by the vkernel.
- **vkernel Syscall Whitelisting**: Syscalls are whitelisted per-binary in the manifest, limiting the blast radius of any single tool's compromise.

Specific Threats & Mitigations
- **Threat**: LLM-generated code tries to execute a malicious shell script it just wrote.
  - **Mitigation**: The execution attempt is intercepted. The hash of the new script is not in the signed `manifest.json`. The vkernel denies execution.
- **Threat**: Exploiting a bug in the WASM engine to gain host access.
  - **Mitigation**: Standard WASM sandboxing + vkernel syscall filtering + host-level seccomp/LSM layers.
- **Threat**: "Living off the land" using original binaries.
  - **Mitigation**: Original binaries are inert; only recompiled WASM code in the megabinary can run. The vkernel doesn't even have a mechanism to run ELFs.

Residual risks
- **Vkernel/WASM Engine Bugs**: Mitigated via fuzzing, AOT-only execution, and minimal privilege for the host-side runtime.
- **Polyfill Gaps**: Bugs in the WASM-based Python or Node interpreters could lead to logic bypasses within the sandbox.

Next steps
- Fuzz the hash-lookup and `execve` interception logic.
- Conduct a security review of the "Global Container Linking" to ensure no cross-app contamination within the megabinary.
