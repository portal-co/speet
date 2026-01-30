Shared Virtual Kernel (vkernel)

Goals
- Provide a controlled, Linux-compatible syscall surface for recompiled containers.
- Enforce the "Megabinary Hash Rule": only allow execution of binaries with verified hashes.
- Optimize for high-density AI agent execution and low-latency tool invocation.

High-level design
- Implementation model: host-side vkernel process or runtime shim.
- **Megabinary Integration**: The vkernel is aware of the container's WASM megabinary and its `manifest.json`.
- **Hash-to-Entry Point Router**: When a process calls `execve`, the vkernel intercepts it, calculates the target file's hash, and routes the call to the corresponding WASM entry point in the megabinary.

Responsibilities
- **Hash Verification**: Strictly enforce that any file being executed matches a hash in the signed `manifest.json`. Fail with `EACCES` for any mismatch or unknown file.
- **Megabinary Instance Management**: Efficiently spawn and manage multiple WASM instances from the same megabinary, sharing AOT-compiled code.
- **Syscall Emulation**: Standard Linux syscall support (I/O, sockets, threading, signals) mapped to WASI+vkernel extensions.
- **Agent Resource Pooling**: Allow multiple "tools" or "agents" within the same container to share cached resources (e.g., a pre-loaded Python interpreter state) while maintaining memory isolation.
- **No-JIT Enforcement**: Strictly block any attempt to allocate or modify executable memory. All code execution is confined to the AOT-compiled WASM megabinary.

Performance for Agentic AI
- **Instantaneous `execve`**: Since the megabinary is already loaded and AOT-compiled by the vkernel, an `execve` call to a recompiled tool (like `ls` or a custom script) is transformed into a simple WASM instantiation and jump, taking milliseconds.
- **Snapshot/Restore**: The vkernel can support rapid snapshotting of agent WASM memory to allow pausing and resuming complex AI tasks.

Security Controls
- **Content-Addressable Execution**: Even if an attacker manages to write a file to the container filesystem, they cannot execute it because its hash won't be in the signed manifest.
- **Signed Policy Enforcement**: The vkernel reads syscall whitelists directly from the signed manifest, ensuring that even if the WASM code is compromised, its syscall surface is strictly limited.

Testing and Validation
- Conformance tests for the hash-verification logic: ensure no edge case allows executing unhashed code.
- Latency benchmarks for tool invocation: target <10ms for a full `execve` cycle into a recompiled binary.

Next Actions
- Implement the hash-lookup and `execve` interception in the vkernel prototype.
- Extend the vkernel to support "Megabinary Dispatching" based on the entry point IDs in the manifest.
