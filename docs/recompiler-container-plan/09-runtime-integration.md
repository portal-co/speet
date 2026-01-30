Runtime Shim and Orchestration Integration

Goals
- Provide a runtime shim that launches recompiled containers using the single WASM megabinary and hash-based execution model.
- Integrate with containerd, Kubernetes, and AI agent frameworks for rapid, secure execution.
- Ensure that "fail-on-hash-mismatch" is enforced at the earliest possible stage.

Runtime shim responsibilities
- **Manifest Verification**: Read the container's `manifest.json`, verify its signature, and load the hash-to-entry-point map.
- **Megabinary Activation**: Load the AOT-compiled WASM megabinary into the host-side WASM engine.
- **Syscall Interception**: Monitor for `execve` and other execution-related syscalls within the container's namespace.
- **Hash-to-WASM Mapping**: On an `execve` attempt:
  1. Intercept the call.
  2. Compute the hash of the target file.
  3. Look up the entry point in the manifest.
  4. If found, instantiate the WASM module at that entry point.
  5. If not found, **fail the syscall with EACCES or similar**.
- **Resource Management**: Apply cgroups and limits to the WASM instances.

Integration points
1. Containerd / CRI
   - A specialized `io.containerd.recompiler.v1` shim.
   - Bypasses standard ELF execution; the shim itself acts as the "loader" by interacting with the vkernel and WASM engine.

2. AI Agent Runtimes (e.g., AutoGPT, LangGraph)
   - Provide a direct API or CLI for agent frameworks to execute tools within the "Megabinary Container".
   - This bypasses the overhead of standard container starts, allowing tools to be invoked in milliseconds.

3. Kubernetes
   - Use RuntimeClasses to specify that certain workloads should run via the recompiler shim.
   - Enable high-density "Agent Nodes" where many short-lived agent tasks run in a shared WASM environment.

Operational workflows
- **Content-Addressable Security**: Operators can whitelist specific binary hashes across the entire fleet, regardless of file path or container name.
- **Rapid Scaling**: Since the megabinary is already AOT-compiled on the host, starting a new "container" is just a memory instantiation of a WASM module.

Failure modes and mitigation
- **Hash Mismatch**: If an attacker attempts to overwrite a binary or run an unauthorized script, the hash lookup fails, and the execution is blocked. This is a critical defense against many container breakout and persistence techniques.
- **WASM Isolation**: Any crash or exploit within a WASM instance is contained by the WASM sandbox and the vkernel's syscall filtering.

Developer and operator tooling
- `recompiler run <image> <path>`: A CLI tool to simulate the hash lookup and execution of a specific binary within a megabinary-based container.

Next steps
- Implement the hash-lookup and `execve` interception logic in a containerd-compatible shim.
- Integrate the WASM engine with the vkernel for seamless syscall handling.
