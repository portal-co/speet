WASM Megabinary and Hash-Based Mapping

Goals
- Replace per-binary appending with a single, unified WASM megabinary per container.
- Eliminate loader stubs in favor of a central execution controller (integrated into the vkernel or shim).
- Use cryptographic file hashes to map execution requests to specific code segments within the megabinary.
- Ensure integrity and prevent execution of un-recompiled or unauthorized code.

Megabinary Format
The megabinary is a single standard WASM module containing the combined code of all containerized applications. It includes:
- Global data sections shared across processes (where safe).
- Isolated memory and tables for each logical entry point.
- A "dispatcher" function that routes execution based on a provided entry point ID (the file hash).

Container Manifest (manifest.json)
A central manifest maps original file paths and hashes to megabinary entry points:
```json
{
  "container_id": "sha256:...",
  "megabinary": "container.wasm",
  "binaries": {
    "/usr/bin/ls": {
      "hash": "sha256:abcd...",
      "entry_point": "ls_start"
    },
    "/usr/bin/python3": {
      "hash": "sha256:efgh...",
      "entry_point": "python_start"
    }
  },
  "agent_tools": {
    "search_web": "hash_xyz",
    "write_file": "hash_abc"
  }
}
```

Hash-Based Execution Flow
1. **Execution Request**: The container runtime (or AI agent) attempts to execute `/usr/bin/ls`.
2. **Intercept**: The vkernel or shim intercepts the `execve` syscall.
3. **Hash Validation**:
   - The runtime calculates the hash of the requested file.
   - It compares this hash against the manifest.
   - If the hash is missing or incorrect, the execution is **aborted** immediately.
4. **Megabinary Activation**:
   - The runtime identifies the corresponding entry point (e.g., `ls_start`).
   - A new WASM instance is created from the megabinary.
   - Execution begins at the mapped entry point.

Advantages for Agentic AI
- **Instant Tool Invocation**: No need to load new ELFs or bootstrap a loader for every tool use.
- **Strict Sandboxing**: AI-generated code or tools can only call validated entry points.
- **Resource Pooling**: The megabinary can share polyfills (like a Python interpreter) across multiple agent tasks, reducing memory footprint.
- **Deterministic State**: Snapshotting and restoring agent execution state is simplified with a single WASM memory model.

Security Properties
- **Hash Enforcement**: Prevents "living off the land" attacks using original binaries if they weren't recompiled.
- **Atomic Updates**: Replacing the megabinary and manifest ensures all container binaries are updated/recompiled together.
- **No-JIT Persistence**: The megabinary is entirely AOT-compiled by the host before execution.

Next Steps
- Define the megabinary internal routing mechanism (how multiple binaries coexist in one WASM module).
- Detail the hash verification logic in the vkernel/runtime shim.
- Prototype a build tool that merges multiple WASM modules into one megabinary.
