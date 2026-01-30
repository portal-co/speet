Megabinary Format and Hash-Mapping Manifest Schema

Goals
- Define the structure of the single WASM megabinary and the manifest that maps binary hashes to its entry points.
- Ensure the format is optimized for content-addressable execution and rapid startup.

Megabinary Structure
The megabinary is a standard WASM module with specific internal layout conventions:
1. **Dispatcher Export**: A main export function (e.g., `_dispatch`) that takes a `target_id` (the binary hash) and arguments.
2. **Internal Module Map**: A set of internal entry points, one for each recompiled application.
3. **Shared Memory/Tables**: A base layer of shared code (libc, shared libraries) that all recompiled apps can link against internally.
4. **Metadata Section**: A custom WASM section containing the container ID and version.

Manifest Schema (manifest.json)
The central manifest is signed and describes the container's entire execution surface:
```json
{
  "container_id": "sha256:...",
  "version": "1.0",
  "megabinary": "container.wasm",
  "vkernel_version": ">=0.2.0",
  "global_policies": {
    "default_syscalls": ["read", "write", "open", "close"],
    "networking": "restricted"
  },
  "binaries": {
    "sha256:abcd...": {
      "path": "/usr/bin/ls",
      "entry_point": "ls_start",
      "overrides": { "syscalls": ["getdents"] }
    },
    "sha256:efgh...": {
      "path": "/usr/bin/python3",
      "entry_point": "python_start",
      "args": ["-u"]
    }
  },
  "agent_tools": {
    "sha256:tool_xyz...": {
      "name": "search_tool",
      "entry_point": "search_start"
    }
  },
  "signatures": {
    "key_id": "build-server-2026",
    "signature": "..."
  }
}
```

Hash-Based Execution Logic
- When a process inside the container calls `execve("/usr/bin/ls", args)`:
  1. The vkernel/shim calculates `hash("/usr/bin/ls")`.
  2. It looks up this hash in the `binaries` map of the `manifest.json`.
  3. If the hash matches `sha256:abcd...`, it maps to `ls_start`.
  4. The runtime then instantiates the megabinary and calls `_dispatch("sha256:abcd...", args)`.
  5. **Critical Security Rule**: If the file at `/usr/bin/ls` has been modified, its hash will change, causing a lookup failure. If the file is not in the manifest, execution is denied.

Advantages for Agentic AI
- **Multi-Tool Fast-Path**: A single AI agent container can contain hundreds of "tools", each mapped by hash. The overhead to switch between running `grep` and a custom Python tool is just a WASM function call.
- **Deterministic Toolchains**: An agent environment is defined entirely by the megabinary and the hash-map.

Versioning and Signatures
- The `manifest.json` and `container.wasm` are signed as a single unit.
- Any change to the container (updating a binary, adding a tool) results in a new megabinary and a new signed manifest.

Next steps
- Implement the "Dispatcher" generation in the recompiler toolchain.
- Standardize the custom WASM section for container metadata.
