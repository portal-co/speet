Implementation Notes and Prototype Suggestions

Goals
- Provide concrete files, APIs, and small prototypes to jumpstart the megabinary and hash-based execution model.
- Optimize for rapid startup (agentic AI) and strict enforcement.

Prototype components
1. Megabinary Dispatcher (WASM)
   - A global WASM module that contains multiple sub-modules (or entry points).
   - Exports a `_dispatch(hash_id, argc, argv)` function.
   - Internally routes execution to the correct pre-compiled code based on `hash_id`.

2. Hash-Based Execution Controller (C/Rust)
   - Integrated into a containerd shim or a standalone runtime.
   - Intercepts `execve` calls.
   - Uses a `BLAKE3` hash of the target path to lookup the entry point in a pre-loaded `manifest.json`.
   - Rejects execution if the hash is unknown or doesn't match the manifest.

3. Minimal vkernel with Megabinary Support (Go or Rust)
   - Expose a UNIX socket API that accepts syscall requests.
   - Specifically handles "WASM instantiation" requests from the execution controller.
   - Shares memory and AOT-compiled code across multiple instances of the same megabinary.

4. Megabinary Builder CLI
   - `recompiler-build --image <oci-image> --output container.wasm --manifest mapping.json`
   - Scans the image, lifts ELFs to WASM, merges them into the megabinary, and generates the hash-mapping manifest.

Suggested APIs and socket protocol
- **Exec Request** (Execution Controller to vkernel):
  - `{"type":"exec","path":"/usr/bin/python3","hash":"sha256:efgh..."}`
  - Returns: `{"status":"ok", "entry_point": "python_start"}` or `{"status":"denied"}`

- **Syscall RPC**:
  - `{"type":"syscall","name":"openat","args":[...],"cid":"container-123"}`

AI Agent Environment Prototype
- Create a megabinary containing:
  - `bash` (recompiled)
  - `python3` (polyfill/recompiled)
  - `custom_agent_tool` (recompiled)
- Demonstrate running the agent tool via its hash with sub-10ms startup time.

Security
- Ensure the `manifest.json` is signed using a build-pipeline key.
- vkernel must verify the signature of both the `manifest.json` and the `container.wasm` before any execution begins.

Next-step checklist for prototype
1. Implement a script that takes three small C programs, compiles them to WASM, and bundles them into a single "dispatcher" module.
2. Implement the hash-verification logic that selects the right entry point from the dispatcher.
3. Integrate the dispatcher with a minimal vkernel to execute a "multi-tool" container.

Files to create in prototype repo
- `src/dispatcher/main.wasm` (template)
- `src/vkernel/router.go`
- `src/builder/megabinary.py`
- `docs/recompiler-container-plan/` (this folder)

If you would like, I can:
- Generate starter code for the WASM dispatcher and the hash-mapping builder.
- Create a skeleton for the `execve` interception shim.

Which prototype would you prefer me to generate first?
