Implementation Notes and Prototype Suggestions

Goals
- Provide concrete files, APIs, and small prototypes to jumpstart implementation.
- Offer developer tools and CLI commands as a base for integration into CI and dev workflows.

Prototype components
1. Loader stub (C)
   - Reads appended trailer from argv[0] (binary path) by seeking to EOF and scanning for MAGIC.
   - Verifies signature using a configured public key file (e.g., /etc/recompiler/keys/pubkeys.json).
   - Connects to vkernel socket (e.g., /run/vkernel/<container-id>.sock) and sends manifest.
   - Initializes tiny WASM runtime (e.g., wasm3 or wasmtime in interpreter/AOT mode) and maps WASI calls to vkernel-proxy via RPC.
   - Transfers control to WASM module and handles exit codes and signals.

2. Minimal vkernel prototype (Go or Rust)
   - Expose a UNIX socket API that accepts syscall requests in a compact binary format.
   - Implement a small subset: open/read/write, socket connect/send/receive, epoll_wait, futex wait/wake, exit.
   - Enforce policy to deny PROT_EXEC and memfd_create with exec bits.

3. Simple recompiler/lifter (prototype)
   - For a start, use clang/LLVM to compile a simple C program to WASM via clang --target=wasm32-wasi.
   - Build a small wrapper to append the WASM blob using the trailer format.

4. CLI tools
   - recompiler inspect <binary|image>
   - recompiler append --blob=module.wasm --manifest=manifest.json <binary>
   - recompiler verify <binary>

Suggested APIs and socket protocol
- vkernel RPC messages (simple JSON over UNIX socket for prototype):
  - {"type":"syscall","name":"open","args":["/etc/hosts",0]} -> {"ret":3}
  - {"type":"signal","sig":9}

- Production: migrate to a compact binary protocol with request IDs and batching.

Development environment
- Provide Docker images for building WASM artifacts and running local vkernel for testing.
- Document keys location and development public keys.

Security
- Run vkernel on a dedicated machine or as a managed service with limited privileges.
- Keep prototype keys separate from production keys; store production keys in KMS.

Next-step checklist for prototype
- Implement loader that can extract and verify trailer; connect to test vkernel and issue a "write" syscall to stdout.
- Implement vkernel server that responds to write syscalls by writing to a host FD passed during setup.
- Recompile "hello world" to wasm32-wasi, append it, and run through loader and vkernel to produce output.

Files to create in prototype repo
- cmd/loader/main.c
- vkernel/main.go (or Rust)
- tools/recompiler_cli/...
- docs/recompiler-container-plan/ (this folder)

If you would like, I can:
- Generate starter code for the loader stub and minimal vkernel prototype.
- Create the CLI tool skeleton and a script to produce an appended artifact for a hello-world WASM module.

Which prototype would you prefer me to generate first?