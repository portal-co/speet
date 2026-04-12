# Container Megabinary Plan

Translate entire container images into a single, standalone, AOT-compiled WASM megabinary.
This doc is the canonical reference for architecture, formats, pipeline, security, and rollout.

---

## Overview

**Core goals:**
- One container = one WASM blob (`container.wasm`) + one signed `manifest.json`.
- Every executable maps to a cryptographic hash (BLAKE3/SHA-256); execution is routed by hash, not path.
- No JIT: all code is statically recompiled or replaced by AOT-compiled polyfills.
- Ultra-low startup latency for agentic AI tool invocation (<10ms `execve` → WASM).

**Non-goals (for now):** native drivers, FUSE, special device access.

---

## Architecture

### 1. Build/Analysis Pipeline
1. **Image extractor** — unpacks OCI layers, enumerates executables and scripts.
2. **Static analyzer** — extracts symbols, syscall requirements, shebang dependencies.
3. **Recompiler** — translates each ELF to WASM IR using `speet-{riscv,x86_64,mips,…}`.
4. **Library deduplicator** — lifts shared libraries (libc, libssl) once; shares internally.
5. **Polyfill packager** — bundles WASM-compiled interpreters for non-recompilable runtimes.
6. **Megabinary linker** — merges all WASM IR + polyfills via `speet-link` into `container.wasm`.
7. **Content indexer** — computes file hashes, generates `manifest.json`.
8. **Signer** — signs `container.wasm` and `manifest.json` as a unit.

### 2. Shared Virtual Kernel (vkernel)
- Host-side process (Rust or Go) exposing a Linux-compatible syscall surface to WASM instances.
- Loads and AOT-compiles the megabinary once; spawning a "process" is a WASM instantiation + jump.
- Intercepts `execve`: compute hash → look up manifest → jump to entry point. Fail with `EACCES` on mismatch.
- Blocks `mmap`/`mprotect`/`memfd_create` with `PROT_EXEC`.
- Enforces per-binary syscall whitelists from the manifest.
- Supports snapshot/restore of WASM memory for agent pause/resume.

### 3. Hash-Based Execution Controller
- Integrated into the vkernel or as a standalone containerd shim.
- On `execve("/usr/bin/ls", args)`:
  1. Hash the file at the requested path.
  2. Look up hash in `manifest.json`.
  3. If found: instantiate megabinary, call `_dispatch(hash, argc, argv)`.
  4. If missing or mismatch: deny execution immediately.

### 4. WASM Megabinary
- Standard WASM module; exports `_dispatch(hash_id: i64, argc: i32, argv: i32)`.
- Internal entry points per recompiled binary (e.g., `ls_start`, `python_start`).
- Shared memory/table base layer for common libraries.
- Custom WASM section: container ID + version metadata.
- AOT-compiled once by the vkernel at container start; subsequent "execve"s are function calls.

### 5. Polyfill Layer
- Replaces JIT-dependent runtimes: Node.js → QuickJS, Python → WASM-compiled CPython/MicroPython.
- Compiled as core components of the megabinary (not separate files).
- Multiple scripts/tools in the original image share a single polyfill entry point.
- Native extensions: attempt recompilation; if impossible, polyfill reports error at runtime.

### 6. Multi-Architecture Dispatch
- Megabinary can contain code from multiple source ISAs (x86-64, RISC-V, ARM64, DEX).
- `_dispatch` identifies the source architecture from the manifest entry and routes accordingly.
- Unified `MegabinaryContext` owns the reactor and `FuncIdx` registry across all architectures.

---

## Manifest Schema (`manifest.json`)

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
      "entry_point": "python_start"
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

Hash is BLAKE3 or SHA-256. Any change to a binary → new hash → full megabinary rebuild + re-sign.

---

## No-JIT Policy

**Enforcement layers (defense in depth):**

| Layer | Mechanism |
|---|---|
| Build-time | Recompiler produces statically verified, signed megabinary. |
| vkernel | Denies `mmap`/`mprotect` with `PROT_EXEC`; WASM engine in AOT/interpreter mode only. |
| Host kernel | seccomp-bpf on vkernel + shim processes; LSM (Landlock/AppArmor) for isolation. |
| Attestation | vkernel provides signed attestation of megabinary hash + manifest before execution. |

AI agents generating code cannot execute it: the script's hash won't be in the signed manifest.
Violations → immediate process termination + structured audit log.

---

## Pipeline CLI (Target)

```
recompiler-build --image <oci-image> --output container.wasm --manifest manifest.json
```

Stages:
1. `scan` — enumerate ELFs and scripts, compute hashes
2. `lift` — recompile each ELF to WASM IR
3. `merge` — global linking: deduplicate libraries, generate dispatcher
4. `compile` — AOT-compile megabinary for target host architecture
5. `sign` — sign megabinary + manifest as a unit

---

## Threat Model

**Assumed attacker:** can modify container filesystem or gain userland code execution in a WASM instance.

| Threat | Mitigation |
|---|---|
| Write + execute a malicious file | Hash won't be in signed manifest → `EACCES`. |
| "Living off the land" with original binaries | vkernel has no ELF loader; only WASM runs. |
| Exploit WASM engine for host access | WASM sandboxing + vkernel syscall filter + host seccomp/LSM. |
| AI agent executes LLM-generated script | Hash not in manifest → denied; agent may only invoke pre-approved tools. |
| Cross-binary contamination in megabinary | Each entry point runs in isolated WASM memory space. |

**Residual risks:** vkernel/WASM engine bugs; polyfill logic bypasses within sandbox.

---

## Phased Rollout

| Phase | Goal |
|---|---|
| 0 — Hello Megabinary | Recompile `ls`, `cat`, `echo` into one megabinary; execute via hash-router. |
| 1 — AI Tool-Sandbox | Agent containers with Python, Bash, Git; demonstrate <10ms tool invocation. |
| 2 — containerd Shim | Hash-interception in a containerd-compatible shim; run "megabinary-fied" OCI images. |
| 3 — Library Deduplication | Deduplicate libc/openssl across all binaries; measure memory savings. |
| 4 — Production Canary | Deploy NGINX, Redis on dedicated WASM nodes in a Kubernetes cluster. |

---

## Speet Integration Points

- **`speet-link`**: `MegabinaryBuilder` already accumulates code/type/export sections. Extend with multi-binary `base_func_offset` registry.
- **`osctx`**: `OS::syscall` / `OS::osfuncall` traits form the vkernel bridge. Needs concrete implementations.
- **`speet-traps`**: `RopDetectTrap` + `CfiReturnTrap` deploy as standard trap hooks in all frontends.
- **Global `FuncIdx` registry**: unified `MegabinaryContext` prevents index collisions across architectures.
- **Shared polyfill linking**: use `yecta`'s `Pool` + `Target::Static` to wire calls from recompiled code directly to polyfill function indices.

---

## Immediate Next Steps

1. Define `_dispatch` routing format and internal megabinary layout.
2. Implement `manifest.json` generation in `speet-link` / build tooling.
3. Prototype hash-interception shim for `execve`.
4. Build Phase 0: three small C programs → single megabinary → hash-dispatch demo.
