Shared Virtual Kernel (vkernel)

Goals
- Provide a controlled, Linux-compatible syscall surface that all recompiled containers use.
- Enforce security policies (no-exec, syscall whitelists, cgroup-like resource limits).
- Support enough Linux semantics to run common userland workloads (threads, futex, epoll, file I/O, sockets, signals).
- Be auditable, debuggable, and incrementally extensible.

High-level design
- Implementation model: user-mode vkernel process running on the host (recommended).
  - Containers connect to vkernel via a per-container UNIX socket or ephemeral IPC channel.
  - vkernel translates guest syscalls to host syscalls while enforcing policies and mapping namespaces.
- Policy model: per-container policy manifest describing allowed syscalls, file system mounts, network egress rules, and resource quotas.

Responsibilities
- Syscall emulation: open, read, write, close, socket, bind, listen, accept, connect, send, recv, poll/epoll, select, getrandom, clock_gettime, futex, nanosleep, signalfd, kill, fork/clone semantics to support threads (user-level threads mapping), and process exit semantics.
- File system mediation: virtual rootfs overlays, readonly layers, controlled host path mounts, symlink policy, and inode namespace mapping.
- Threading & synchronization: futex implementation to support pthreads and glibc synchronization.
- Signal delivery: emulate POSIX signals semantics to processes running in the vkernel.
- Enforcement: block or audit operations that attempt to create executable memory, create memfd with PROT_EXEC, or change protections to allow execution.

Performance considerations
- Hot paths: read/write, epoll_wait, send/recv, futex wake/wait.
- Use shared memory regions for high-throughput I/O and event notification where safe.
- Batch syscalls where possible and safe to reduce IPC overhead.

Security controls
- Deny mmap/mprotect with PROT_EXEC and deny memfd_create with PROT_EXEC.
- Enforce per-container seccomp-like rules derived from policy manifest.
- Validate loader attestations and signed artifacts before mapping code pages.
- Audit and log attempts to use disallowed syscalls; provide alerting hooks.

Extension points
- Allow pluggable backends for networking (native sockets vs. host-proxied) and storage (host FS vs. FUSE-like mediator).
- Provide optional compatibility modes for advanced features under stricter admin approval.

Testing and validation
- Implement a syscall conformance suite (subset of LTP) and fuzz syscall interfaces.
- Test futex and pthread-heavy workloads for correctness.
- Benchmark I/O and event loop workloads and tune shared memory vs. IPC tradeoffs.

Open questions
- How to best represent process/threads mapping (one-to-one OS threads vs. user-mode threading multiplexing)?
- How to support file descriptor passing between processes inside the vkernel while using host resources?
- Policy distribution and secrets: how to securely provision vkernel per-container policies and keys.

Next actions
- Draft initial syscall mapping table to WASI+extensions (see 15-implementation-notes.md).
- Build a minimal vkernel prototype that supports read/write, open, sockets, and exec-like semantics.