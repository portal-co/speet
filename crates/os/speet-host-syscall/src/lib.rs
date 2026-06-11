//! Host-ambient syscall dispatch for the full-binary recompiler.
//!
//! Placeholder: to be implemented in M2. This crate will provide a
//! `HostSyscallDispatcher` that, instead of routing guest syscalls to WASI
//! imports (as `speet-linux-wasi` does), emits ambient calls to the host
//! `syscall(2)` wrapper (Linux, "shape A") or to libSystem wrappers (macOS,
//! "shape B"), reusing `speet_syscall::{SyscallTable, SyscallEntry, ParamSource}`.
