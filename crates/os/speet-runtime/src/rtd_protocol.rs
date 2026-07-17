//! Compatibility shim for `os-daemon-protocol`.
//!
//! `speet-runtime::rtd_protocol` now lives in
//! `os-emulation/crates/daemon/os-daemon-protocol`. This module re-exports
//! the same wire types so existing `@speet` consumers keep compiling.
//!
//! This is protocol version 2: `Request::Obtain`'s `backend` field is now
//! required and honored by the daemon, replacing v1's `host_id` field,
//! which was parsed off the wire but silently ignored server-side.

pub use os_daemon_protocol::*;
