//! Full-binary recompiler driver.
//!
//! Pipeline (see /Users/g/.claude/plans/plan-to-implement-a-smooth-parrot.md):
//!   host binary --binary-io load--> sections/symbols/relocs/entry
//!     --speet--> WASM megabinary (externals -> ambient, syscalls -> host-ambient)
//!     --wasm-blitz--> machine-code bytes + external relocations
//!     --binary-io write--> ELF/Mach-O .o
//!     --system clang + speet-rt + tunnel dylibs--> final host binary
//!
//! This is the integration crux and is built up across milestones M1-M5. The
//! frontend (load + external-target detection + data + entry) and backend
//! (blitz -> object) live in the `frontend` and `backend` modules.

pub mod backend;
pub mod drive;
pub mod frontend;
pub mod guest_func_catalog;
pub mod instrument;
pub mod plt;

/// Re-exports of the abstracted components this driver is built on.
pub use binary_io;
pub use tunnel;
