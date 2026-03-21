//! OS context abstraction for speet recompilers.
//!
//! This crate defines the interface between a recompiler's generated WASM and
//! the host operating system or emulation layer.  Two traits sit at different
//! levels of the stack:
//!
//! - [`Ctx`] — per-call register and memory view; passed to every OS callback.
//! - [`OS`] — the OS personality; implements syscall dispatch and OS-managed
//!   function calls.

#![no_std]

use core::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64};

/// Per-call view of the guest CPU state: registers and memory.
///
/// Passed by the recompiler runtime to [`OS`] callbacks on every syscall or
/// OS-managed function call.  Implementations typically wrap the live WASM
/// linear memory and the current register file.
pub trait Ctx {
    /// Read bytes from guest virtual address `addr` into `cell`.
    fn read(&self, addr: NonZeroU64, cell: &mut (dyn ArgCell + '_));
    /// Write bytes from `cell` to guest virtual address `addr`.
    fn write(&mut self, addr: NonZeroU64, cell: &mut (dyn Arg + '_));
    /// Read guest register `idx` into `cell`.
    fn reg(&self, idx: u8, cell: &mut (dyn ArgCell + '_));
    /// Write `cell` into guest register `idx`.
    fn set_reg(&mut self, idx: u8, cell: &mut (dyn Arg + '_));
    /// Perform an indirect call to `addr` and return where execution should
    /// resume.  Returns [`SysResult::Jump`] if the call target is itself an OS
    /// hook that redirects control flow, or [`SysResult::Return`] to fall
    /// through.
    fn jalr(&mut self, addr: NonZeroU64, cell: &mut (dyn Arg + '_)) -> SysResult;
}

/// Result of an OS callback: where should the recompiler go next?
pub enum SysResult {
    /// Resume execution after the call site (normal return).
    Return,
    /// Transfer control to the given guest address (tail-call / longjmp style).
    Jump(NonZeroU64),
}

/// The OS personality: syscall dispatch and OS-managed function calls.
///
/// A single `OS` implementation handles all guest-to-host boundary crossings
/// for a translated binary.  The recompiler runtime calls [`OS::syscall`] for
/// guest `ECALL`/`SYSCALL`/`INT` instructions and [`OS::osfuncall`] for calls
/// to addresses that are owned by the OS layer (e.g. PLT stubs, vDSO entries).
pub trait OS {
    /// Handle a guest syscall.  `ctx` provides access to the guest register
    /// file and memory at the point of the syscall.
    fn syscall(&mut self, ctx: &mut (dyn Ctx + '_)) -> SysResult;
    /// Handle a call to an OS-managed function at `addr` (e.g. a PLT stub).
    fn osfuncall(&mut self, addr: NonZeroU64, ctx: &mut (dyn Ctx + '_)) -> SysResult;
}

/// Mutable, typed slot that can be filled from a byte iterator.
///
/// Used to pass values of statically unknown width from the OS layer into a
/// guest register or memory location via [`Ctx::read`] / [`Ctx::reg`].
pub trait ArgCell {
    /// Fill `self` with bytes from `vals`, interpreting them as little-endian.
    fn fill(&mut self, vals: &mut (dyn Iterator<Item = u8> + '_));
    /// Freeze `self` into a read-only [`Arg`].
    fn done(self) -> impl Arg
    where
        Self: Sized;
}

/// A value produced by the guest or OS that can be read as bytes or as an
/// address.
pub trait Arg {
    /// Interpret the value as a non-null guest virtual address, if it is one.
    fn as_addr(&self) -> Option<NonZeroU64>;
    /// Byte width of this value.
    fn len(&self) -> u64;
}
macro_rules! int_arg{
    ($($t:ty),*) => {
        $(impl Arg for $t {
            fn as_addr(&self) -> Option<NonZeroU64> {
                NonZeroU64::new(*self as u64)
            }
            fn len(&self) -> u64 {
                core::mem::size_of::<$t>() as u64
            }
        }
        impl ArgCell for $t {
            fn fill(&mut self, vals: &mut (dyn Iterator<Item = u8> + '_)) {
                let mut bytes = [0u8; core::mem::size_of::<$t>()];
                for b in bytes.iter_mut() {
                    *b = vals.next().unwrap_or(0);
                }
                *self = <$t>::from_le_bytes(bytes);
            }
            fn done(self) -> impl Arg {
                self
            }
        }
    )*

    };
}
int_arg!(u8, u16, u32, u64);
macro_rules! option_nonzero_arg {
    ($($t:ty),*) => {
        $(impl Arg for Option<$t> {
            fn as_addr(&self) -> Option<NonZeroU64> {
                NonZeroU64::new(self.as_ref().cloned().map(|a| a.get() as u64)?)
            }
            fn len(&self) -> u64 {
                core::mem::size_of::<$t>() as u64
            }
        })*
    };
}
option_nonzero_arg!(NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64);
