//! Shared Unix → WASI preview1 handlers as a linkable WASM module.
//!
//! Scratch iovec marshalling uses [`__speet_host_mem`] imports; parent adapter
//! builds lower them to multi-memory ops before embed.
//!
//! OS-specific syscall-number dispatch lives in the arch callback (not here),
//! so one guest binary serves Linux and Darwin.

#![no_std]

use core::panic::PanicInfo;

/// Shared with [`os_unix_wasi::IOVEC_SCRATCH_OFFSET`].
const IOVEC_SCRATCH: i32 = 0x200;

#[link(wasm_import_module = "__speet_host_mem")]
extern "C" {
    fn store_i32(offset: i32, value: i32);
    fn load_i32(offset: i32) -> i32;
}

#[link(wasm_import_module = "wasi_snapshot_preview1")]
extern "C" {
    fn fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32;
    fn fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32;
    fn fd_close(fd: i32) -> i32;
    fn proc_exit(code: i32) -> !;
}

#[inline]
fn wasi_result(ret: i32, success_offset: i32) -> i64 {
    if ret != 0 {
        return -(ret as i64);
    }
    // SAFETY: host-mem imports are provided by the embedder.
    unsafe { load_i32(success_offset) as i64 }
}

#[inline]
fn marshal_iovec(a1: i64, a2: i64) {
    // SAFETY: host-mem imports are provided by the embedder before any handler runs.
    unsafe {
        store_i32(IOVEC_SCRATCH, a1 as i32);
        store_i32(IOVEC_SCRATCH + 4, a2 as i32);
    }
}

/// Unix `read(2)` → `wasi_snapshot_preview1::fd_read`.
#[no_mangle]
pub extern "C" fn handler_read(a0: i64, a1: i64, a2: i64) -> i64 {
    marshal_iovec(a1, a2);
    let ret = unsafe {
        fd_read(
            a0 as i32,
            IOVEC_SCRATCH,
            1,
            IOVEC_SCRATCH + 8,
        )
    };
    wasi_result(ret, IOVEC_SCRATCH + 8)
}

/// Unix `write(2)` → `wasi_snapshot_preview1::fd_write`.
#[no_mangle]
pub extern "C" fn handler_write(a0: i64, a1: i64, a2: i64) -> i64 {
    marshal_iovec(a1, a2);
    let ret = unsafe {
        fd_write(
            a0 as i32,
            IOVEC_SCRATCH,
            1,
            IOVEC_SCRATCH + 8,
        )
    };
    wasi_result(ret, IOVEC_SCRATCH + 8)
}

/// Unix `close(2)` → `wasi_snapshot_preview1::fd_close`.
#[no_mangle]
pub extern "C" fn handler_close(a0: i64) -> i64 {
    let ret = unsafe { fd_close(a0 as i32) };
    if ret != 0 {
        -(ret as i64)
    } else {
        0
    }
}

/// Unix `exit(2)` → `wasi_snapshot_preview1::proc_exit`.
#[no_mangle]
pub extern "C" fn handler_exit(a0: i64) -> ! {
    unsafe { proc_exit(a0 as i32) }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
