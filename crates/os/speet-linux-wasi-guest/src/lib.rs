//! RV64 Linux → WASI preview1 handlers as a linkable WASM module.
//!
//! Scratch iovec marshalling uses [`__speet_host_mem`] imports; the parent
//! `speet-linux-wasi` build lowers them to multi-memory ops before embed.

#![no_std]

use core::panic::PanicInfo;

/// Shared with [`os_linux_wasi::IOVEC_SCRATCH_OFFSET`].
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

/// Linux `read(2)` → `wasi_snapshot_preview1::fd_read`.
#[no_mangle]
pub extern "C" fn handler_read(a0: i64, a1: i64, a2: i64) -> i64 {
    marshal_iovec(a1, a2);
    // SAFETY: WASI imports are wired by the megabinary linker.
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

/// Linux `write(2)` → `wasi_snapshot_preview1::fd_write`.
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

/// Linux `close(2)` → `wasi_snapshot_preview1::fd_close`.
#[no_mangle]
pub extern "C" fn handler_close(a0: i64) -> i64 {
    let ret = unsafe { fd_close(a0 as i32) };
    if ret != 0 {
        -(ret as i64)
    } else {
        0
    }
}

/// Linux `exit(2)` / `exit_group(2)` → `wasi_snapshot_preview1::proc_exit`.
#[no_mangle]
pub extern "C" fn handler_exit(a0: i64) -> ! {
    unsafe { proc_exit(a0 as i32) }
}

/// Syscall-number dispatch (replaces inline guest-site `br_table`).
#[no_mangle]
pub extern "C" fn syscall_dispatch(num: i64, a0: i64, a1: i64, a2: i64) -> i64 {
    match num as u64 {
        63 => handler_read(a0, a1, a2),
        64 => handler_write(a0, a1, a2),
        57 => handler_close(a0),
        93 | 94 => handler_exit(a0),
        _ => -(38i64), // ENOSYS
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
