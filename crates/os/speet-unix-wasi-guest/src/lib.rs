//! Shared Unix → WASI preview1 handlers as a linkable WASM module.
//!
//! Scratch iovec marshalling uses [`__speet_host_mem`] imports; parent adapter
//! builds lower them to multi-memory ops before embed.
//!
//! OS-specific syscall-number dispatch lives in the arch callback (not here),
//! so one guest binary serves Linux and Darwin.

#![no_std]

use core::panic::PanicInfo;
use os_unix_emulation::{
    wasi_close, wasi_exit, wasi_read, wasi_write, WasiHostMemory, WasiPreview1,
};

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

struct SpeetHostMemory;

impl WasiHostMemory for SpeetHostMemory {
    fn store_i32(&mut self, offset: u32, value: i32) {
        // SAFETY: the embedding supplies this import before any handler runs.
        unsafe { store_i32(offset as i32, value) }
    }

    fn load_i32(&mut self, offset: u32) -> i32 {
        // SAFETY: the embedding supplies this import before any handler runs.
        unsafe { load_i32(offset as i32) }
    }
}

struct SpeetWasi;

impl WasiPreview1 for SpeetWasi {
    fn fd_write(&mut self, fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32 {
        // SAFETY: imported from the canonical WASI preview1 host.
        unsafe { fd_write(fd, iovs, iovs_len, nwritten) }
    }

    fn fd_read(&mut self, fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> i32 {
        // SAFETY: imported from the canonical WASI preview1 host.
        unsafe { fd_read(fd, iovs, iovs_len, nread) }
    }

    fn fd_close(&mut self, fd: i32) -> i32 {
        // SAFETY: imported from the canonical WASI preview1 host.
        unsafe { fd_close(fd) }
    }

    fn proc_exit(&mut self, code: i32) -> ! {
        // SAFETY: imported from the canonical WASI preview1 host and never returns.
        unsafe { proc_exit(code) }
    }
}

/// Unix `read(2)` → `wasi_snapshot_preview1::fd_read`.
#[no_mangle]
pub extern "C" fn handler_read(a0: i64, a1: i64, a2: i64) -> i64 {
    wasi_read(
        &mut SpeetWasi,
        &mut SpeetHostMemory,
        a0 as u64,
        a1 as u64,
        a2 as u64,
    )
}

/// Unix `write(2)` → `wasi_snapshot_preview1::fd_write`.
#[no_mangle]
pub extern "C" fn handler_write(a0: i64, a1: i64, a2: i64) -> i64 {
    wasi_write(
        &mut SpeetWasi,
        &mut SpeetHostMemory,
        a0 as u64,
        a1 as u64,
        a2 as u64,
    )
}

/// Unix `close(2)` → `wasi_snapshot_preview1::fd_close`.
#[no_mangle]
pub extern "C" fn handler_close(a0: i64) -> i64 {
    wasi_close(&mut SpeetWasi, a0 as u64)
}

/// Unix `exit(2)` → `wasi_snapshot_preview1::proc_exit`.
#[no_mangle]
pub extern "C" fn handler_exit(a0: i64) -> ! {
    wasi_exit(&mut SpeetWasi, a0 as u64)
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
