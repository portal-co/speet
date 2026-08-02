//! Registry-driven speet WASM import bridge generation.

use speet_host_api::{FuncImport, ImportManifest};

mod defaults;
mod internal;

use speet_host_api::{FuncImport, ImportManifest};

pub use defaults::{BridgeHandler, SpeetBridgeRegistry};

/// Emit C source defining `env__*` stubs, shared helpers, and bootstrap `main`.
pub fn generate_shim(manifest: &ImportManifest) -> String {
    generate_shim_with_registry(manifest, &SpeetBridgeRegistry::integrated())
}

/// Emit with an explicit bridge handler registry (for custom hosts).
pub fn generate_shim_with_registry(manifest: &ImportManifest, registry: &SpeetBridgeRegistry) -> String {
    let mut out = shim_preamble();
    for imp in &manifest.func_imports {
        registry.emit_import_stub(&mut out, imp);
    }
    out.push_str(shim_main());
    out
}

fn shim_preamble() -> String {
    format!(
        r#"#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "os_shim.h"

#define WASM_PAGE_SIZE 65536u
#ifndef SPEET_RT_INITIAL_PAGES
#define SPEET_RT_INITIAL_PAGES 256u
#endif

extern uint8_t *__wasm_mem;
extern uint32_t __wasm_mem_pages;
extern long __guest_entry(void);
extern void __speet_start(int argc, char **argv);

#define SPEET_HOST_STR_SCRATCH_BYTES {HOST_STR_SCRATCH_BYTES}u
static uint32_t speet_copy_host_str_to_wasm(const char *s) {{
    if (!s) return 0;
    uint64_t scratch_off = (uint64_t)__wasm_mem_pages * (uint64_t)WASM_PAGE_SIZE
        - SPEET_HOST_STR_SCRATCH_BYTES;
    size_t cap = SPEET_HOST_STR_SCRATCH_BYTES - 1;
    size_t len = strlen(s);
    if (len > cap) len = cap;
    memcpy(__wasm_mem + scratch_off, s, len);
    __wasm_mem[scratch_off + len] = 0;
    return (uint32_t)scratch_off;
}}

"#,
        HOST_STR_SCRATCH_BYTES = crate::HOST_STR_SCRATCH_BYTES,
    )
}

fn shim_main() -> &'static str {
    r#"
static void speet_rt_bootstrap(void) {
    extern uint8_t *__wasm_mem;
    extern uint32_t __wasm_mem_pages;
    __wasm_mem_pages = SPEET_RT_INITIAL_PAGES;
    uint64_t bytes = (uint64_t)SPEET_RT_INITIAL_PAGES * (uint64_t)WASM_PAGE_SIZE;
    __wasm_mem = (uint8_t *)malloc(bytes);
    if (!__wasm_mem) abort();
    for (uint64_t i = 0; i < bytes; i++) __wasm_mem[i] = 0;
}

int main(void) {
    speet_rt_bootstrap();
    __guest_entry();
    return 0;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use speet_host_api::ImportManifest;

    #[test]
    fn shim_defines_env_exit_and_write_via_os_shim() {
        let src = generate_shim(&ImportManifest::native_syscall());
        assert!(src.contains("env__exit"));
        assert!(src.contains("env__write"));
        assert!(src.contains("os_shim_exit"));
        assert!(src.contains("os_shim_write"));
    }

    #[test]
    fn integrated_manifest_uses_os_shim_execve() {
        let src = generate_shim(&ImportManifest::integrated_native());
        assert!(src.contains("os_shim_execve"));
        assert!(!src.contains("__speet_execve_hook"));
    }
}
