//! Generate C link-shim sources from a host import manifest.

use speet_host_api::{FuncImport, ImportManifest};

/// Emit C source defining `env__*` stubs and a `main` that bootstraps memory
/// and calls `__guest_entry`. Compile with `-DSPEET_RT_NO_MAIN` alongside
/// [`RUNTIME_C`] when using the split runtime + shim layout.
pub fn generate_shim(manifest: &ImportManifest) -> String {
    let mut out = String::from(
        r#"#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define WASM_PAGE_SIZE 65536u
#ifndef SPEET_RT_INITIAL_PAGES
#define SPEET_RT_INITIAL_PAGES 256u
#endif

extern uint8_t *__wasm_mem;
extern uint32_t __wasm_mem_pages;
extern long __guest_entry(void);

"#,
    );

    for imp in &manifest.func_imports {
        emit_import_stub(&mut out, imp);
    }

    out.push_str(
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
"#,
    );
    out
}

fn emit_import_stub(out: &mut String, imp: &FuncImport) {
    let sym = speet_host_api::ImportManifest::external_symbol(imp);
    match (imp.module.as_str(), imp.name.as_str()) {
        ("env", "exit") => {
            out.push_str(&format!(
                "void {sym}(int code) {{ _exit(code); }}\n\n"
            ));
        }
        ("env", "write") => {
            out.push_str(&format!(
                "int {sym}(int fd, int ptr, int len) {{
    return (int)write(fd, __wasm_mem + (unsigned)ptr, (size_t)len);
}}\n\n"
            ));
        }
        _ => {
            out.push_str(&format!(
                "/* TODO: stub for {} */\nvoid {sym}(void) {{ abort(); }}\n\n",
                imp.name
            ));
        }
    }
}

/// Emit a TU that provides `__wasm_mem` / growth symbols without `main`.
pub fn generate_memory_tu() -> String {
    String::from(
        r#"#define SPEET_RT_NO_MAIN
#include <stdint.h>
#include <stdlib.h>
extern void abort(void);
extern void *realloc(void *, unsigned long);

#define WASM_PAGE_SIZE 65536u
uint8_t *__wasm_mem = 0;
uint32_t __wasm_mem_pages = 0;

uint32_t __wasm_memory_grow(uint32_t delta, uint8_t **mem, uint32_t *pages) {
    uint32_t old = *pages;
    uint64_t new_pages = (uint64_t)old + (uint64_t)delta;
    if (new_pages > 0xFFFFu) return (uint32_t)-1;
    uint64_t new_bytes = new_pages * (uint64_t)WASM_PAGE_SIZE;
    uint8_t *p = (uint8_t *)realloc(*mem, new_bytes);
    if (!p) return (uint32_t)-1;
    uint64_t old_bytes = (uint64_t)old * (uint64_t)WASM_PAGE_SIZE;
    for (uint64_t i = old_bytes; i < new_bytes; i++) p[i] = 0;
    *mem = p;
    *pages = (uint32_t)new_pages;
    return old;
}
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_defines_env_exit_and_write() {
        let src = generate_shim(&ImportManifest::native_syscall());
        assert!(src.contains("env__exit"));
        assert!(src.contains("env__write"));
        assert!(src.contains("_exit(code)"));
    }
}
