//! Emit C TU that provides `__wasm_mem` / growth symbols without `main`.

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
