//! Emit C TU that provides `__wasm_mem` / growth symbols without `main`.

pub fn generate_memory_tu() -> String {
    String::from(
        r#"#define SPEET_RT_NO_MAIN
#include <stdint.h>
#include <stdlib.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
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

/*
 * ZeroOffset: mark [text_base, text_base+len) inside the host mirror as
 * inaccessible so unrecompiled machine code is never readable via
 * `__wasm_mem + va`, and host BridgeSupport cannot touch it. If the hole
 * lies entirely outside the committed buffer, this is a no-op (already
 * unmapped from the guest's perspective).
 */
void speet_protect_text_hole(uint64_t text_base, uint64_t text_len) {
#if defined(__APPLE__) || defined(__linux__)
    if (!__wasm_mem || text_len == 0) return;
    uint64_t mem_bytes = (uint64_t)__wasm_mem_pages * (uint64_t)WASM_PAGE_SIZE;
    if (text_base >= mem_bytes) return;
    uint64_t end = text_base + text_len;
    if (end > mem_bytes) end = mem_bytes;
    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0) psz = 4096;
    uintptr_t abs = (uintptr_t)(__wasm_mem + text_base);
    uintptr_t page = abs & ~(uintptr_t)(psz - 1);
    size_t span = (size_t)(end - text_base) + (size_t)(abs - page);
    span = (span + (size_t)psz - 1) & ~(size_t)(psz - 1);
    (void)mprotect((void *)page, span, PROT_NONE);
#else
    (void)text_base;
    (void)text_len;
#endif
}

/* Bulk-memory helpers for wasm-blitz MemoryCopy / MemoryFill lowering.
 * Overlap-safe copy uses memmove (WASM memory.copy semantics). */
#include <string.h>
void __wasm_memory_copy(uint32_t dest_off, uint32_t src_off, uint32_t len) {
    memmove(__wasm_mem + dest_off, __wasm_mem + src_off, len);
}
void __wasm_memory_fill(uint32_t dest_off, uint32_t val, uint32_t len) {
    memset(__wasm_mem + dest_off, (int)(uint8_t)val, len);
}
"#,
    )
}
