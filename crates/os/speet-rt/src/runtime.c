/*
 * speet-rt — runtime shim for recompiled host binaries.
 *
 * The full-binary recompiler lowers a guest binary to WASM (speet) and then to
 * native machine code (wasm-blitz). The generated `.text` references a handful
 * of runtime symbols that this shim must define, and needs a process entry that
 * bootstraps linear memory and transfers control to the recompiled guest entry.
 *
 * Symbols the wasm-blitz native backend emits as `External` and that we provide:
 *   - __wasm_mem        : base pointer of WASM linear memory (byte buffer).
 *   - __wasm_mem_pages  : current linear-memory size in 64 KiB pages.
 *   - __wasm_memory_grow: grow linear memory (delta pages), returns old pages or -1.
 *
 * The recompiled guest entry is exported by the backend as `__guest_entry`
 * (an export dispatcher stub that tail-jumps to the internal entry function).
 *
 * MEMORY MODEL NOTE (gating issue for memory-touching programs):
 *   The wasm-blitz *C* backend addresses memory as `__wasm_mem + (uint32_t)addr`
 *   (base + 32-bit-wrapped offset), but the x86-64/aarch64 *naive* native
 *   backends currently dereference the WASM address as a raw host pointer with
 *   no base added. Until the native backend adds a base register (or is run with
 *   memory mapped such that base == 0), only programs that never touch linear
 *   memory (e.g. a syscall-only `exit`) work end-to-end. M1 deliberately uses
 *   such a program. See STATUS.md.
 */

#include <stdint.h>

/* 64 KiB WASM page. */
#define WASM_PAGE_SIZE 65536u

/* Initial linear-memory reservation, in pages (16 MiB). Tunable. */
#ifndef SPEET_RT_INITIAL_PAGES
#define SPEET_RT_INITIAL_PAGES 256u
#endif

/* Defined here, referenced by generated code. */
uint8_t  *__wasm_mem = 0;
uint32_t  __wasm_mem_pages = 0;

/* libc, tunneled from the host. */
extern void  *malloc(unsigned long);
extern void  *realloc(void *, unsigned long);
extern void   abort(void);

/*
 * Grow linear memory by `delta` pages. Signature matches the wasm-blitz
 * contract: uint32_t __wasm_memory_grow(uint32_t delta, uint8_t** mem, uint32_t* pages).
 * Returns the previous page count, or (uint32_t)-1 on failure.
 */
uint32_t __wasm_memory_grow(uint32_t delta, uint8_t **mem, uint32_t *pages) {
    uint32_t old = *pages;
    uint64_t new_pages = (uint64_t)old + (uint64_t)delta;
    if (new_pages > 0xFFFFu) {
        return (uint32_t)-1;
    }
    uint64_t new_bytes = new_pages * (uint64_t)WASM_PAGE_SIZE;
    uint8_t *p = (uint8_t *)realloc(*mem, new_bytes);
    if (!p) {
        return (uint32_t)-1;
    }
    /* Zero the freshly added pages, per WASM semantics. */
    uint64_t old_bytes = (uint64_t)old * (uint64_t)WASM_PAGE_SIZE;
    for (uint64_t i = old_bytes; i < new_bytes; i++) {
        p[i] = 0;
    }
    *mem = p;
    *pages = (uint32_t)new_pages;
    return old;
}

/*
 * Passive-data initializer: the backend's compiled guest object exports a
 * real one when the guest has data segments, otherwise whoever links this
 * TU must supply a no-op definition (see `speet_rt::generate_data_segments_c`,
 * which does exactly that for the integrated shim path). A weak/`if`-guarded
 * declaration was tried first but doesn't work here: Apple's `ld64` only
 * treats `weak_import` as optional for symbols coming from a *dylib* at load
 * time, and still hard-errors "symbol not found" for an unresolved weak
 * reference against a plain relocatable object in the same static link.
 */
extern void __speet_data_init(void);

/*
 * Recompiled guest entry. The backend emits an export dispatcher named
 * `__guest_entry`. NOTE: the dispatcher uses the wasm-blitz NaiveAbi, not the C
 * ABI; calling it directly from C only works once the driver emits a SysV/AAPCS64
 * entry trampoline (see STATUS.md "ABI bridge"). For the M1 syscall-only program
 * the entry never returns (it exits via syscall), so the mismatch is benign.
 */
extern void __guest_entry(void);

#ifndef SPEET_RT_NO_MAIN
/* Provided by libc crt; we hook in via the standard C `main`. */
int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    /* Bootstrap linear memory. */
    __wasm_mem_pages = SPEET_RT_INITIAL_PAGES;
    uint64_t bytes = (uint64_t)SPEET_RT_INITIAL_PAGES * (uint64_t)WASM_PAGE_SIZE;
    __wasm_mem = (uint8_t *)malloc(bytes);
    if (!__wasm_mem) {
        abort();
    }
    for (uint64_t i = 0; i < bytes; i++) {
        __wasm_mem[i] = 0;
    }

    __speet_data_init();

    __guest_entry();
    return 0;
}
#endif /* SPEET_RT_NO_MAIN */
