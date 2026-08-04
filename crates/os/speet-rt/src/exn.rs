//! Generate the C/asm translation unit implementing the **software EH
//! stack** wasm-blitz's NaiveAbi native backends (x86-64, AArch64, RISC-V 64)
//! use for cross-function WASM exception propagation.
//!
//! # Why a software stack instead of walking CTX
//!
//! `docs/exception-propagation-gap.md` (wasm-blitz) originally specified
//! walking the compile-time CTX frame chain into the caller's saved CTX on a
//! miss. That works for x86-64 (which has a CTX register to walk), but
//! AArch64's NaiveAbi `try_table` doesn't thread a caller-CTX link at all
//! today, and RISC-V's `if_stack` scan is compile-time-only (see
//! `docs/guides/arch-recompilers.md` in `speet`). A **global software EH
//! stack** — a small array of `{dispatch, saved_sp}` frames pushed on
//! `try_table` entry and popped on exit/dispatch — sidesteps all three
//! backends' differing frame layouts: `try_table`/`throw` codegen only ever
//! needs to call four leaf helpers (`__wasm_eh_push`, `__wasm_eh_pop`,
//! `__wasm_eh_take`, and the `__wasm_exn_propagate` trampoline below), never
//! reason about another function's stack shape.
//!
//! # Why `__wasm_exn_propagate` is hand-written asm, not plain C
//!
//! Every NaiveAbi backend's WASM operand stack pointer *is* the real
//! hardware stack pointer (`RSP`/`SP`/`sp`; see `docs/abi.md`). Guest code
//! reaches `__wasm_exn_propagate` via a bare **jump** (not a `call`) so that
//! it never pushes a stray return address onto that shared stack — but that
//! also means this function is entered without the ABI's usual "callee sees
//! `sp` 16-byte-misaligned by exactly one pushed word" invariant, and on
//! AArch64 in particular, an ordinary compiler-generated prologue (e.g. a
//! `stp` pre-index against an unaligned `sp`) can raise an SP-alignment
//! fault before a single line of our C ever runs. The trampoline below is
//! therefore raw assembly with **no compiler-generated prologue**: it
//! aligns `sp`/`rsp` itself, then makes a normal, ABI-conforming call into
//! `__wasm_eh_take` (plain, portable C) to do the actual bookkeeping.
//!
//! `SYM(name)` resolves `name` to the exact assembler-visible symbol clang
//! will look for when linking against a same-named plain C function —
//! Mach-O prepends a leading underscore to every C symbol (`foo` -> `_foo`),
//! ELF does not. This has to be done by hand only inside these raw `asm`
//! blocks: everywhere else in this crate (`__wasm_eh_push`, `__wasm_eh_pop`,
//! `__wasm_eh_take`, `__wasm_unhandled_exception`) is plain C, so clang's own
//! name mangling applies automatically and already matches what
//! `speet-recompile`'s `drive.rs` mangles undefined `External` relocations
//! to on Mach-O.

/// Emit the C translation unit defining the software EH stack and
/// `__wasm_exn_propagate`. Compiled and linked alongside the guest object,
/// same as [`crate::generate_memory_tu`] / [`crate::generate_data_segments_c`].
pub fn generate_exn_tu() -> String {
    String::from(
        r#"#include <stdint.h>

#define WASM_EH_MAX_DEPTH 64

typedef struct {
    uint64_t dispatch;
    uint64_t saved_sp;
} __wasm_eh_frame_t;

static __wasm_eh_frame_t __wasm_eh_frames[WASM_EH_MAX_DEPTH];
static uint32_t __wasm_eh_sp = 0;

/* Called by generated try_table/throw codegen (see wasm-blitz's naive.rs
 * backends) on try_table entry. Silently drops the frame past
 * WASM_EH_MAX_DEPTH nesting — deeply nested try_table beyond that depth
 * degrades to "no local handler found" (propagate walks past it), rather
 * than corrupting memory. */
void __wasm_eh_push(uint64_t dispatch, uint64_t saved_sp) {
    if (__wasm_eh_sp < WASM_EH_MAX_DEPTH) {
        __wasm_eh_frames[__wasm_eh_sp].dispatch = dispatch;
        __wasm_eh_frames[__wasm_eh_sp].saved_sp = saved_sp;
        __wasm_eh_sp++;
    }
}

/* Called on try_table's normal (non-throwing) exit, and at a local throw's
 * jump site right before it jumps directly to its own dispatch stub —
 * either way the frame is being consumed exactly once. */
void __wasm_eh_pop(void) {
    if (__wasm_eh_sp > 0) {
        __wasm_eh_sp--;
    }
}

/* Called only by __wasm_exn_propagate below. Pops and returns the innermost
 * still-open try_table frame from *any* function in the current call chain
 * (this is what makes propagation "cross-function" without walking any
 * per-ISA frame chain): 0 if none remains (unhandled), 1 with *dispatch_out/
 * *sp_out populated otherwise. */
int __wasm_eh_take(uint64_t *dispatch_out, uint64_t *sp_out) {
    if (__wasm_eh_sp == 0) {
        return 0;
    }
    __wasm_eh_sp--;
    *dispatch_out = __wasm_eh_frames[__wasm_eh_sp].dispatch;
    *sp_out = __wasm_eh_frames[__wasm_eh_sp].saved_sp;
    return 1;
}

/* Final "no handler anywhere" trap (docs/abi.md step 6). */
__attribute__((noreturn))
void __wasm_unhandled_exception(void) {
    __builtin_trap();
}

#if defined(__APPLE__)
#define SYM(x) "_" x
#else
#define SYM(x) x
#endif

/*
 * __wasm_exn_propagate — entered via a bare jump (never `call`) from
 * generated guest code whenever a `throw` (or an unmatched catch) finds no
 * try_table handler in the current function. No compiler-generated
 * prologue runs here (see module doc): align the incoming stack pointer by
 * hand, reserve scratch space for __wasm_eh_take's two out-params, and only
 * then make a normal, ABI-conforming call. On a hit, fully overwrite the
 * stack pointer with the handler's saved value and jump to its dispatch
 * stub; on a miss, fall through to __wasm_unhandled_exception.
 */
#if defined(__x86_64__)
__asm__(
    ".text\n"
    ".global " SYM("__wasm_exn_propagate") "\n"
    SYM("__wasm_exn_propagate") ":\n"
    "    and $-16, %rsp\n"
    "    sub $32, %rsp\n"
    "    lea 16(%rsp), %rdi\n"
    "    lea 24(%rsp), %rsi\n"
    "    call " SYM("__wasm_eh_take") "\n"
    "    test %eax, %eax\n"
    "    jz 1f\n"
    "    mov 16(%rsp), %rax\n"
    "    mov 24(%rsp), %rdi\n"
    "    mov %rdi, %rsp\n"
    "    jmp *%rax\n"
    "1:\n"
    "    call " SYM("__wasm_unhandled_exception") "\n"
    "    ud2\n"
);
#elif defined(__aarch64__)
__asm__(
    ".text\n"
    ".global " SYM("__wasm_exn_propagate") "\n"
    SYM("__wasm_exn_propagate") ":\n"
    "    mov x9, sp\n"
    "    and x9, x9, #0xfffffffffffffff0\n"
    "    sub x9, x9, #32\n"
    "    mov sp, x9\n"
    "    add x0, sp, #16\n"
    "    add x1, sp, #24\n"
    "    bl " SYM("__wasm_eh_take") "\n"
    "    cbz w0, 1f\n"
    "    ldr x9, [sp, #16]\n"
    "    ldr x10, [sp, #24]\n"
    "    mov sp, x10\n"
    "    br x9\n"
    "1:\n"
    "    bl " SYM("__wasm_unhandled_exception") "\n"
    "    brk #1\n"
);
#elif defined(__riscv)
__asm__(
    ".text\n"
    ".global __wasm_exn_propagate\n"
    "__wasm_exn_propagate:\n"
    "    andi sp, sp, -16\n"
    "    addi sp, sp, -32\n"
    "    addi a0, sp, 16\n"
    "    addi a1, sp, 24\n"
    "    call __wasm_eh_take\n"
    "    beqz a0, 1f\n"
    "    ld t0, 16(sp)\n"
    "    ld t1, 24(sp)\n"
    "    mv sp, t1\n"
    "    jr t0\n"
    "1:\n"
    "    call __wasm_unhandled_exception\n"
    "    .word 0\n"
);
#endif
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defines_the_software_eh_stack_helpers() {
        let src = generate_exn_tu();
        assert!(src.contains("void __wasm_eh_push(uint64_t dispatch, uint64_t saved_sp)"));
        assert!(src.contains("void __wasm_eh_pop(void)"));
        assert!(src.contains("int __wasm_eh_take(uint64_t *dispatch_out, uint64_t *sp_out)"));
        assert!(src.contains("void __wasm_unhandled_exception(void)"));
    }

    #[test]
    fn defines_propagate_trampoline_for_every_naive_abi_target() {
        let src = generate_exn_tu();
        assert!(src.contains("#if defined(__x86_64__)"));
        assert!(src.contains("#elif defined(__aarch64__)"));
        assert!(src.contains("#elif defined(__riscv)"));
        // Each arch's block must both define the entry label and invoke the
        // two C helpers, so codegen's `External` references resolve.
        for needle in ["__wasm_exn_propagate", "__wasm_eh_take", "__wasm_unhandled_exception"] {
            assert!(
                src.matches(needle).count() >= 4,
                "expected {needle} referenced in every one of the 3 arch blocks plus its own definition"
            );
        }
    }
}
