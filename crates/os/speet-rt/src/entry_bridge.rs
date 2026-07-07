//! Entry bridge: preserve `argc`/`argv` for recompiled guest startup, and
//! call `__guest_entry` with a real, C-ABI-correct argument list.
//!
//! Guest registers ARE and always shall be WASM params (see
//! `docs/guides/thin-runtime-genericity.md`) — every translated function,
//! `__guest_entry` included, shares one big parameter list representing the
//! full guest register file, threaded through every `call`/`return_call` in
//! the translated graph so control can land on any instruction slot with the
//! right register state already in place. That means a C caller invoking
//! `__guest_entry` for the very first time must itself supply real initial
//! values for *every one* of those params — a bare `extern long
//! __guest_entry(void)` prototype leaves them all as whatever garbage was in
//! the incoming registers/stack at the call site. SP is the sharpest edge:
//! the guest's very first instruction is commonly a `stp`/`push`-style
//! prologue that writes relative to it, so a garbage SP corrupts memory (or
//! segfaults) before any real guest logic runs.
//!
//! [`entry_bridge_c`] generates a prototype with the entry's *actual* param
//! count (from `speet_recompile::drive::entry_param_count`) and passes a
//! freshly allocated, 16-byte-aligned native buffer as the SP argument (at
//! the arch's `SP_PARAM_INDEX` — see `speet_aarch64`/`speet_x86_64`), zero
//! for everything else. All params are declared/passed as `uint64_t`
//! uniformly, even the ones that are logically `f64` (guest FP registers):
//! the recompiler's SysV/AAPCS64 entry marshals every argument through a
//! flat "first N in integer registers, then a flat outgoing stack" scheme
//! keyed by *position*, not by C type (see `crates/blitz-aarch64/src/sysv.rs`
//! in `wasm-blitz`) — declaring an FP-register param as `double` would make
//! the C compiler pass it through a SIMD register instead, which is not
//! where the callee's prologue reads it from.
//!
//! # The halt sentinel (`registers -> registers`, principle 4)
//!
//! Every speet-emitted function has type `(registers) -> (registers)` (see
//! `docs/guides/thin-runtime-genericity.md` principle 4), and every guest
//! translation reserves one **halt stub** table slot, one guest address
//! past the last translated instruction (see
//! `speet_recompile::frontend::halt_addr`), whose body surfaces the live
//! register file as `__guest_entry`'s real C return value. This is a
//! *normal* feature, not a test-only convenience: real guest programs may
//! return out of `main` (most commonly when there's no translated crt0
//! chain to call `exit`), and the halt stub is where that return has to
//! land instead of on whatever garbage the return-address register/slot
//! held at entry. [`entry_bridge_c`] therefore always seeds the guest's
//! initial return address to the halt sentinel before the first call:
//! - **Register-based archs** (AArch64 LR, and any future arch with a
//!   dedicated link register): pass `halt_addr` directly as the
//!   `lr_param_index` argument.
//! - **Stack-based archs** (x86-64, where `ret` reads the return address
//!   from guest memory at `[RSP]`): write `halt_addr` 8 bytes below the
//!   16-aligned stack top and start SP there — exactly mimicking what a
//!   real `call` instruction would have pushed, so `main`'s `ret` reads it
//!   back the normal way.
//!
//! `__speet_start` then forwards `__guest_entry`'s return value to the
//! process exit code via `_exit`, mirroring a real `crt0`'s
//! `exit(main())` — the same semantics whether the guest actually calls
//! `exit()` (which never returns to `__guest_entry` at all) or legitimately
//! falls off the end of `main` onto the halt stub.

/// Generate the entry-bridge C source for a `__guest_entry` whose function
/// type has `entry_param_count` params, with the guest SP register at
/// `sp_param_index` (see `speet_aarch64::AArch64Recompiler::SP_PARAM_INDEX` /
/// `speet_x86_64::X86Recompiler::SP_PARAM_INDEX`), seeding the initial
/// return address to `halt_addr` (see `speet_recompile::frontend::halt_addr`)
/// — see this module's doc for the halt-sentinel contract this implements.
///
/// `lr_param_index` selects how the sentinel is seeded: `Some(idx)` for a
/// register-based arch (AArch64's `LR_PARAM_INDEX`), `None` for a
/// stack-based arch (x86-64, whose `ret` reads the return address from
/// guest memory, not a register).
pub fn entry_bridge_c(
    entry_param_count: u32,
    sp_param_index: u32,
    halt_addr: u64,
    lr_param_index: Option<u32>,
) -> String {
    let n = entry_param_count as usize;
    let sp_idx = sp_param_index as usize;

    let params_decl = if n == 0 {
        "void".to_string()
    } else {
        (0..n).map(|i| format!("uint64_t p{i}")).collect::<Vec<_>>().join(", ")
    };
    let args = (0..n)
        .map(|i| {
            if i == sp_idx {
                "__speet_guest_sp".to_string()
            } else if lr_param_index == Some(i as u32) {
                "SPEET_HALT_ADDR".to_string()
            } else {
                "0".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    // Register-based archs (`lr_param_index = Some(_)`): the fresh stack's
    // 16-aligned top is the initial SP as-is; the sentinel goes directly
    // into the LR param above, no stack write needed. Stack-based archs
    // (`None`, x86-64): reserve 8 bytes below the aligned top for the
    // sentinel "return address" and start SP there, then write it —
    // matching a real `call`'s effect on the stack.
    let (sp_init, seed_stack_ra) = match lr_param_index {
        Some(_) => (
            "((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF"
                .to_string(),
            String::new(),
        ),
        None => (
            "(((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF) - 8"
                .to_string(),
            "    *(uint64_t *)(uintptr_t)__speet_guest_sp = SPEET_HALT_ADDR;\n".to_string(),
        ),
    };

    format!(
        r#"#include <stdint.h>
#include <unistd.h>

#define SPEET_HALT_ADDR ((uint64_t){halt_addr}ULL)

int __speet_argc = 0;
char **__speet_argv = 0;

void __speet_set_argv(int argc, char **argv) {{
    __speet_argc = argc;
    __speet_argv = argv;
}}

/* Freshly allocated native stack for the guest's initial SP -- see this
 * file's module doc. 8 MiB matches common default thread-stack sizes. */
#define SPEET_GUEST_STACK_SIZE (8u * 1024u * 1024u)
static uint8_t __speet_guest_stack[SPEET_GUEST_STACK_SIZE] __attribute__((aligned(16)));

extern long __guest_entry({params_decl});

void __speet_start(int argc, char **argv) {{
    __speet_set_argv(argc, argv);
    uint64_t __speet_guest_sp = {sp_init};
{seed_stack_ra}    (void)__speet_guest_sp;
    /* Mirrors a real crt0's `exit(main())`: whether the guest calls
     * `exit()` directly (never returning here) or legitimately falls off
     * the end of `main` onto the halt stub (see this file's module doc),
     * the process exit code ends up as the guest's return value either
     * way. */
    long ret = __guest_entry({args});
    _exit((int)ret);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_args(src: &str) -> Vec<String> {
        let call_line = src
            .lines()
            .find(|l| l.contains("__guest_entry(") && !l.contains("extern"))
            .unwrap();
        let inner = call_line
            .trim()
            .trim_end_matches(';')
            .split("__guest_entry(")
            .nth(1)
            .unwrap()
            .trim_end_matches(')');
        if inner.is_empty() {
            Vec::new()
        } else {
            inner.split(", ").map(str::to_string).collect()
        }
    }

    #[test]
    fn declares_exact_param_count_and_seeds_sp_register_lr() {
        // AArch64-style: LR is a param (register-based seeding).
        let src = entry_bridge_c(72, 71, 0x1000, Some(30));
        assert!(src.contains("uint64_t p0, "));
        assert!(src.contains("uint64_t p71"));
        let args = call_args(&src);
        assert_eq!(args.len(), 72);
        assert_eq!(args[71], "__speet_guest_sp");
        assert_eq!(args[30], "SPEET_HALT_ADDR");
        assert_eq!(args[0], "0");
        assert!(src.contains("#define SPEET_HALT_ADDR ((uint64_t)4096ULL)"));
        // Register-based: no stack write, SP is the aligned top as-is.
        assert!(!src.contains("*(uint64_t *)(uintptr_t)__speet_guest_sp = SPEET_HALT_ADDR;"));
    }

    #[test]
    fn seeds_stack_return_address_when_no_lr_param() {
        // x86-64-style: no LR param; the sentinel is written to guest memory
        // at the initial (reserved) stack top, mimicking a real `call`.
        let src = entry_bridge_c(5, 4, 0x2000, None);
        let args = call_args(&src);
        assert_eq!(args.len(), 5);
        assert_eq!(args[4], "__speet_guest_sp");
        assert!(!args.contains(&"SPEET_HALT_ADDR".to_string()));
        assert!(src.contains("*(uint64_t *)(uintptr_t)__speet_guest_sp = SPEET_HALT_ADDR;"));
        assert!(src.contains(") - 8"));
    }

    #[test]
    fn forwards_guest_entry_return_value_to_process_exit() {
        let src = entry_bridge_c(1, 0, 0x1000, Some(0));
        assert!(src.contains("long ret = __guest_entry("));
        assert!(src.contains("_exit((int)ret);"));
    }

    #[test]
    fn zero_params_uses_void_prototype() {
        let src = entry_bridge_c(0, 0, 0x1000, None);
        assert!(src.contains("__guest_entry(void)"));
        assert!(src.contains("__guest_entry();"));
    }
}
