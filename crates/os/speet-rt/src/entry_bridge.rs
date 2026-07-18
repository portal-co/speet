//! Entry bridge: preserve `argc`/`argv` for recompiled guest startup, and
//! invoke the entry catalog stub with a seeded virtual register file.
//!
//! Guest registers ARE and always shall be WASM params (see
//! `docs/guides/thin-runtime-genericity.md`) — every translated function,
//! `__guest_entry` included, shares one big parameter list representing the
//! full guest register file. [`entry_bridge_c`] seeds SP and the halt
//! sentinel return address, then calls the per-function host stub for the
//! entry slot (same path as fn-ptr callbacks).

/// Generate the entry-bridge C source for a guest whose register file has
/// `entry_param_count` WASM params, with SP at `sp_param_index`.
///
/// `entry_stub_sym` is the link-time host symbol for the guest-function
/// stub catalog's entry slot (`__speet_guest_fn_N`), called directly.
/// `halt_guest_pc` is the catalog's halt-sentinel *guest* address
/// (`GuestFuncCatalog::halt_entry().guest_pc`, i.e. one past the end of the
/// guest `.text`) — seeded as the initial return address so the guest's own
/// `ret` handling (which treats a "return address" as a guest PC and
/// dispatches through `__wasm_table`, since `ret` is architecturally `br
/// LR`) lands on the catalog's dedicated halt entry.
pub fn entry_bridge_c(
    entry_param_count: u32,
    sp_param_index: u32,
    lr_param_index: Option<u32>,
    entry_stub_sym: &str,
    halt_guest_pc: u64,
) -> String {
    entry_bridge_with_catalog(
        entry_param_count,
        sp_param_index,
        lr_param_index,
        entry_stub_sym,
        halt_guest_pc,
    )
}

/// Direct `__guest_entry` bridge for callers without a guest-function catalog.
///
/// `abi_pad_args` is the number of unused leading `uint64_t` dummy params to
/// prepend to the call, needed only on backends whose `CallAbi::AllStack`
/// truly marshals *every* param through the stack (ignoring the target's
/// normal C argument registers) — currently x86-64's wasm-blitz backend
/// (pass 6, its SysV integer-register count). AArch64's backend implements
/// "AllStack" as ordinary AAPCS64 marshalling (X0-X7 then stack), which
/// already matches a plain C call byte-for-byte, so it must be called with
/// `abi_pad_args = 0` — padding it would misalign every real argument by
/// `abi_pad_args` register slots instead of fixing anything. See
/// `speet_rt::guest_stubs::generate_guest_stubs_c`'s `__speet_invoke` for the
/// identical, x86-64-only issue on the catalog-based entry path.
pub fn entry_bridge_direct_c(
    entry_param_count: u32,
    sp_param_index: u32,
    halt_addr: u64,
    lr_param_index: Option<u32>,
    abi_pad_args: u32,
) -> String {
    let n = entry_param_count as usize;
    let sp_idx = sp_param_index as usize;

    let params_decl = (0..n)
        .map(|i| format!("uint64_t p{i}"))
        .collect::<Vec<_>>()
        .join(", ");
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

    let dummy_decl = (0..abi_pad_args)
        .map(|i| format!("uint64_t __speet_d{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let dummy_args = (0..abi_pad_args).map(|_| "0").collect::<Vec<_>>().join(", ");
    let full_params_decl = match (dummy_decl.is_empty(), params_decl.is_empty()) {
        (true, true) => "void".to_string(),
        (true, false) => params_decl,
        (false, true) => dummy_decl,
        (false, false) => format!("{dummy_decl}, {params_decl}"),
    };
    let full_args = match (dummy_args.is_empty(), args.is_empty()) {
        (true, _) => args,
        (false, true) => dummy_args,
        (false, false) => format!("{dummy_args}, {args}"),
    };

    // Reserve `HOST_STR_SCRATCH_BYTES` above the stack for host-import stubs
    // that need to copy host-owned data (e.g. `getenv`'s result) into a
    // guest-visible location — see `crate::HOST_STR_SCRATCH_BYTES`'s doc
    // comment. The stack only ever grows *down* from here, so it can never
    // collide with that region.
    let mem_top = format!(
        "((uint64_t)__wasm_mem_pages * 65536u - {}u)",
        crate::HOST_STR_SCRATCH_BYTES
    );
    let (sp_init, seed_stack_ra) = match lr_param_index {
        Some(_) => (
            format!("({mem_top}) & ~(uint64_t)0xF"),
            String::new(),
        ),
        None => (
            format!("(({mem_top}) & ~(uint64_t)0xF) - 8"),
            format!(
                "    *(uint64_t *)(uintptr_t)(__wasm_mem + __speet_guest_sp) = SPEET_HALT_ADDR;\n",
            ),
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

// The guest's stack lives *inside* `__wasm_mem`, at its high end, not in a
// separate host buffer. Guest code translated from SP-relative loads/stores
// (`stp`/`ldr [sp, ...]`) compiles down to ordinary WASM memory ops, which
// address relative to `__wasm_mem`'s base with a 32-bit-wrapped offset (see
// `docs/naive-abi-deprecation.md`/WASM's linear-memory model). Seeding SP
// with a *raw host pointer* into some other buffer (the previous scheme, a
// separate `__speet_guest_stack` static array) silently truncated the
// pointer's upper 32 bits on every such access — the truncated address only
// happened to land in mapped memory some of the time, producing an
// ASLR-dependent SIGBUS/SIGSEGV. Seeding SP as a small offset within
// `__wasm_mem` (well under 4GB) keeps that wrap lossless.
extern uint8_t *__wasm_mem;
extern uint32_t __wasm_mem_pages;

extern long __guest_entry({full_params_decl});

void __speet_start(int argc, char **argv) {{
    __speet_set_argv(argc, argv);
    uint64_t __speet_guest_sp = {sp_init};
{seed_stack_ra}    long ret = __guest_entry({full_args});
    _exit((int)ret);
}}
"#
    )
}

fn entry_bridge_with_catalog(
    entry_param_count: u32,
    sp_param_index: u32,
    lr_param_index: Option<u32>,
    entry_stub_sym: &str,
    halt_guest_pc: u64,
) -> String {
    let n = entry_param_count as usize;
    let sp_idx = sp_param_index as usize;

    // The seeded "return address" is read back by the guest's own compiled
    // `ret` handling, which treats it as a GUEST address and converts it via
    // the same guest-PC -> `__wasm_table` index arithmetic used for any other
    // indirect branch (`ret` is architecturally just `br LR`) — see
    // `GuestFuncCatalog::halt_entry`'s `guest_pc`, a sentinel one past the end
    // of the guest `.text` that lands on the catalog's dedicated halt table
    // entry. It must be that numeric guest address, NOT a native function
    // pointer's own address (taking `&halt_stub_sym` here was the bug: it
    // fed a native address into guest-address arithmetic, producing a wild
    // `__wasm_table` index and a segfault on the very first `ret`).
    // See the analogous comment in `entry_bridge_direct_c` — reserve
    // `HOST_STR_SCRATCH_BYTES` above the stack for host-import stubs that
    // copy host-owned data into a guest-visible location.
    let mem_top = format!(
        "((uint64_t)__wasm_mem_pages * 65536u - {}u)",
        crate::HOST_STR_SCRATCH_BYTES
    );
    let (sp_init, seed_stack_ra, seed_lr) = match lr_param_index {
        Some(lr) => (
            format!("({mem_top}) & ~(uint64_t)0xF"),
            String::new(),
            format!("    __speet_set_reg_seed({lr}, {halt_guest_pc}ULL);\n", lr = lr),
        ),
        None => (
            format!("(({mem_top}) & ~(uint64_t)0xF) - 8"),
            format!("    *(uint64_t *)(uintptr_t)(__wasm_mem + __speet_guest_sp) = {halt_guest_pc}ULL;\n"),
            String::new(),
        ),
    };

    let zero_seeds = (0..n)
        .filter(|&i| i != sp_idx && lr_param_index != Some(i as u32))
        .map(|i| format!("    __speet_set_reg_seed({i}, 0);\n"))
        .collect::<String>();

    format!(
        r#"#include <stdint.h>
#include <unistd.h>

int __speet_argc = 0;
char **__speet_argv = 0;

void __speet_set_argv(int argc, char **argv) {{
    __speet_argc = argc;
    __speet_argv = argv;
}}

void __speet_set_reg_seed(uint32_t idx, uint64_t val);

// See `entry_bridge_direct_c`'s doc comment: the guest's stack lives inside
// `__wasm_mem` (SP is a small offset from its base), not a separate host
// buffer — a raw host pointer here gets silently truncated to 32 bits by
// every WASM-memory-relative access the guest's SP-based loads/stores
// compile down to.
extern uint8_t *__wasm_mem;
extern uint32_t __wasm_mem_pages;

extern uint64_t {entry_stub_sym}(void);

void __speet_start(int argc, char **argv) {{
    __speet_set_argv(argc, argv);
    uint64_t __speet_guest_sp = {sp_init};
{seed_stack_ra}{seed_lr}{zero_seeds}    __speet_set_reg_seed({sp_idx}, __speet_guest_sp);
    long ret = (long){entry_stub_sym}();
    _exit((int)ret);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_halt_guest_pc_into_the_link_register() {
        let src = entry_bridge_c(72, 71, Some(30), "__speet_guest_fn_1", 0x100000370);
        assert!(src.contains("__speet_guest_fn_1"));
        assert!(src.contains("__speet_set_reg_seed(71, __speet_guest_sp)"));
        // The seeded value must be the *guest* halt address as a plain
        // numeric constant, not a native symbol's own address — the guest's
        // `ret` handling treats whatever is here as a guest PC and dispatches
        // through `__wasm_table` accordingly (see `entry_bridge_with_catalog`'s
        // doc comment).
        assert!(src.contains(&format!("__speet_set_reg_seed(30, {}ULL)", 0x100000370u64)));
        assert!(!src.contains("(uint64_t)(uintptr_t)__speet_guest_fn"));
    }

    #[test]
    fn seeds_stack_return_address_for_x86_style() {
        let src = entry_bridge_c(5, 4, None, "__speet_guest_fn_0", 0x100000370);
        assert!(src.contains(&format!(
            "*(uint64_t *)(uintptr_t)(__wasm_mem + __speet_guest_sp) = {}ULL;",
            0x100000370u64
        )));
    }

    #[test]
    fn direct_bridge_pads_for_x86_64_but_not_aarch64() {
        let x86 = entry_bridge_direct_c(9, 4, 0x100000370, Some(8), 6);
        assert!(x86.contains("__speet_d0") && x86.contains("__speet_d5"), "x86-64 (abi_pad_args=6) must pad");
        // Dummies must come before the real params in both the declaration and the call.
        assert!(x86.find("__speet_d5").unwrap() < x86.find("uint64_t p0").unwrap());

        let aarch64 = entry_bridge_direct_c(9, 4, 0x100000370, Some(8), 0);
        assert!(!aarch64.contains("__speet_d0"), "aarch64 (abi_pad_args=0) must not pad");
        assert!(aarch64.contains("extern long __guest_entry(uint64_t p0"));
    }

    #[test]
    fn direct_bridge_zero_params_still_valid_c() {
        let src = entry_bridge_direct_c(0, 0, 0x100000370, None, 0);
        assert!(src.contains("extern long __guest_entry(void);"));
    }
}
