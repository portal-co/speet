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
/// `entry_stub_sym` / `halt_stub_sym` are link-time host symbols from the
/// guest-function stub catalog (`__speet_guest_fn_N`).
pub fn entry_bridge_c(
    entry_param_count: u32,
    sp_param_index: u32,
    lr_param_index: Option<u32>,
    entry_stub_sym: &str,
    halt_stub_sym: &str,
) -> String {
    entry_bridge_with_catalog(
        entry_param_count,
        sp_param_index,
        lr_param_index,
        entry_stub_sym,
        halt_stub_sym,
    )
}

/// Direct `__guest_entry` bridge for callers without a guest-function catalog.
pub fn entry_bridge_direct_c(
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
        (0..n)
            .map(|i| format!("uint64_t p{i}"))
            .collect::<Vec<_>>()
            .join(", ")
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

    let (sp_init, seed_stack_ra) = match lr_param_index {
        Some(_) => (
            "((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF"
                .to_string(),
            String::new(),
        ),
        None => (
            "(((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF) - 8"
                .to_string(),
            format!(
                "    *(uint64_t *)(uintptr_t)__speet_guest_sp = SPEET_HALT_ADDR;\n",
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

#define SPEET_GUEST_STACK_SIZE (8u * 1024u * 1024u)
static uint8_t __speet_guest_stack[SPEET_GUEST_STACK_SIZE] __attribute__((aligned(16)));

extern long __guest_entry({params_decl});

void __speet_start(int argc, char **argv) {{
    __speet_set_argv(argc, argv);
    uint64_t __speet_guest_sp = {sp_init};
{seed_stack_ra}    long ret = __guest_entry({args});
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
    halt_stub_sym: &str,
) -> String {
    let n = entry_param_count as usize;
    let sp_idx = sp_param_index as usize;

    let (sp_init, seed_stack_ra, seed_lr) = match lr_param_index {
        Some(lr) => (
            "((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF"
                .to_string(),
            String::new(),
            format!(
                "    __speet_set_reg_seed({lr}, (uint64_t)(uintptr_t){halt_stub_sym});\n",
                lr = lr
            ),
        ),
        None => (
            "(((uint64_t)(uintptr_t)(__speet_guest_stack + SPEET_GUEST_STACK_SIZE)) & ~(uint64_t)0xF) - 8"
                .to_string(),
            format!(
                "    *(uint64_t *)(uintptr_t)__speet_guest_sp = (uint64_t)(uintptr_t){halt_stub_sym};\n",
            ),
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

#define SPEET_GUEST_STACK_SIZE (8u * 1024u * 1024u)
static uint8_t __speet_guest_stack[SPEET_GUEST_STACK_SIZE] __attribute__((aligned(16)));

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
    fn uses_entry_and_halt_stub_symbols() {
        let src = entry_bridge_c(72, 71, Some(30), "__speet_guest_fn_1", "__speet_guest_fn_2");
        assert!(src.contains("__speet_guest_fn_1"));
        assert!(src.contains("__speet_guest_fn_2"));
        assert!(src.contains("__speet_set_reg_seed(71, __speet_guest_sp)"));
        assert!(src.contains("__speet_set_reg_seed(30, (uint64_t)(uintptr_t)__speet_guest_fn_2)"));
    }

    #[test]
    fn seeds_stack_return_address_for_x86_style() {
        let src = entry_bridge_c(5, 4, None, "__speet_guest_fn_0", "__speet_guest_fn_9");
        assert!(src.contains(
            "*(uint64_t *)(uintptr_t)__speet_guest_sp = (uint64_t)(uintptr_t)__speet_guest_fn_9;"
        ));
    }
}
