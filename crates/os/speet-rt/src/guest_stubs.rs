//! Per-guest-function host stubs and runtime invoke helpers for fn-ptr support.

/// One catalog row passed from the recompiler at link time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestStubEntry {
    pub guest_pc: u64,
    pub wasm_func_idx: u32,
    pub local_func_idx: u32,
    pub is_halt: bool,
}

/// Generate C for guest-function stubs, `__speet_stub_for_pc`, and `__speet_invoke`.
///
/// `abi_pad_args` (see `entry_bridge::entry_bridge_direct_c`'s doc comment
/// for the full explanation) is the number of unused leading `uint64_t`
/// dummy arguments `__speet_invoke` must prepend to its call through
/// `__wasm_table` — 6 for x86-64 output (whose `CallAbi::AllStack` ignores
/// argument registers entirely), 0 for AArch64 output (whose "AllStack" is
/// ordinary AAPCS64 marshalling and already matches a plain C call).
pub fn generate_guest_stubs_c(
    entries: &[GuestStubEntry],
    entry_param_count: u32,
    abi_pad_args: u32,
) -> String {
    let mut out = String::from(
        r#"#include <stdint.h>
#include <stddef.h>

extern uint64_t __wasm_table[];

void __speet_set_reg_seed(uint32_t idx, uint64_t val);
uint64_t __speet_invoke(uint32_t wasm_func_idx);
uint64_t __speet_stub_for_pc(uint64_t guest_pc);

"#,
    );

    out.push_str("typedef struct {\n    uint64_t guest_pc;\n    uint32_t wasm_func_idx;\n    uint32_t local_func_idx;\n    uint8_t is_halt;\n} SpeetGuestFuncEntry;\n\n");

    out.push_str("static const SpeetGuestFuncEntry __speet_func_catalog[] = {\n");
    for e in entries {
        out.push_str(&format!(
            "    {{ 0x{:x}ULL, {}, {}, {} }},\n",
            e.guest_pc,
            e.wasm_func_idx,
            e.local_func_idx,
            u8::from(e.is_halt)
        ));
    }
    out.push_str("};\n\n");
    out.push_str(&format!(
        "static const size_t __speet_func_catalog_len = {};\n\n",
        entries.len()
    ));

    for e in entries {
        out.push_str(&format!(
            "uint64_t __speet_guest_fn_{local}(void) {{\n    return __speet_invoke({wasm}u);\n}}\n\n",
            local = e.local_func_idx,
            wasm = e.wasm_func_idx,
        ));
    }

    out.push_str(
        r#"uint64_t __speet_stub_for_pc(uint64_t guest_pc) {
    for (size_t i = 0; i < __speet_func_catalog_len; ++i) {
        if (__speet_func_catalog[i].guest_pc == guest_pc) {
            uint32_t local = __speet_func_catalog[i].local_func_idx;
            switch (local) {
"#,
    );
    for e in entries {
        out.push_str(&format!(
            "            case {}: return (uint64_t)(uintptr_t)__speet_guest_fn_{};\n",
            e.local_func_idx, e.local_func_idx
        ));
    }
    out.push_str(
        r#"            default: break;
            }
        }
    }
    return 0;
}

"#,
    );

    out.push_str(&format!(
        "#define SPEET_REG_SEED_CAP {entry_param_count}u\n",
        entry_param_count = entry_param_count.max(1)
    ));
    out.push_str(
        r#"static uint64_t __speet_reg_seed[SPEET_REG_SEED_CAP];

void __speet_set_reg_seed(uint32_t idx, uint64_t val) {
    if (idx < SPEET_REG_SEED_CAP) {
        __speet_reg_seed[idx] = val;
    }
}

"#,
    );

    // On backends whose `CallAbi::AllStack` truly marshals *every* param
    // through the stack, ignoring argument registers entirely (currently
    // x86-64's wasm-blitz backend only — see this function's doc comment),
    // a plain `fn(seeds[0], ..., seeds[n-1])` call would have param `i`
    // actually land in a register the callee never reads, while the stack
    // slot the callee reads for `i` holds whatever real arg `i+abi_pad_args`
    // put there (or nothing, once out of range) — every param the callee
    // sees is silently shifted/garbage. Prepending `abi_pad_args` unused
    // `uint64_t` args soaks up that many argument registers, forcing the
    // compiler to spill the real `seeds` starting at stack position 0 —
    // exactly where AllStack's prologue looks for param 0.
    let (params, seeds) = (
        (0..entry_param_count)
            .map(|i| format!("uint64_t p{i}"))
            .collect::<Vec<_>>()
            .join(", "),
        (0..entry_param_count)
            .map(|i| format!("__speet_reg_seed[{i}]"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    let dummy_decl = (0..abi_pad_args)
        .map(|i| format!("uint64_t __speet_d{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let dummy_args = (0..abi_pad_args).map(|_| "0").collect::<Vec<_>>().join(", ");
    let full_params = match (dummy_decl.is_empty(), params.is_empty()) {
        (true, true) => "void".to_string(),
        (true, false) => params,
        (false, true) => dummy_decl,
        (false, false) => format!("{dummy_decl}, {params}"),
    };
    let full_args = match (dummy_args.is_empty(), seeds.is_empty()) {
        (true, _) => seeds,
        (false, true) => dummy_args,
        (false, false) => format!("{dummy_args}, {seeds}"),
    };
    out.push_str(&format!(
        r#"uint64_t __speet_invoke(uint32_t wasm_func_idx) {{
    typedef long (*GuestFn)({full_params});
    GuestFn fn = (GuestFn)(uintptr_t)__wasm_table[wasm_func_idx];
    return (uint64_t)fn({full_args});
}}

"#
    ));

    out
}

/// Symbol for the entry-function stub (`__speet_start` tail-calls this).
pub fn entry_stub_symbol(entry_local_idx: u32) -> String {
    format!("__speet_guest_fn_{entry_local_idx}")
}

/// Symbol for the halt-sentinel stub (return-address seeding target).
pub fn halt_stub_symbol(halt_local_idx: u32) -> String {
    format!("__speet_guest_fn_{halt_local_idx}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_stub_lookup_invoke_and_halt_stub() {
        let entries = vec![
            GuestStubEntry {
                guest_pc: 0x1000,
                wasm_func_idx: 3,
                local_func_idx: 0,
                is_halt: false,
            },
            GuestStubEntry {
                guest_pc: 0x1004,
                wasm_func_idx: 4,
                local_func_idx: 1,
                is_halt: true,
            },
        ];
        let src = generate_guest_stubs_c(&entries, 2, 6);
        assert!(src.contains("__speet_stub_for_pc"));
        assert!(src.contains("__speet_guest_fn_0"));
        assert!(src.contains("__speet_guest_fn_1"));
        assert!(src.contains("__speet_set_reg_seed"));
        assert!(src.contains("__speet_d0"), "x86-64 (abi_pad_args=6) must pad");
    }

    #[test]
    fn zero_abi_pad_args_omits_dummy_params() {
        let entries = vec![GuestStubEntry {
            guest_pc: 0x1000,
            wasm_func_idx: 0,
            local_func_idx: 0,
            is_halt: false,
        }];
        let src = generate_guest_stubs_c(&entries, 2, 0);
        assert!(!src.contains("__speet_d0"), "aarch64 (abi_pad_args=0) must not pad");
    }
}
