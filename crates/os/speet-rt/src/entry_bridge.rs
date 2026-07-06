//! Entry bridge: preserve `argc`/`argv` for recompiled guest startup.

/// C source compiled alongside the runtime shim.
pub const ENTRY_BRIDGE_C: &str = r#"
#include <stdint.h>

int __speet_argc = 0;
char **__speet_argv = 0;

void __speet_set_argv(int argc, char **argv) {
    __speet_argc = argc;
    __speet_argv = argv;
}

void __speet_start(int argc, char **argv) {
    __speet_set_argv(argc, argv);
    extern long __guest_entry(void);
    __guest_entry();
}
"#;
