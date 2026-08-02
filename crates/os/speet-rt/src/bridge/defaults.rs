//! Default BridgeSupport-driven bridges calling shared `os_shim_*` core.

use speet_host_api::FuncImport;

/// Pluggable speet bridge handlers (defaults + speet-internal overrides).
pub struct SpeetBridgeRegistry {
    handlers: Vec<Box<dyn BridgeHandler + Send + Sync>>,
}

impl SpeetBridgeRegistry {
    pub fn integrated() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(super::internal::InternalBridgeHandler));
        reg.register(Box::new(DefaultBridgeHandler));
        reg
    }

    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    pub fn register(&mut self, handler: Box<dyn BridgeHandler + Send + Sync>) {
        self.handlers.push(handler);
    }

    pub fn emit_import_stub(&self, out: &mut String, imp: &FuncImport) {
        for h in &self.handlers {
            if h.try_emit(out, imp) {
                return;
            }
        }
        out.push_str(&format!(
            "/* TODO: stub for {} */\nvoid {}(void) {{ abort(); }}\n\n",
            imp.name,
            speet_host_api::ImportManifest::external_symbol(imp)
        ));
    }
}

pub trait BridgeHandler {
    fn try_emit(&self, out: &mut String, imp: &FuncImport) -> bool;
}

struct DefaultBridgeHandler;

impl BridgeHandler for DefaultBridgeHandler {
    fn try_emit(&self, out: &mut String, imp: &FuncImport) -> bool {
        if imp.module != "env" {
            return false;
        }
        let sym = speet_host_api::ImportManifest::external_symbol(imp);
        match imp.name.as_str() {
            "write" => {
                out.push_str(&format!(
                    "long {sym}(int fd, int ptr, int len) {{
    return os_shim_write(fd, __wasm_mem + (unsigned)ptr, (long)len);
}}

"
                ));
                true
            }
            "exit" => {
                out.push_str(&format!("void {sym}(int code) {{ os_shim_exit(code); }}\n\n"));
                true
            }
            "__speet_execve" => {
                out.push_str(&format!(
                    "int {sym}(long path, long argv, long envp) {{
    return os_shim_execve((const char *)path, (char *const *)argv, (char *const *)envp);
}}

"
                ));
                true
            }
            "printf" => {
                out.push_str(&format!(
                    "int {sym}(long fmt, long fn_ptr) {{
    extern uint64_t __speet_stub_for_pc(uint64_t guest_pc);
    long rewritten = (long)__speet_stub_for_pc((uint64_t)fn_ptr);
    return os_shim_printf((const char *)(uintptr_t)fmt, (void *)(uintptr_t)rewritten);
}}

"
                ));
                true
            }
            "putchar" => {
                out.push_str(&format!("int {sym}(int c) {{ return os_shim_putchar(c); }}\n\n"));
                true
            }
            "strlen" => {
                out.push_str(&format!(
                    "long {sym}(int ptr) {{
    return os_shim_strlen((const char *)(__wasm_mem + (unsigned)ptr));
}}

"
                ));
                true
            }
            "getenv" => {
                out.push_str(&format!(
                    "long {sym}(int name_ptr) {{
    const char *name = (const char *)(__wasm_mem + (unsigned)name_ptr);
    return (long)speet_copy_host_str_to_wasm(os_shim_getenv(name));
}}

"
                ));
                true
            }
            _ => false,
        }
    }
}
