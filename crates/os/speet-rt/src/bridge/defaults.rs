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
            // ZeroOffset autogen: BridgeSupport data-pointer stubs with
            // checked-in metadata and no unbound fn-ptr args → rewrite each
            // pointer arg as `__wasm_mem + (unsigned)arg` and call `os_shim_*`.
            // Handlers above take precedence for symbols with custom glue.
            name if speet_abi_stubs::zero_offset_data_pointer_safe(name) => {
                emit_autogen_mem_base_stub(out, &sym, name, &imp.params, &imp.results)
            }
            _ => false,
        }
    }
}

fn emit_autogen_mem_base_stub(
    out: &mut String,
    sym: &str,
    name: &str,
    params: &[speet_host_api::WasmValType],
    results: &[speet_host_api::WasmValType],
) -> bool {
    use speet_host_api::WasmValType;
    let ptr_idxs: std::collections::HashSet<usize> = speet_abi_stubs::pointer_arg_indices(name)
        .unwrap_or(&[])
        .iter()
        .copied()
        .collect();
    let c_ty = |t: &WasmValType| -> &'static str {
        match t {
            WasmValType::I32 => "int",
            WasmValType::I64 => "long",
            WasmValType::F32 => "float",
            WasmValType::F64 => "double",
        }
    };
    let ret_ty = match results.first() {
        None => "void",
        Some(t) => c_ty(t),
    };
    let formals: Vec<String> = params
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{} a{i}", c_ty(t)))
        .collect();
    let actuals: Vec<String> = params
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if ptr_idxs.contains(&i) {
                format!("__wasm_mem + (unsigned)a{i}")
            } else {
                format!("a{i}")
            }
        })
        .collect();
    let formals_s = if formals.is_empty() {
        "void".to_string()
    } else {
        formals.join(", ")
    };
    let actuals_s = actuals.join(", ");
    let bare = name.strip_prefix('_').unwrap_or(name);
    let ret = if results.is_empty() { "" } else { "return " };
    out.push_str(&format!(
        "{ret_ty} {sym}({formals_s}) {{
    {ret}os_shim_{bare}({actuals_s});
}}

"
    ));
    true
}
