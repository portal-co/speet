//! Speet-only import bridges (not backed by `os_shim_*` core).

use speet_host_api::FuncImport;

use super::BridgeHandler;

pub struct InternalBridgeHandler;

impl BridgeHandler for InternalBridgeHandler {
    fn try_emit(&self, out: &mut String, imp: &FuncImport) -> bool {
        if imp.module != "env" {
            return false;
        }
        let sym = speet_host_api::ImportManifest::external_symbol(imp);
        match imp.name.as_str() {
            "__speet_hint" => {
                out.push_str(&format!("void {sym}(int id) {{ (void)id; }}\n\n"));
                true
            }
            "__speet_log_unreachable" => {
                out.push_str(&format!(
                    "void {sym}(int pc) {{
    fprintf(stderr, \"speet: unsupported instruction at guest pc=0x%x\\n\", (unsigned)pc);
}}

"
                ));
                true
            }
            "__speet_stub_for_pc" => {
                out.push_str(&format!(
                    "extern uint64_t __speet_stub_for_pc(uint64_t guest_pc);
uint64_t {sym}(long guest_pc) {{
    return __speet_stub_for_pc((uint64_t)guest_pc);
}}

"
                ));
                true
            }
            _ => false,
        }
    }
}
