//! Lower pre-pass `__speet_host_mem_*` import calls to multi-memory ops (memory 1).
//!
//! Produces the canonical multi-memory megabinary consumed by wasmi, wasmtime,
//! and wasm-blitz. Full body rewrite lands incrementally; this module provides
//! the validation gate and idempotent passthrough when no host-mem imports exist.

use std::collections::HashMap;
use wasmparser::{Parser, Payload};

/// Host linear memory index after shimming (guest stays at 0).
pub const HOST_MEMORY_INDEX: u32 = 1;

const HOST_MEM_MODULE: &str = "__speet_host_mem";

/// Collect `__speet_host_mem` import function indices from a pre-shimming module.
pub fn host_mem_import_map(wasm: &[u8]) -> Result<HashMap<u32, String>, String> {
    let mut map = HashMap::new();
    let mut import_func_idx = 0u32;
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| e.to_string())? {
            for imp in reader {
                let imp = imp.map_err(|e| e.to_string())?;
                if let wasmparser::TypeRef::Func(_) = imp.ty {
                    if imp.module == HOST_MEM_MODULE {
                        map.insert(import_func_idx, imp.name.to_string());
                    }
                    import_func_idx += 1;
                }
            }
        }
    }
    Ok(map)
}

/// Rewrite pre-shimming module → canonical multi-memory form.
///
/// Idempotent when the module has no `__speet_host_mem` imports (returns input unchanged).
pub fn lower_host_mem_imports(input: &[u8]) -> Result<Vec<u8>, String> {
    validate_standard_core(input)?;
    let host_imports = host_mem_import_map(input)?;
    if host_imports.is_empty() {
        return Ok(input.to_vec());
    }
    // TODO(phase-2): full InstructionStitched rewrite via wax-core when host imports present.
    Err(format!(
        "host-mem shimming not yet implemented for {} import(s): {:?}",
        host_imports.len(),
        host_imports
    ))
}

/// Validate standard WASM core (required parity gate).
pub fn validate_standard_core(wasm: &[u8]) -> Result<(), String> {
    wasmparser::validate(wasm)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use binary_io::BinArch;

    #[test]
    fn idempotent_without_host_mem_imports() {
        use binary_io::BinArch;
        use crate::frontend::recompile_to_wasm;
        let (wasm, _) = recompile_to_wasm(&[0x90], 0x1000, BinArch::X86_64);
        validate_standard_core(&wasm).unwrap();
        let out = lower_host_mem_imports(&wasm).unwrap();
        assert_eq!(out, wasm);
    }
}
