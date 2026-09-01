//! Backend drive: lower a WASM module to blitz-c / blitz-js source.
//!
//! Parallel to [`crate::drive::compile_wasm_to_object`] for the native backends.
//! Used by e2e B-wasm (PathKind::BlitzC / BlitzJs) compile-checks.

use portal_solutions_blitz_c::{c_emit_import_decls, c_module_preamble, CWrite, State as CState};
use portal_solutions_blitz_common::{
    dce_pass,
    ops::mach_operators,
    wasm_encoder::{self, reencode::RoundtripReencoder},
    wasmparser,
};
use portal_solutions_blitz_js::{js_emit_imports, js_module_preamble, JsWrite, State as JsState};

fn parse_sigs(
    wasm: &[u8],
) -> (
    Vec<wasmparser::FuncType>,
    Vec<wasm_encoder::FuncType>,
    Vec<u32>,
) {
    let mut sigs_wp: Vec<wasmparser::FuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        match payload {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader.into_iter().flatten() {
                    for subtype in group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            subtype.composite_type.inner
                        {
                            sigs_wp.push(ft);
                        }
                    }
                }
            }
            wasmparser::Payload::ImportSection(reader) => {
                for imp in reader.into_iter().flatten() {
                    if let wasmparser::TypeRef::Func(ty_idx) = imp.ty {
                        fsigs.push(ty_idx);
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                fsigs.extend(reader.into_iter().flatten());
            }
            _ => {}
        }
    }
    let sigs_enc: Vec<wasm_encoder::FuncType> = sigs_wp
        .iter()
        .cloned()
        .map(|ft| wasm_encoder::FuncType::try_from(ft).expect("func type"))
        .collect();
    (sigs_wp, sigs_enc, fsigs)
}

fn function_bodies(wasm: &[u8]) -> Vec<wasmparser::FunctionBody<'_>> {
    let mut bodies = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            bodies.push(body);
        }
    }
    bodies
}

fn function_imports(wasm: &[u8]) -> Vec<(String, String)> {
    let mut imports = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ImportSection(reader) = payload {
            for imp in reader.into_iter().flatten() {
                if matches!(imp.ty, wasmparser::TypeRef::Func(_)) {
                    imports.push((imp.module.to_string(), imp.name.to_string()));
                }
            }
        }
    }
    imports
}

fn parse_tags(wasm: &[u8]) -> Vec<u32> {
    let mut result = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::TagSection(reader) = payload {
            for tag in reader.into_iter().flatten() {
                result.push(tag.func_type_idx);
            }
        }
    }
    result
}

/// Compile a WASM megabinary to blitz-c source (with memory preamble + import decls).
pub fn compile_wasm_to_c(wasm: &[u8]) -> Result<String, String> {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let bodies = function_bodies(wasm);
    let imports = function_imports(wasm);
    let tags = parse_tags(wasm);
    let import_count = imports.len() as u32;
    let import_refs: Vec<(&str, &str)> = imports
        .iter()
        .map(|(m, n)| (m.as_str(), n.as_str()))
        .collect();

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies,
        &fsigs,
        &sigs_wp,
        import_count,
    );
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    c_module_preamble(&mut out).map_err(|e| e.to_string())?;
    c_emit_import_decls(&mut out, &import_refs, &sigs_enc, &fsigs).map_err(|e| e.to_string())?;

    let mut state = CState::default();
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        CWrite::on_mach(
            &mut out,
            &sigs_enc,
            &fsigs,
            &tags,
            &import_refs,
            &mut state,
            &op,
            &mut reencoder,
        )
        .map_err(|e| e.to_string())?;
    }
    if out.is_empty() {
        return Err("blitz-c produced empty source".into());
    }
    Ok(out)
}

/// Compile a WASM megabinary to blitz-js source (with memory preamble + import stubs).
pub fn compile_wasm_to_js(wasm: &[u8]) -> Result<String, String> {
    compile_wasm_to_js_with_options(wasm, false)
}

/// Like [`compile_wasm_to_js`], optionally enabling Promise-bail call mode.
pub fn compile_wasm_to_js_with_options(wasm: &[u8], promise_calls: bool) -> Result<String, String> {
    let (sigs_wp, sigs_enc, fsigs) = parse_sigs(wasm);
    let bodies = function_bodies(wasm);
    let imports = function_imports(wasm);
    let tags = parse_tags(wasm);
    let import_count = imports.len() as u32;
    let import_refs: Vec<(&str, &str)> = imports
        .iter()
        .map(|(m, n)| (m.as_str(), n.as_str()))
        .collect();

    let raw_ops = mach_operators::<(), wasmparser::BinaryReaderError>(
        &bodies,
        &fsigs,
        &sigs_wp,
        import_count,
    );
    let ops = dce_pass!(raw_ops);

    let mut out = String::new();
    js_module_preamble(&mut out).map_err(|e| e.to_string())?;
    js_emit_imports(&mut out, &import_refs).map_err(|e| e.to_string())?;

    let mut state = JsState::default();
    if promise_calls {
        state.enable_promise_calls();
    }
    let mut reencoder = RoundtripReencoder;
    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        JsWrite::on_mach(
            &mut out,
            &sigs_enc,
            &fsigs,
            &tags,
            &import_refs,
            &mut state,
            &op,
            &mut reencoder,
        )
        .map_err(|e| e.to_string())?;
    }
    if out.is_empty() {
        return Err("blitz-js produced empty source".into());
    }
    Ok(out)
}
