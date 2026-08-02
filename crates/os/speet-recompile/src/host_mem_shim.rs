//! Lower pre-pass `__speet_host_mem_*` import calls to multi-memory ops (memory 1).
//!
//! Produces the canonical multi-memory megabinary consumed by wasmi, wasmtime,
//! and wasm-blitz.

use std::collections::HashMap;
use wasm_encoder::{
    reencode::{utils, Error, Reencode},
    Function, ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module,
};
use wasmparser::{Operator, Parser, Payload, TypeRef};

/// Host linear memory index after shimming (guest stays at 0).
pub const HOST_MEMORY_INDEX: u32 = 1;

const HOST_MEM_MODULE: &str = "__speet_host_mem";

const HOST_MEM_TYPE: MemoryType = MemoryType {
    minimum: 1,
    maximum: None,
    memory64: false,
    shared: false,
    page_size_log2: None,
};

/// Collect `__speet_host_mem` import function indices from a pre-shimming module.
pub fn host_mem_import_map(wasm: &[u8]) -> Result<HashMap<u32, String>, String> {
    let mut map = HashMap::new();
    let mut import_func_idx = 0u32;
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| e.to_string())? {
            for imp in reader {
                let imp = imp.map_err(|e| e.to_string())?;
                if let TypeRef::Func(_) = imp.ty {
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
    let (n_import_funcs, n_defined_funcs) = module_func_counts(input)?;
    let func_remap = build_func_remap(n_import_funcs, n_defined_funcs, &host_imports);
    let mut reencoder = HostMemShimReencoder {
        host_imports,
        func_remap,
    };
    let mut module = Module::new();
    utils::parse_core_module(
        &mut reencoder,
        &mut module,
        Parser::new(0),
        input,
    )
    .map_err(reencode_err)?;
    Ok(module.finish())
}

/// Validate standard WASM core (required parity gate).
pub fn validate_standard_core(wasm: &[u8]) -> Result<(), String> {
    wasmparser::validate(wasm)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn reencode_err<E: std::fmt::Display>(err: Error<E>) -> String {
    err.to_string()
}

fn module_func_counts(wasm: &[u8]) -> Result<(u32, u32), String> {
    let mut n_import_funcs = 0u32;
    let mut n_defined_funcs = 0u32;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for imp in reader {
                    if matches!(imp.map_err(|e| e.to_string())?.ty, TypeRef::Func(_)) {
                        n_import_funcs += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                n_defined_funcs = reader
                    .into_iter()
                    .count()
                    .try_into()
                    .map_err(|_| "too many defined functions".to_string())?;
            }
            _ => {}
        }
    }
    Ok((n_import_funcs, n_defined_funcs))
}

fn build_func_remap(
    n_import_funcs: u32,
    n_defined_funcs: u32,
    host_imports: &HashMap<u32, String>,
) -> Vec<Option<u32>> {
    let total = (n_import_funcs + n_defined_funcs) as usize;
    let mut map = vec![None; total];
    let mut new_idx = 0u32;
    for old in 0..n_import_funcs {
        if host_imports.contains_key(&old) {
            continue;
        }
        map[old as usize] = Some(new_idx);
        new_idx += 1;
    }
    let defined_base_new = new_idx;
    for d in 0..n_defined_funcs {
        map[(n_import_funcs + d) as usize] = Some(defined_base_new + d);
    }
    map
}

fn host_mem_load_store_arg(align: u32) -> MemArg {
    MemArg {
        offset: 0,
        align,
        memory_index: HOST_MEMORY_INDEX,
    }
}

fn host_mem_instruction(name: &str) -> Result<Instruction<'static>, String> {
    Ok(match name {
        "load_i32" => Instruction::I32Load(host_mem_load_store_arg(2)),
        "load_i64" => Instruction::I64Load(host_mem_load_store_arg(3)),
        "store_i32" => Instruction::I32Store(host_mem_load_store_arg(2)),
        "store_i64" => Instruction::I64Store(host_mem_load_store_arg(3)),
        "copy" => Instruction::MemoryCopy {
            dst_mem: HOST_MEMORY_INDEX,
            src_mem: HOST_MEMORY_INDEX,
        },
        "fill" => Instruction::MemoryFill(HOST_MEMORY_INDEX),
        other => return Err(format!("unknown __speet_host_mem import: {other}")),
    })
}

struct HostMemShimReencoder {
    host_imports: HashMap<u32, String>,
    func_remap: Vec<Option<u32>>,
}

impl HostMemShimReencoder {
    fn remap_func(&self, old: u32) -> Result<u32, String> {
        if self.host_imports.contains_key(&old) {
            return Err(format!("unlowered host-mem function index {old}"));
        }
        self.func_remap
            .get(old as usize)
            .and_then(|v| *v)
            .ok_or_else(|| format!("invalid function index {old}"))
    }
}

impl Reencode for HostMemShimReencoder {
    type Error = String;

    fn function_index(&mut self, func: u32) -> Result<u32, Error<String>> {
        self.remap_func(func).map_err(Error::UserError)
    }

    fn parse_import(
        &mut self,
        imports: &mut ImportSection,
        import: wasmparser::Import<'_>,
    ) -> Result<(), Error<String>> {
        if import.module == HOST_MEM_MODULE && matches!(import.ty, TypeRef::Func(_)) {
            return Ok(());
        }
        utils::parse_import(self, imports, import)
    }

    fn parse_memory_section(
        &mut self,
        memories: &mut MemorySection,
        section: wasmparser::MemorySectionReader<'_>,
    ) -> Result<(), Error<String>> {
        let mut n = 0u32;
        for memory in section {
            memories.memory(self.memory_type(memory?)?);
            n += 1;
        }
        if n >= 1 && n < 2 {
            memories.memory(HOST_MEM_TYPE);
        }
        Ok(())
    }

    fn parse_custom_section(
        &mut self,
        module: &mut Module,
        section: wasmparser::CustomSectionReader<'_>,
    ) -> Result<(), Error<String>> {
        // rustc's `name` custom section references import indices that disappear
        // once host-mem imports are stripped — drop it rather than remap.
        if matches!(section.as_known(), wasmparser::KnownCustom::Name(_)) {
            return Ok(());
        }
        utils::parse_custom_section(self, module, section)
    }

    fn instruction<'a>(
        &mut self,
        op: Operator<'a>,
    ) -> Result<Instruction<'a>, Error<String>> {
        match op {
            Operator::Call { function_index } => {
                if let Some(name) = self.host_imports.get(&function_index) {
                    return host_mem_instruction(name).map_err(Error::UserError);
                }
                Ok(Instruction::Call(self.remap_func(function_index).map_err(Error::UserError)?))
            }
            Operator::ReturnCall { function_index } => {
                if self.host_imports.contains_key(&function_index) {
                    return Err(Error::UserError(
                        "return_call to __speet_host_mem import".into(),
                    ));
                }
                Ok(Instruction::ReturnCall(
                    self.remap_func(function_index).map_err(Error::UserError)?,
                ))
            }
            other => utils::instruction(self, other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binary_io::BinArch;
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, FunctionSection, ImportSection, MemorySection,
        MemoryType, Module, TypeSection, ValType,
    };
    use wasmparser::Operator;

    fn module_with_host_mem_load_i64() -> Vec<u8> {
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I64]); // import load_i64
        types.ty().function([], [ValType::I64]); // exported main

        let mut imports = ImportSection::new();
        imports.import(
            HOST_MEM_MODULE,
            "load_i64",
            wasm_encoder::EntityType::Function(0),
        );

        let mut funcs = FunctionSection::new();
        funcs.function(1);

        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });

        let mut exports = ExportSection::new();
        exports.export("main", ExportKind::Func, 1);

        let mut code = CodeSection::new();
        let mut main = Function::new([]);
        main.instruction(&Instruction::I32Const(0x200));
        main.instruction(&Instruction::Call(0));
        main.instruction(&Instruction::Return);
        main.instruction(&Instruction::End);
        code.function(&main);

        let mut module = Module::new();
        module.section(&types);
        module.section(&imports);
        module.section(&funcs);
        module.section(&mems);
        module.section(&exports);
        module.section(&code);
        module.finish()
    }

    #[test]
    fn idempotent_without_host_mem_imports() {
        use crate::frontend::recompile_to_wasm;
        let (wasm, _) = recompile_to_wasm(&[0x90], 0x1000, BinArch::X86_64);
        validate_standard_core(&wasm).unwrap();
        let out = lower_host_mem_imports(&wasm).unwrap();
        assert_eq!(out, wasm);
    }

    #[test]
    fn lowers_host_mem_import_to_memory_load() {
        let wasm = module_with_host_mem_load_i64();
        validate_standard_core(&wasm).unwrap();
        assert_eq!(host_mem_import_map(&wasm).unwrap().len(), 1);

        let out = lower_host_mem_imports(&wasm).unwrap();
        validate_standard_core(&out).unwrap();
        assert!(host_mem_import_map(&out).unwrap().is_empty());

        let mut memory_count = 0u32;
        for payload in Parser::new(0).parse_all(&out) {
            if let Payload::MemorySection(reader) = payload.unwrap() {
                memory_count = reader.into_iter().count() as u32;
            }
        }
        assert_eq!(memory_count, 2, "single guest memory gains host memory at index 1");

        let body = extract_first_function_body(&out);
        let mut ops = body.get_operators_reader().unwrap();
        assert!(matches!(ops.read().unwrap(), Operator::I32Const { value: 0x200 }));
        match ops.read().unwrap() {
            Operator::I64Load { memarg } => {
                assert_eq!(memarg.memory, HOST_MEMORY_INDEX);
            }
            other => panic!("expected I64Load on host memory, got {other:?}"),
        }
        assert!(matches!(ops.read().unwrap(), Operator::Return));
    }

    fn extract_first_function_body(wasm: &[u8]) -> wasmparser::FunctionBody<'_> {
        for payload in Parser::new(0).parse_all(wasm) {
            if let Payload::CodeSectionEntry(body) = payload.unwrap() {
                return body;
            }
        }
        panic!("no code section");
    }
}
