//! Backend drive: lower a WASM module to a native relocatable object.
//!
//! This is the executable core of the recompiler's backend half. Given WASM
//! bytes it runs them through wasm-blitz's SysV/AAPCS64 codegen into a single
//! asm-arch binary writer, surfaces external relocations via
//! `into_parts_with_relocs`, and emits an ELF/Mach-O `.o` via `binary-io`.
//!
//! The entry WASM function (index 0) is exported as `__guest_entry`, a directly
//! C-callable SysV/AAPCS64 symbol (see STATUS.md finding #2): the runtime shim's
//! `main` calls it, passing the guest's initial register values as C arguments.

use binary_io::{
    BinArch, BinOs, DefinedSym, ObjReloc, ObjectInput, ObjectWriter, RelocKind, SymSection,
};
use portal_solutions_blitz_common::{
    dce_pass, ops::mach_operators, wasm_encoder::reencode::RoundtripReencoder, wasmparser,
    HandleOpError,
};

/// The symbol name under which the guest entry (WASM func 0) is exported.
pub const GUEST_ENTRY: &str = "__guest_entry";

/// Parse a WASM module's function signatures (type index per function).
fn parse_sigs(wasm: &[u8]) -> (Vec<wasmparser::FuncType>, Vec<u32>) {
    let mut sigs: Vec<wasmparser::FuncType> = Vec::new();
    let mut fsigs: Vec<u32> = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        match payload {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader.into_iter().flatten() {
                    for subtype in group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            subtype.composite_type.inner
                        {
                            sigs.push(ft);
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
    (sigs, fsigs)
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

/// Imported functions as `(module, name)` pairs, in import order. blitz renders
/// a call to import `i` as the external symbol `{module}__{name}` and offsets
/// internal function indices past the imports.
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

/// Render a reloc-target label to its emitted object symbol name, or `None` for
/// internal (`Func`/`Indexed`) labels that should never appear as relocations.
enum LabelSym {
    External(String),
    Internal(String),
}

/// Compile a WASM module to a native relocatable object (`.o`) for `arch`/`os`.
///
/// Exports `__guest_entry` at the start of the code (WASM func 0). External and
/// ambient labels become undefined-symbol relocations. Memory accesses use the
/// `__wasm_mem` base (provided by `speet-rt`).
pub fn compile_wasm_to_object(
    wasm: &[u8],
    arch: BinArch,
    os: BinOs,
) -> Result<Vec<u8>, String> {
    let _portal_log = speet_log::LlmtrimLogger::from_env();
    _portal_log.log_event("INFO", "drive", "compile_wasm_to_object start", &[("arch", &format!("{arch:?}"))]);    
    let (sigs, fsigs) = parse_sigs(wasm);
    let bodies = function_bodies(wasm);
    let imports = function_imports(wasm);
    let import_count = imports.len() as u32;
    let raw_ops =
        mach_operators::<(), wasmparser::BinaryReaderError>(&bodies, &fsigs, &sigs, import_count);
    let ops = dce_pass!(raw_ops);
    let import_refs: Vec<(&str, &str)> =
        imports.iter().map(|(m, n)| (m.as_str(), n.as_str())).collect();

    // Per-WASM-function-index param/result counts (imports first, then internal),
    // used by the backend to marshal call arguments.
    let call_params: Vec<u32> = fsigs
        .iter()
        .map(|&ti| sigs[ti as usize].params().len() as u32)
        .collect();
    let call_results: Vec<u32> = fsigs
        .iter()
        .map(|&ti| sigs[ti as usize].results().len() as u32)
        .collect();

    _portal_log.log_event("INFO", "drive", "dispatch to backend", &[("arch", &format!("{arch:?}"), ), ("n_funcs", &bodies.len().to_string())]);
    match arch {
        BinArch::AArch64 => compile_aarch64(ops, &import_refs, arch, os),
        BinArch::X86_64 => {
            compile_x86_64(ops, &import_refs, import_count, &call_params, &call_results, arch, os)
        }
    }
}

fn finish_object<L>(
    arch: BinArch,
    os: BinOs,
    text: Vec<u8>,
    labels: impl IntoIterator<Item = (L, usize)>,
    relocs: Vec<(usize, LabelSym, RelocKind, i64)>,
) -> Result<Vec<u8>, String>
where
    L: LabelName,
{
    // Defined symbols: every External label that was placed (set_label'd).
    let mut defined_syms = Vec::new();
    for (label, off) in labels {
        if let LabelSym::External(name) = label.label_sym() {
            defined_syms.push(DefinedSym {
                name,
                section: SymSection::Text,
                offset: off as u64,
                global: true,
            });
        }
    }
    // Relocations: each unresolved external/ambient reference. On Mach-O, C
    // symbols carry a leading underscore; `binary-io` auto-mangles *defined*
    // symbols but not undefined relocation targets, so prepend it here to match
    // the shim/libc definitions (e.g. `env__exit` -> `_env__exit`).
    let mangle = |name: String| match os {
        BinOs::MacOs => format!("_{name}"),
        BinOs::Linux => name,
    };
    // Mach-O uses *implicit* addends (the value lives in the relocated field),
    // while ELF (RELA) carries the addend in the relocation entry. asm-arch's
    // RIP-relative `lea` placeholder is encoded by iced as `disp = -(next_ip)`
    // (it treats the 0 displacement as an absolute target), which pollutes the
    // field. For Mach-O we zero the 4-byte field so the linker resolves a clean
    // `S - next_ip`; the addend (`-4`) becomes 0 via object's pcrel adjustment.
    // (CALL/JMP placeholders are already zero.)
    let mut text = text;
    if os == BinOs::MacOs {
        for (off, sym, _, _) in &relocs {
            if matches!(sym, LabelSym::External(_)) {
                let end = (off + 4).min(text.len());
                for b in &mut text[*off..end] {
                    *b = 0;
                }
            }
        }
    }
    let mut obj_relocs = Vec::new();
    for (off, sym, kind, addend) in relocs {
        match sym {
            LabelSym::External(name) => obj_relocs.push(ObjReloc {
                off: off as u64,
                symbol: mangle(name),
                kind,
                addend,
            }),
            LabelSym::Internal(dbg) => {
                return Err(format!(
                    "internal label {dbg} produced an unresolved relocation at offset {off} (codegen bug)"
                ))
            }
        }
    }

    let input = ObjectInput {
        arch,
        os,
        text: &text,
        data: Vec::new(),
        defined_syms,
        relocs: obj_relocs,
    };
    let bytes = match os {
        BinOs::MacOs => binary_io::MachOObjectWriter::write_object(&input),
        BinOs::Linux => binary_io::ElfObjectWriter::write_object(&input),
    }
    .map_err(|e| format!("object write failed: {e:?}"))?;
    Ok(bytes)
}

/// Trait to extract a neutral symbol classification from an arch-specific label.
trait LabelName {
    fn label_sym(&self) -> LabelSym;
}

fn compile_aarch64<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    arch: BinArch,
    os: BinOs,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_aarch64::out::bin::{AsmRelocKind, AArch64Writer};
    use portal_solutions_asm_aarch64::out::Writer as _;
    use portal_solutions_blitz_aarch64::{naive, sysv, AArch64Arch, AArch64Label};

    impl LabelName for AArch64Label {
        fn label_sym(&self) -> LabelSym {
            match self {
                AArch64Label::External { name } => LabelSym::External(name.clone()),
                AArch64Label::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
    }

    let mut out = AArch64Writer::<AArch64Label>::new();
    let mut ctx = ();
    let archc = AArch64Arch::default();
    let mut state = naive::State::default();
    state.mem_base = naive::MemBase::WasmMemSymbol;
    let mut reencoder = RoundtripReencoder;

    // Export the entry (func 0) at offset 0.
    out.set_label(&mut ctx, archc, AArch64Label::External { name: GUEST_ENTRY.into() })
        .map_err(|e| format!("set_label: {e:?}"))?;

    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out, &mut ctx, archc, &mut state, func_imports, &op, &mut reencoder, 0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels, relocs) = out.into_parts_with_relocs();
    let mapped: Vec<_> = relocs
        .into_iter()
        .map(|r| {
            let kind = match r.kind {
                AsmRelocKind::A64Call26 => RelocKind::A64Call26,
                AsmRelocKind::A64Jump26 => RelocKind::A64Jump26,
                AsmRelocKind::A64AdrPrel21 => RelocKind::A64AdrPrel21,
                AsmRelocKind::A64CondBr19 => RelocKind::A64Jump26, // unreached for externals
            };
            (r.byte_offset, r.label.label_sym(), kind, r.addend)
        })
        .collect();
    finish_object(arch, os, text, labels, mapped)
}

fn compile_x86_64<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    call_params: &[u32],
    call_results: &[u32],
    arch: BinArch,
    os: BinOs,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_x86_64::out::iced::{AsmRelocKind, IcedWriter};
    use portal_solutions_asm_x86_64::out::Writer as _;
    use portal_solutions_blitz_x86_64::{sysv, X64Arch, X64Label};

    impl LabelName for X64Label {
        fn label_sym(&self) -> LabelSym {
            match self {
                X64Label::External { name } => LabelSym::External(name.clone()),
                X64Label::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
    }

    let mut out = IcedWriter::<X64Label>::new(0);
    let mut ctx = ();
    let archc = X64Arch::default();
    let mut state = sysv::SysVState::default();
    state.mem_base = portal_solutions_blitz_x86_64::naive::MemBase::WasmMemSymbol;
    // Use the all-on-stack inter-function ABI so the per-instruction register-file
    // threading round-trips; import calls still use the C ABI (register args).
    state.call_abi = sysv::CallAbi::AllStack;
    state.n_imports = n_imports;
    state.call_params = call_params.to_vec();
    state.call_results = call_results.to_vec();
    let mut reencoder = RoundtripReencoder;

    out.set_label(&mut ctx, archc, X64Label::External { name: GUEST_ENTRY.into() })
        .map_err(|e| format!("set_label: {e:?}"))?;

    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out, &mut ctx, archc, &mut state, func_imports, &op, &mut reencoder, 0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels, relocs) = out.into_parts_with_relocs();
    let mapped: Vec<_> = relocs
        .into_iter()
        .map(|r| {
            let kind = match r.kind {
                AsmRelocKind::X86Branch32 => RelocKind::X86Plt32,
                AsmRelocKind::X86Pcrel32 => RelocKind::X86Pc32,
            };
            (r.byte_offset, r.label.label_sym(), kind, r.addend)
        })
        .collect();
    finish_object(arch, os, text, labels, mapped)
}
