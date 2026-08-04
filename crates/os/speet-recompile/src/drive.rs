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
    BinArch, BinOs, DataBlob, DataReloc, DefinedSym, ObjReloc, ObjectInput, ObjectWriter,
    RelocKind, SymSection,
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

/// Number of WASM function imports (host capability slots).
pub fn import_func_count(wasm: &[u8]) -> u32 {
    function_imports(wasm).len() as u32
}

/// Number of defined functions in the code section (translated slots + halt).
pub fn code_func_count(wasm: &[u8]) -> u32 {
    function_bodies(wasm).len() as u32
}

/// Local function index exported as `_start` (before `n_imports` offset).
pub fn entry_local_func_idx(wasm: &[u8]) -> u32 {
    let n_imports = import_func_count(wasm);
    entry_export_func_idx(wasm).saturating_sub(n_imports)
}

/// The WASM function index the module exports as `_start` — the guest's
/// real entry point (see [`crate::frontend::Translated::entry_func_idx`]),
/// not necessarily function 0. Falls back to `0` if the module doesn't
/// export `_start` under that exact name (defensive; every module this
/// backend actually receives is expected to, per `frontend::finish_module`).
fn entry_export_func_idx(wasm: &[u8]) -> u32 {
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(reader) = payload {
            for exp in reader.into_iter().flatten() {
                if exp.name == "_start" && matches!(exp.kind, wasmparser::ExternalKind::Func) {
                    return exp.index;
                }
            }
        }
    }
    0
}

/// Whether the module exports [`crate::frontend::DATA_INIT_EXPORT_NAME`].
pub fn has_data_init_export(wasm: &[u8]) -> bool {
    use crate::frontend::DATA_INIT_EXPORT_NAME;
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(reader) = payload {
            for exp in reader.into_iter().flatten() {
                if exp.name == DATA_INIT_EXPORT_NAME && matches!(exp.kind, wasmparser::ExternalKind::Func) {
                    return true;
                }
            }
        }
    }
    false
}

/// The WASM function index the module exports under
/// [`crate::frontend::DATA_INIT_EXPORT_NAME`], if any — `finish_module` only
/// emits that export when the guest has data segments, so this is `None` for
/// every guest without any.
fn data_init_export_func_idx(wasm: &[u8]) -> Option<u32> {
    for payload in wasmparser::Parser::new(0).parse_all(wasm).flatten() {
        if let wasmparser::Payload::ExportSection(reader) = payload {
            for exp in reader.into_iter().flatten() {
                if exp.name == crate::frontend::DATA_INIT_EXPORT_NAME
                    && matches!(exp.kind, wasmparser::ExternalKind::Func)
                {
                    return Some(exp.index);
                }
            }
        }
    }
    None
}

/// Param count of the module's `_start` export's function type — i.e. the
/// number of arguments a C caller must supply to call `__guest_entry`
/// correctly (guest registers are WASM params; see
/// `docs/guides/thin-runtime-genericity.md`). Falls back to `0` if the
/// module doesn't export `_start`. Callers building the runtime shim
/// combine this with the arch's `SP_PARAM_INDEX` constant (`speet_aarch64`/
/// `speet_x86_64`) to seed a valid guest stack at the right argument
/// position instead of leaving it as caller-side garbage.
pub fn entry_param_count(wasm: &[u8]) -> u32 {
    let entry_idx = entry_export_func_idx(wasm);
    let (sigs, fsigs) = parse_sigs(wasm);
    fsigs
        .get(entry_idx as usize)
        .and_then(|&ti| sigs.get(ti as usize))
        .map(|t| t.params().len() as u32)
        .unwrap_or(0)
}

/// The guest SP register's WASM param index for `arch` — see
/// `speet_aarch64::AArch64Recompiler::SP_PARAM_INDEX` /
/// `speet_x86_64::X86Recompiler::SP_PARAM_INDEX`. Callers building a
/// C-callable entry bridge combine this with [`entry_param_count`] to seed a
/// freshly allocated guest stack at the right argument position instead of
/// leaving it as caller-side garbage — see
/// `docs/guides/thin-runtime-genericity.md`.
pub fn sp_param_index(arch: BinArch) -> u32 {
    match arch {
        BinArch::AArch64 => speet_aarch64::AArch64Recompiler::<(), ()>::SP_PARAM_INDEX,
        BinArch::X86_64 => speet_x86_64::X86Recompiler::<(), ()>::SP_PARAM_INDEX,
        BinArch::RiscV64 | BinArch::RiscV32 => speet_riscv::RV64_SP_PARAM_INDEX,
        // AArch32: SP is r13 in a flat r0–r15 register-file ABI (Phase 4 frontend).
        BinArch::Arm => 13,
        // i686: ESP occupies the same param slot index as x86_64's RSP.
        BinArch::X86 => speet_x86_64::X86Recompiler::<(), ()>::SP_PARAM_INDEX,
    }
}

/// The guest link-register's WASM param index for `arch`, or `None` for an
/// arch whose `ret` reads the return address from guest memory rather than
/// a dedicated register (x86-64) — see `speet_rt::entry_bridge_c`'s
/// `lr_param_index` and `docs/guides/thin-runtime-genericity.md` principle
/// 4 (the halt-sentinel contract this seeds).
pub fn lr_param_index(arch: BinArch) -> Option<u32> {
    match arch {
        BinArch::AArch64 => Some(speet_aarch64::AArch64Recompiler::<(), ()>::LR_PARAM_INDEX),
        BinArch::RiscV64 | BinArch::RiscV32 => Some(speet_riscv::RV64_RA_PARAM_INDEX),
        // AArch32: LR is r14.
        BinArch::Arm => Some(14),
        BinArch::X86_64 | BinArch::X86 => None,
    }
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
    // Internal (post-import) function index of the module's real entry —
    // see `entry_export_func_idx`. Do NOT assume `_start` is always
    // function 0: a linked binary's `.text` commonly holds several
    // functions ahead of the one at the actual entry address (see
    // `docs/guides/thin-runtime-genericity.md` principle 2).
    let entry_func_idx = entry_export_func_idx(wasm).saturating_sub(import_count);
    // Same idea for the optional data-init function (see
    // `data_init_export_func_idx`): `None` when the guest has no data
    // segments, so `finish_module` never emitted the export.
    let data_init_func_idx = data_init_export_func_idx(wasm).map(|i| i.saturating_sub(import_count));
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

    // Per-WASM-*type*-index param/result counts, used by `call_indirect`
    // (whose operand is a type index, not a function index).
    let sig_params: Vec<u32> = sigs.iter().map(|t| t.params().len() as u32).collect();
    let sig_results: Vec<u32> = sigs.iter().map(|t| t.results().len() as u32).collect();

    _portal_log.log_event("INFO", "drive", "dispatch to backend", &[("arch", &format!("{arch:?}"), ), ("n_funcs", &bodies.len().to_string())]);
    match arch {
        BinArch::AArch64 => compile_aarch64(
            ops, &import_refs, import_count, &call_params, &call_results, &sig_params,
            &sig_results, arch, os, entry_func_idx, data_init_func_idx,
        ),
        BinArch::X86_64 => compile_x86_64(
            ops, &import_refs, import_count, &call_params, &call_results, &sig_params,
            &sig_results, arch, os, entry_func_idx, data_init_func_idx,
        ),
        BinArch::RiscV64 => Err(
            "host-riscv object emission is not wired (thin runtime hosts x86_64/aarch64)".into(),
        ),
        BinArch::RiscV32 => compile_riscv32(
            ops, &import_refs, import_count, arch, os, entry_func_idx, data_init_func_idx,
        ),
        BinArch::Arm => compile_arm(
            ops, &import_refs, import_count, arch, os, entry_func_idx, data_init_func_idx,
        ),
        BinArch::X86 => compile_x86(
            ops, &import_refs, import_count, arch, os, entry_func_idx, data_init_func_idx,
        ),
    }
}

fn finish_object<L>(
    arch: BinArch,
    os: BinOs,
    text: Vec<u8>,
    labels: impl IntoIterator<Item = (L, usize)>,
    relocs: Vec<(usize, LabelSym, RelocKind, i64)>,
    n_imports: u32,
    func_imports: &[(&str, &str)],
) -> Result<Vec<u8>, String>
where
    L: LabelName,
{
    if matches!(arch, BinArch::RiscV32 | BinArch::Arm | BinArch::X86) && os == BinOs::MacOs {
        return Err("ILP32 Mach-O object emission is unsupported (Linux ELF only)".into());
    }
    let mangle = |name: String| match os {
        BinOs::MacOs => format!("_{name}"),
        BinOs::Linux => name,
    };

    // Defined symbols: every External label that was placed (set_label'd), plus a
    // `__wasm_func_N` local symbol per internal function entry (so `__wasm_table`
    // slots can take its address).
    let mut defined_syms = Vec::new();
    let mut func_offsets: Vec<(u32, u64)> = Vec::new();
    for (label, off) in labels {
        if let LabelSym::External(name) = label.label_sym() {
            defined_syms.push(DefinedSym {
                name,
                section: SymSection::Text,
                offset: off as u64,
                global: true,
            });
        }
        if let Some(k) = label.func_index() {
            defined_syms.push(DefinedSym {
                name: format!("__wasm_func_{k}"),
                section: SymSection::Text,
                offset: off as u64,
                global: false,
            });
            func_offsets.push((k, off as u64));
        }
    }

    // Build `__wasm_table`: an identity-mapped function-pointer table over every
    // WASM function index (imports first, then internal). `call_indirect` indexes
    // it with the guest function index. Slot i < n_imports points at the import's
    // external symbol; otherwise at the `__wasm_func_{i-n_imports}` local symbol.
    let n_internal = func_offsets.iter().map(|(k, _)| *k + 1).max().unwrap_or(0);
    let n_slots = n_imports + n_internal;
    let (abs_kind, ptr_size) = match arch {
        BinArch::X86_64 => (RelocKind::X86Abs64, 8u64),
        BinArch::AArch64 => (RelocKind::A64Abs64, 8u64),
        BinArch::RiscV64 => {
            return Err("RiscV64 abs64 reloc not modeled in binary-io yet".into());
        }
        BinArch::X86 => (RelocKind::X86Abs32, 4u64),
        BinArch::Arm => (RelocKind::ArmAbs32, 4u64),
        BinArch::RiscV32 => (RelocKind::RiscVAbs32, 4u64),
    };
    let mut table_relocs = Vec::with_capacity(n_slots as usize);
    for i in 0..n_slots {
        let symbol = if i < n_imports {
            let (m, n) = func_imports[i as usize];
            mangle(format!("{m}__{n}"))
        } else {
            format!("__wasm_func_{}", i - n_imports)
        };
        table_relocs.push(DataReloc {
            off: (i as u64) * ptr_size,
            symbol,
            kind: abs_kind,
            addend: 0,
        });
    }
    let data = vec![DataBlob {
        name: "__wasm_table".to_string(),
        bytes: vec![0u8; n_slots as usize * ptr_size as usize],
        align: ptr_size,
        // Writable so the absolute function-pointer relocations are permitted:
        // macOS rejects text-relocations in a read-only section (they would need
        // load-time fixups dyld only applies to writable data).
        writable: true,
        relocs: table_relocs,
    }];
    defined_syms.push(DefinedSym {
        name: "__wasm_table".to_string(),
        section: SymSection::Data(0),
        offset: 0,
        global: true,
    });
    // Relocations: each unresolved external/ambient reference. On Mach-O, C
    // symbols carry a leading underscore; `binary-io` auto-mangles *defined*
    // symbols but not undefined relocation targets, so prepend it here to match
    // the shim/libc definitions (e.g. `env__exit` -> `_env__exit`).
    // Mach-O uses *implicit* addends (the value lives in the relocated field),
    // while ELF (RELA) carries the addend in the relocation entry. On x86-64,
    // asm-arch's RIP-relative `lea` placeholder is encoded by iced as
    // `disp = -(next_ip)` (it treats the 0 displacement as an absolute target),
    // which pollutes the field, so we zero the 4-byte rel32 so the linker
    // resolves a clean `S - next_ip`. AArch64 emits ADRP/ADD with a *clean* zero
    // immediate already, and the opcode bits must survive — so this x86-only
    // workaround must NOT run there (it would destroy the instruction).
    let mut text = text;
    if os == BinOs::MacOs && arch == BinArch::X86_64 {
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
            // `__wasm_table` is defined in *this* object, so it must not be
            // pre-mangled — its `DefinedSym` keys `binary-io`'s symbol map by the
            // bare name (the format writer applies any leading underscore itself).
            LabelSym::External(name) if name == "__wasm_table" => obj_relocs.push(ObjReloc {
                off: off as u64,
                symbol: name,
                kind,
                addend,
            }),
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
        data,
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
    /// If this label marks an internal function entry, its 0-based internal
    /// function id (i.e. `wasm_func_idx - n_imports`). Used to emit the
    /// `__wasm_func_N` symbols that `__wasm_table` slots point at.
    fn func_index(&self) -> Option<u32> {
        None
    }
}

fn compile_aarch64<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    call_params: &[u32],
    call_results: &[u32],
    sig_params: &[u32],
    sig_results: &[u32],
    arch: BinArch,
    os: BinOs,
    entry_func_idx: u32,
    data_init_func_idx: Option<u32>,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_aarch64::out::bin::{AsmRelocKind, AArch64Writer};
    use portal_solutions_asm_aarch64::out::Writer as _;
    use portal_solutions_blitz_aarch64::{sysv, AArch64Arch, AArch64Label};

    impl LabelName for AArch64Label {
        fn label_sym(&self) -> LabelSym {
            match self {
                AArch64Label::External { name } => LabelSym::External(name.clone()),
                AArch64Label::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
        fn func_index(&self) -> Option<u32> {
            // aarch64 function entries use `Indexed { id + 0x8000_0000 }`.
            match self {
                AArch64Label::Indexed { idx } if *idx >= 0x8000_0000 => {
                    Some((*idx - 0x8000_0000) as u32)
                }
                _ => None,
            }
        }
    }

    let mut out = AArch64Writer::<AArch64Label>::new();
    let mut ctx = ();
    let archc = AArch64Arch::default();
    // `sysv::{SysVState, CallAbi, MemBase}`, not `naive::*` — the "naive"
    // module is slated for deprecation and is a frequent source of ABI
    // confusion (see `docs/naive-abi-deprecation.md` in `wasm-blitz`); its
    // `State`/`CallAbi` are what `SysVWriterExt` actually configures to
    // produce genuinely C-callable AAPCS64 code, despite the module name.
    let mut state = sysv::SysVState::default();
    state.mem_base = sysv::MemBase::WasmMemSymbol;
    // Marshal the full guest register file per AAPCS64 (X0-X7 then stack), matching
    // the SysV prologue, so inter-function tail calls thread all params correctly.
    state.call_abi = sysv::CallAbi::AllStack;
    state.n_imports = n_imports;
    state.call_params = call_params.to_vec();
    state.call_results = call_results.to_vec();
    state.sig_params = sig_params.to_vec();
    state.sig_results = sig_results.to_vec();
    let mut reencoder = RoundtripReencoder;

    // Export the guest entry at the *actual* entry function's offset —
    // never unconditionally function 0 (`_start` names which function that
    // is; see `entry_export_func_idx`/`docs/guides/thin-runtime-genericity.md`
    // principle 2). Watch for `StartFn { id, .. }` matching `entry_func_idx`
    // and place the label right there, before that function's code is
    // emitted.
    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            if *id == entry_func_idx {
                out.set_label(&mut ctx, archc, AArch64Label::External { name: GUEST_ENTRY.into() })
                    .map_err(|e| format!("set_label: {e:?}"))?;
            }
            if data_init_func_idx == Some(*id) {
                out.set_label(
                    &mut ctx,
                    archc,
                    AArch64Label::External { name: crate::frontend::DATA_INIT_EXPORT_NAME.into() },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
        }
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out, &mut ctx, archc, &mut state, func_imports, &op, &mut reencoder, 0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels, relocs) = out.into_parts_with_relocs();
    let mut mapped = Vec::with_capacity(relocs.len());
    for r in relocs {
        let kind = match r.kind {
            AsmRelocKind::A64Call26 => RelocKind::A64Call26,
            AsmRelocKind::A64Jump26 => RelocKind::A64Jump26,
            // asm-arch `A64AdrPrel21` is plain ADR. Mach-O has no ADR reloc
            // (binary-io historically aliased the name to PAGE21, which made
            // `ld` reject "PAGE21 on non-ADRP"). External addresses must be
            // materialized with ADRP+ADD via blitz `load_label_addr`.
            AsmRelocKind::A64AdrPrel21 => {
                let sym = match r.label.label_sym() {
                    LabelSym::External(n) => n,
                    LabelSym::Internal(n) => n,
                };
                return Err(format!(
                    "unresolved ADR reloc at text+{:#x} for {sym} — use ADRP+ADD \
                     (blitz load_label_addr) for external/ambient symbols on aarch64",
                    r.byte_offset,
                ));
            }
            AsmRelocKind::A64AdrpPage21 => RelocKind::A64AdrpPage21,
            AsmRelocKind::A64AddAbsLo12 => RelocKind::A64AddAbsLo12,
            AsmRelocKind::A64CondBr19 => RelocKind::A64Jump26, // unreached for externals
        };
        mapped.push((r.byte_offset, r.label.label_sym(), kind, r.addend));
    }
    finish_object(arch, os, text, labels, mapped, n_imports, func_imports)
}

fn compile_x86_64<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    call_params: &[u32],
    call_results: &[u32],
    sig_params: &[u32],
    sig_results: &[u32],
    arch: BinArch,
    os: BinOs,
    entry_func_idx: u32,
    data_init_func_idx: Option<u32>,
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
        fn func_index(&self) -> Option<u32> {
            match self {
                X64Label::Func { r#fn } => Some(*r#fn),
                _ => None,
            }
        }
    }

    let mut out = IcedWriter::<X64Label>::new(0);
    let mut ctx = ();
    let archc = X64Arch::default();
    let mut state = sysv::SysVState::default();
    state.mem_base = sysv::MemBase::WasmMemSymbol;
    // Use the all-on-stack inter-function ABI so the per-instruction register-file
    // threading round-trips; import calls still use the C ABI (register args).
    state.call_abi = sysv::CallAbi::AllStack;
    state.n_imports = n_imports;
    state.call_params = call_params.to_vec();
    state.call_results = call_results.to_vec();
    state.sig_params = sig_params.to_vec();
    state.sig_results = sig_results.to_vec();
    let mut reencoder = RoundtripReencoder;

    // See the matching comment in `compile_aarch64`: `_start` (not
    // necessarily function 0) names the real entry.
    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            if *id == entry_func_idx {
                out.set_label(&mut ctx, archc, X64Label::External { name: GUEST_ENTRY.into() })
                    .map_err(|e| format!("set_label: {e:?}"))?;
            }
            if data_init_func_idx == Some(*id) {
                out.set_label(
                    &mut ctx,
                    archc,
                    X64Label::External { name: crate::frontend::DATA_INIT_EXPORT_NAME.into() },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
        }
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
    finish_object(arch, os, text, labels, mapped, n_imports, func_imports)
}

fn compile_riscv32<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    arch: BinArch,
    os: BinOs,
    entry_func_idx: u32,
    data_init_func_idx: Option<u32>,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_riscv32::out::rv_asm_backend::RvAsmWriter;
    use portal_solutions_asm_riscv32::out::Writer as _;
    use portal_solutions_blitz_riscv32::{sysv, RiscV32Arch, RiscvLabel};

    impl LabelName for RiscvLabel {
        fn label_sym(&self) -> LabelSym {
            match self {
                RiscvLabel::External { name } => LabelSym::External(name.clone()),
                RiscvLabel::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
        fn func_index(&self) -> Option<u32> {
            match self {
                RiscvLabel::Func { r#fn } => Some(*r#fn),
                RiscvLabel::Indexed { idx } if *idx & (1 << 28) != 0 => {
                    Some((*idx & !(1 << 28)) as u32)
                }
                _ => None,
            }
        }
    }

    let mut out = RvAsmWriter::<RiscvLabel>::new();
    let mut ctx = ();
    let archc = RiscV32Arch::default();
    let mut state = sysv::SysVState::default();
    state.mem_base = sysv::MemBase::WasmMemSymbol;
    state.call_abi = sysv::CallAbi::AllStack;
    state.n_imports = n_imports;
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            if *id == entry_func_idx {
                out.set_label(
                    &mut ctx,
                    archc,
                    RiscvLabel::External {
                        name: GUEST_ENTRY.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
            if data_init_func_idx == Some(*id) {
                out.set_label(
                    &mut ctx,
                    archc,
                    RiscvLabel::External {
                        name: crate::frontend::DATA_INIT_EXPORT_NAME.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
        }
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            archc,
            &mut state,
            func_imports,
            &op,
            &mut reencoder,
            0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels) = out.into_parts();
    finish_object(arch, os, text, labels, Vec::new(), n_imports, func_imports)
}

fn compile_arm<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    arch: BinArch,
    os: BinOs,
    entry_func_idx: u32,
    data_init_func_idx: Option<u32>,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_arm::out::bin::ArmWriter;
    use portal_solutions_asm_arm::out::Writer as _;
    use portal_solutions_blitz_arm::{sysv, ArmArch, ArmLabel};

    impl LabelName for ArmLabel {
        fn label_sym(&self) -> LabelSym {
            match self {
                ArmLabel::External { name } => LabelSym::External(name.clone()),
                ArmLabel::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
        fn func_index(&self) -> Option<u32> {
            match self {
                ArmLabel::Func { r#fn } => Some(*r#fn),
                _ => None,
            }
        }
    }

    let mut out = ArmWriter::<ArmLabel>::new();
    let mut ctx = ();
    let archc = ArmArch::default();
    let mut state = sysv::SysVState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            if *id == entry_func_idx {
                out.set_label(
                    &mut ctx,
                    archc,
                    ArmLabel::External {
                        name: GUEST_ENTRY.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
            if data_init_func_idx == Some(*id) {
                out.set_label(
                    &mut ctx,
                    archc,
                    ArmLabel::External {
                        name: crate::frontend::DATA_INIT_EXPORT_NAME.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
        }
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            archc,
            &mut state,
            func_imports,
            &op,
            &mut reencoder,
            0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels) = out.into_parts();
    finish_object(arch, os, text, labels, Vec::new(), n_imports, func_imports)
}

fn compile_x86<'a>(
    ops: impl IntoIterator<Item = Result<portal_solutions_blitz_common::MachOperator<'a, ()>, wasmparser::BinaryReaderError>>,
    func_imports: &[(&str, &str)],
    n_imports: u32,
    arch: BinArch,
    os: BinOs,
    entry_func_idx: u32,
    data_init_func_idx: Option<u32>,
) -> Result<Vec<u8>, String> {
    use portal_solutions_asm_x86::out::iced::IcedWriter;
    use portal_solutions_asm_x86::out::Writer as _;
    use portal_solutions_blitz_i686::{sysv, I686Label, X86Arch};

    impl LabelName for I686Label {
        fn label_sym(&self) -> LabelSym {
            match self {
                I686Label::External { name } => LabelSym::External(name.clone()),
                I686Label::Ambient { name } => LabelSym::External(format!("__ambient_{name}")),
                other => LabelSym::Internal(format!("{other:?}")),
            }
        }
        fn func_index(&self) -> Option<u32> {
            match self {
                I686Label::Func { r#fn } => Some(*r#fn),
                _ => None,
            }
        }
    }

    let mut out = IcedWriter::<I686Label>::new(0);
    let mut ctx = ();
    let archc = X86Arch::default();
    let mut state = sysv::SysVState::default();
    let mut reencoder = RoundtripReencoder;

    for op in ops {
        let op = op.map_err(|e| format!("mach op: {e:?}"))?;
        if let portal_solutions_blitz_common::MachOperator::StartFn { id, .. } = &op {
            if *id == entry_func_idx {
                out.set_label(
                    &mut ctx,
                    archc,
                    I686Label::External {
                        name: GUEST_ENTRY.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
            if data_init_func_idx == Some(*id) {
                out.set_label(
                    &mut ctx,
                    archc,
                    I686Label::External {
                        name: crate::frontend::DATA_INIT_EXPORT_NAME.into(),
                    },
                )
                .map_err(|e| format!("set_label: {e:?}"))?;
            }
        }
        sysv::SysVWriterExt::sysv_handle_op::<_, HandleOpError<_>>(
            &mut out,
            &mut ctx,
            archc,
            &mut state,
            func_imports,
            &op,
            &mut reencoder,
            0,
        )
        .map_err(|e| format!("sysv_handle_op: {e:?}"))?;
    }

    let (text, labels) = out.into_parts();
    finish_object(arch, os, text, labels, Vec::new(), n_imports, func_imports)
}

/// Minimal WASM module used by the ILP32 emission smoke test:
/// `(module (func (export "_start") (nop)))`.
fn tiny_empty_wasm() -> Vec<u8> {
    use wasm_encoder::{
        CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
        TypeSection,
    };
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([], []);
    module.section(&types);
    let mut functions = FunctionSection::new();
    functions.function(0);
    module.section(&functions);
    let mut exports = ExportSection::new();
    exports.export("_start", ExportKind::Func, 0);
    module.section(&exports);
    let mut codes = CodeSection::new();
    // Empty body (just `end`) — thin ILP32 naive backends do not implement `nop`.
    let mut f = Function::new([]);
    f.instruction(&Instruction::End);
    codes.function(&f);
    module.section(&codes);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilp32_compile_wasm_to_object_emits_elf() {
        let wasm = tiny_empty_wasm();
        let mut ok = 0u32;
        for arch in [BinArch::RiscV32, BinArch::Arm, BinArch::X86] {
            let bytes = match compile_wasm_to_object(&wasm, arch, BinOs::Linux) {
                Ok(b) => b,
                Err(e) => {
                    // Soft-skip when a thin backend/dep is not yet wired enough.
                    eprintln!("soft-skip {arch:?}: {e}");
                    continue;
                }
            };
            assert!(
                bytes.len() > 4 && &bytes[0..4] == b"\x7fELF",
                "{arch:?}: expected ELF magic, got {} bytes",
                bytes.len()
            );
            ok += 1;
        }
        assert_eq!(ok, 3, "expected ELF emission for RiscV32, Arm, and X86");
    }
}
