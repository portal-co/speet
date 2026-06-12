//! Frontend stage: load a host binary and drive speet to a WASM megabinary.
//!
//! Responsibilities (M1 wires the minimal path; M3 adds PLT tunneling):
//! 1. Load via [`binary_io::load_auto`]; assert OS+arch == host (v1 same-platform).
//! 2. Build an external-target table from `imports` (undefined dyn syms + PLT
//!    addrs) and undefined-symbol relocations, mapping `plt_addr -> guest_name`.
//!    During speet translation, a call/jmp whose target is in this table is
//!    lowered via `AmbientSink::call_ambient(name, sig)` instead of an internal
//!    tail-call.
//! 3. Feed non-text sections as passive WASM data segments + a `data_init_fn`.
//! 4. Register the page-table memory model (`speet_memory::VirtualMemory`).
//! 5. Resolve the entry point to a megabinary export (`__guest_entry`).

use binary_io::{BinArch, BinOs, ImportSym, LoadedBinary};

/// External call targets discovered in the loaded binary: guest address (PLT
/// stub or relocation site target) -> external symbol name to tunnel.
#[derive(Debug, Default, Clone)]
pub struct ExternalTargets {
    pub by_plt_addr: std::collections::BTreeMap<u64, String>,
}

impl ExternalTargets {
    /// Build the table from a loaded binary's import list (PLT addresses).
    pub fn from_imports(imports: &[ImportSym]) -> Self {
        let mut by_plt_addr = std::collections::BTreeMap::new();
        for imp in imports {
            if let Some(addr) = imp.plt_addr {
                by_plt_addr.insert(addr, imp.name.clone());
            }
        }
        Self { by_plt_addr }
    }

    /// Look up an external symbol name for a call/jmp target address.
    pub fn lookup(&self, target: u64) -> Option<&str> {
        self.by_plt_addr.get(&target).map(|s| s.as_str())
    }
}

/// Returns the current host (OS, arch) so the driver can enforce same-platform.
pub fn host_platform() -> (BinOs, BinArch) {
    let os = if cfg!(target_os = "macos") {
        BinOs::MacOs
    } else {
        BinOs::Linux
    };
    let arch = if cfg!(target_arch = "aarch64") {
        BinArch::AArch64
    } else {
        BinArch::X86_64
    };
    (os, arch)
}

/// Assert the loaded binary matches the host platform (v1 constraint).
pub fn assert_same_platform(bin: &LoadedBinary) -> Result<(), String> {
    let (os, arch) = host_platform();
    if !matches!((&bin.os, os), (BinOs::Linux, BinOs::Linux) | (BinOs::MacOs, BinOs::MacOs)) {
        return Err(format!("input OS {:?} != host {:?} (v1 is same-platform)", bin.os, os));
    }
    if !matches!((&bin.arch, arch), (BinArch::X86_64, BinArch::X86_64) | (BinArch::AArch64, BinArch::AArch64)) {
        return Err(format!("input arch {:?} != host {:?} (v1 is same-platform)", bin.arch, arch));
    }
    Ok(())
}

// ── speet translate: guest machine code -> WASM module ───────────────────────

use core::convert::Infallible;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use wasm_encoder::{
    CodeSection, ConstExpr, ElementSection, Elements, ExportKind, ExportSection, Function,
    FunctionSection, ImportSection, MemorySection, MemoryType, Module, RefType, TableSection,
    TableType, TypeSection, ValType,
};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};

/// Imports speet's output assumes: `env.__speet_hint`, `env.write`, `env.exit`.
const N_IMPORTS: u32 = 3;

static REACTOR_TABLE: TableIdx = TableIdx(0);

/// Build a fresh [`ReactorAdapter`] (mirrors the speet-e2e harness `make_rctx`,
/// with exception handling disabled).
fn make_rctx<'r>(
    reactor: &'r mut Reactor<(), Infallible, Function, LocalPool>,
    base_func_offset: u32,
) -> ReactorAdapter<'r, (), Infallible, Function, LocalPool> {
    let mut rctx = ReactorAdapter {
        reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        pool: yecta::Pool { handler: &REACTOR_TABLE, ty: TypeIdx(0) },
        escape_tag: None,
    };
    rctx.set_base_func_offset(base_func_offset);
    rctx
}

fn collect_params(rctx: &ReactorAdapter<'_, (), Infallible, Function, LocalPool>) -> Vec<ValType> {
    let mark = rctx.locals_mark();
    rctx.layout()
        .iter_before(&mark)
        .flat_map(|(count, ty)| core::iter::repeat(ty).take(count as usize))
        .collect()
}

/// Recompiled output of one guest binary: function bodies + their (shared) param
/// signature (the guest register file) + any unsupported instructions seen.
pub struct Translated {
    pub fns: Vec<Function>,
    pub params: Vec<ValType>,
    pub unsupported: Vec<String>,
}

/// Translate a `.text` blob of guest machine code to speet WASM functions.
pub fn translate(text: &[u8], start_addr: u64, arch: BinArch) -> Translated {
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, N_IMPORTS);
    let mut ctx = ();

    let (params, unsupported) = match arch {
        BinArch::X86_64 => {
            let mut rc = speet_x86_64::X86Recompiler::new_with_base_rip(start_addr);
            rc.setup_traps(&mut rctx, &mut ctx);
            let params = collect_params(&rctx);
            rc.translate_bytes(&mut ctx, &mut rctx, text, start_addr, &mut |a| {
                Function::new(a.collect::<Vec<_>>())
            })
            .expect("translate_bytes");
            (params, rc.unsupported_insns().iter().cloned().collect())
        }
        BinArch::AArch64 => {
            let mut rc = speet_aarch64::AArch64Recompiler::<(), Infallible>::new_with_base_pc(start_addr);
            rc.setup_traps(&mut rctx, &mut ctx);
            let params = collect_params(&rctx);
            rc.translate_bytes(&mut ctx, &mut rctx, text, start_addr, &mut |a| {
                Function::new(a.collect::<Vec<_>>())
            })
            .expect("translate_bytes");
            (params, rc.unsupported_insns().iter().cloned().collect())
        }
    };

    Translated { fns: rctx.drain_fns(), params, unsupported }
}

/// Assemble a single translated binary into a complete WASM module (no exception
/// handling). The entry (first function) is exported as `_start`. Mirrors the
/// speet-e2e harness `assemble_module` for one slice.
pub fn assemble_module(t: &Translated) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function(t.params.clone(), []); // type 0: register-file -> ()
    types.ty().function([ValType::I32], []); // type 1: hint/exit
    types.ty().function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]); // type 2: write

    let mut imports = ImportSection::new();
    imports.import("env", "__speet_hint", wasm_encoder::EntityType::Function(1));
    imports.import("env", "write", wasm_encoder::EntityType::Function(2));
    imports.import("env", "exit", wasm_encoder::EntityType::Function(1));

    let mut funcs = FunctionSection::new();
    for _ in &t.fns {
        funcs.function(0);
    }

    let total = t.fns.len() as u32;
    let table_size = N_IMPORTS + total;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 64,
        maximum: None,
        memory64: true,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, N_IMPORTS);

    let indices: Vec<u32> = (N_IMPORTS..N_IMPORTS + total).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i64_const(N_IMPORTS as i64),
        Elements::Functions(std::borrow::Cow::Borrowed(&indices)),
    );

    let mut code = CodeSection::new();
    for f in &t.fns {
        code.function(f);
    }

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    module.finish()
}

/// Recompile a `.text` blob of guest machine code to a complete WASM module.
pub fn recompile_to_wasm(text: &[u8], start_addr: u64, arch: BinArch) -> (Vec<u8>, Vec<String>) {
    let t = translate(text, start_addr, arch);
    let unsupported = t.unsupported.clone();
    (assemble_module(&t), unsupported)
}

// ── RISC-V guest with host syscall lowering ──────────────────────────────────

/// Number of host-syscall imports (`env.exit`, `env.write`).
pub const N_SYSCALL_IMPORTS: u32 = 2;

/// Import layout for the native syscall shim: `env.exit` = 0, `env.write` = 1.
pub fn native_syscall_imports() -> speet_host_syscall::NativeSyscallImports {
    speet_host_syscall::NativeSyscallImports { exit: 0, write: 1 }
}

/// Translate an RV64 Linux `.text` blob, lowering `ecall` to the native host
/// syscall imports (`env.exit`/`env.write`) via [`speet_syscall::WasmSyscallDispatcher`].
pub fn translate_rv64(text: &[u8], start_addr: u64) -> Translated {
    use rv_asm::Xlen;
    let imports = native_syscall_imports();
    let table = speet_host_syscall::linux_rv64_table(&imports);

    let mut recompiler =
        speet_riscv::RiscVRecompiler::<(), Infallible, Function>::new_with_full_config(
            start_addr, false, true, false,
        );
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    let mut rctx = make_rctx(&mut reactor, N_SYSCALL_IMPORTS);
    let mut ctx = ();
    recompiler.setup_traps(&mut rctx, &mut ctx);
    let params = collect_params(&rctx);

    let mut dispatcher = speet_syscall::WasmSyscallDispatcher {
        table: &table,
        syscall_num_local: speet_host_syscall::A7,
        syscall_num_is_i64: true,
        num_params: params.len() as u32,
        next_pc_func: N_SYSCALL_IMPORTS, // dummy: exit terminates, no continuation
    };
    recompiler.set_ecall_callback(&mut dispatcher);

    recompiler
        .translate_bytes(&mut ctx, &mut rctx, text, start_addr as u32, Xlen::Rv64, &mut |a| {
            Function::new(a.collect::<Vec<_>>())
        })
        .expect("translate_bytes");

    Translated { fns: rctx.drain_fns(), params, unsupported: vec![] }
}

/// Assemble an RV64 translation with the native syscall imports
/// (`env.exit`/`env.write`), entry exported as `_start`. 32-bit linear memory.
pub fn assemble_syscall_module(t: &Translated) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function(t.params.clone(), []); // type 0: register-file -> ()
    types.ty().function([ValType::I32], []); // type 1: exit(code)
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I32]); // type 2: write

    let mut imports = ImportSection::new();
    imports.import("env", "exit", wasm_encoder::EntityType::Function(1));
    imports.import("env", "write", wasm_encoder::EntityType::Function(2));

    let mut funcs = FunctionSection::new();
    for _ in &t.fns {
        funcs.function(0);
    }

    let total = t.fns.len() as u32;
    let table_size = N_SYSCALL_IMPORTS + total;
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        minimum: table_size as u64,
        maximum: Some(table_size as u64),
        table64: true,
        shared: false,
    });

    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: 64,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, N_SYSCALL_IMPORTS);

    let indices: Vec<u32> = (N_SYSCALL_IMPORTS..N_SYSCALL_IMPORTS + total).collect();
    let mut elems = ElementSection::new();
    elems.active(
        Some(0),
        &ConstExpr::i64_const(N_SYSCALL_IMPORTS as i64),
        Elements::Functions(std::borrow::Cow::Borrowed(&indices)),
    );

    let mut code = CodeSection::new();
    for f in &t.fns {
        code.function(f);
    }

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&tables);
    module.section(&mems);
    module.section(&exports);
    module.section(&elems);
    module.section(&code);
    module.finish()
}

/// Recompile an RV64 Linux `.text` blob (with `ecall` → host syscalls) to WASM.
pub fn recompile_rv64_to_wasm(text: &[u8], start_addr: u64) -> Vec<u8> {
    let t = translate_rv64(text, start_addr);
    assemble_syscall_module(&t)
}
