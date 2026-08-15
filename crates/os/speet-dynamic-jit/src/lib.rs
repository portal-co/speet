//! Phase 2a of the dynamic-JIT plan: in-process linking of a vane-compiled
//! WASM function into speet-interp's `emit_jit_lookup_stub` dynamic dispatch
//! table.
//!
//! **Scope note**: this targets a reference `wasmi` engine directly, not a
//! production "vkernel" host process — no such container-megabinary runtime
//! exists in this codebase yet (confirmed: no `vkernel` crate anywhere in
//! speet). This crate proves the actual integration risk the wider plan
//! hinges on (can a runtime-compiled WASM function be spliced into a live
//! dispatch chain and reached via `return_call_indirect`?) against the same
//! kind of engine `speet-corpus-harness` already uses for testing, while
//! being explicit that wiring this into a real container runtime is
//! follow-up work gated on that runtime existing.
//!
//! Two pieces:
//!
//! 1. [`compile_pc`] — drives vane's real RV64 frontend
//!    (`vane_riscv::TemplateJit`/`RiscvWasmJit`) plus its WASM renderer
//!    (`vane_arch::template::render_stack_to_wasm`, completed in Phase 0) to
//!    produce a standalone, single-function WASM module for one guest PC.
//! 2. [`link_jit_function`] — instantiates that module against a `wasmi`
//!    `Linker` already carrying the running guest's `ecall`/`jit_invalidate`/
//!    lookup-stub imports, then splices its exported function into a live
//!    dynamic dispatch table + side-table memory (matching
//!    `speet_link_core::JitConfig`'s layout).

pub mod cache;

use speet_link_core::JitConfig;
use vane_arch::{JitOpcode, WasmAbiConfig, WasmJitCtx};
use vane_riscv::template::{Flags, Labels, Params, TemplateJit};
use vane_riscv::{Heat, Mem, RiscvWasmJit};
use wasm_encoder::reencode::{RoundtripReencoder, utils::instruction};
use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, Module, TypeSection, ValType,
};

/// Absolute WASM function indices a [`compile_pc`]-produced module imports,
/// in declaration order (imports always occupy the lowest function indices).
/// The exported JIT function itself is `func index 3`, immediately after.
pub const ECALL_FUNC_IDX: u32 = 0;
pub const JIT_INVALIDATE_FUNC_IDX: u32 = 1;
pub const LOOKUP_STUB_FUNC_IDX: u32 = 2;
pub const JIT_FUNC_IDX: u32 = 3;

/// Name of the function exported by a [`compile_pc`]-produced module.
pub const JIT_EXPORT_NAME: &str = "jit_fn";

struct StaticAbiCtx(WasmAbiConfig);
impl WasmJitCtx for StaticAbiCtx {
    fn wasm_abi(&self) -> WasmAbiConfig {
        self.0
    }
}

/// `regfn` type: `num_regs` arch-reg params + `target_pc: i64` -> `num_regs`
/// arch-reg results — speet's `OobConfig` shared calling convention.
fn regfn_params(num_regs: u32) -> Vec<ValType> {
    vec![ValType::I64; (num_regs + 1) as usize]
}
fn regfn_results(num_regs: u32) -> Vec<ValType> {
    vec![ValType::I64; num_regs as usize]
}

/// Compile the guest RV64 code at `pc` (read from `mem`) into a standalone
/// WASM module with one exported function ([`JIT_EXPORT_NAME`]) of the
/// shared `regfn` type, matching `num_regs` architectural registers.
///
/// The module imports three functions from `"env"` at indices
/// [`ECALL_FUNC_IDX`]/[`JIT_INVALIDATE_FUNC_IDX`]/[`LOOKUP_STUB_FUNC_IDX`] —
/// [`link_jit_function`]'s caller is responsible for having registered
/// matching definitions on the `Linker` it instantiates this module with.
///
/// Vane may compile more than the single instruction at `pc`: RISC-V has no
/// fixed block boundary, so straight-line fallthrough and taken/not-taken
/// branches are followed until a genuine control transfer (a jump/call,
/// lowered to `StackOp::TailCall`) is reached.
pub fn compile_pc(mem: &Mem, pc: u64, num_regs: u32) -> Vec<u8> {
    let abi = WasmAbiConfig {
        reg_base: 0,
        num_regs,
        target_pc_local: num_regs,
        scratch_a: num_regs + 1,
        scratch_b: num_regs + 2,
        mem_idx: 0,
        ecall_func_idx: ECALL_FUNC_IDX,
        jit_invalidate_func_idx: JIT_INVALIDATE_FUNC_IDX,
        lookup_stub_func_idx: LOOKUP_STUB_FUNC_IDX,
    };
    let ctx = StaticAbiCtx(abi);

    let flate = vane_arch::DebugFlate {};
    let params = Params {
        react: mem,
        trial: &|_pc| Heat::New, // always compile fresh — no per-trace cache for a one-shot request
        flate: &flate, // unused on the WASM path; required by Params' shape regardless
        root: pc,
        flags: Flags::default(),
    };
    let jit = TemplateJit { pc, labels: Labels::default(), depth: 0, params };

    let ops: Vec<JitOpcode<'_>> = jit.Riscv(&ctx).collect();

    let mut types = TypeSection::new();
    types.ty().function(regfn_params(num_regs), regfn_results(num_regs)); // type 0: regfn
    types.ty().function(vec![ValType::I64; num_regs as usize], vec![ValType::I64; num_regs as usize]); // type 1: ecall (num_regs -> num_regs)
    types.ty().function([ValType::I64], []); // type 2: jit_invalidate

    let mut imports = ImportSection::new();
    imports.import("env", "ecall", EntityType::Function(1));
    imports.import("env", "jit_invalidate", EntityType::Function(2));
    imports.import("env", "lookup_stub", EntityType::Function(0));
    // Shares the running guest's actual memory (not a private one) so
    // LoadMem/StoreMem/CheckCode observe the same address space AOT-compiled
    // and interpreted code does — the caller must register its export under
    // this same "env"."guest_memory" name before instantiating this module.
    imports.import(
        "env",
        "guest_memory",
        EntityType::Memory(wasm_encoder::MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );

    let mut functions = FunctionSection::new();
    functions.function(0); // the JIT'd function itself, type 0 (regfn)

    let mut exports = ExportSection::new();
    exports.export(JIT_EXPORT_NAME, ExportKind::Func, JIT_FUNC_IDX);

    let mut f = Function::new([(1, ValType::I64), (1, ValType::I64)]); // scratch_a, scratch_b
    let mut reencoder = RoundtripReencoder;
    for JitOpcode::Operator { op } in ops {
        let instr = instruction(&mut reencoder, op).expect("reencode should not fail");
        f.instruction(&instr);
    }
    // `rv_emit` wraps every trace in an outer `StackOp::LoopBegin`/`LoopEnd`
    // for its label/branch-target bookkeeping (see vane-riscv's `rv_emit`
    // doc comment), and every finite trace terminates via a `TailCall`
    // nested inside that loop, not by falling off the end. WASM validation
    // does *not* propagate "unreachable" across a block/loop boundary —
    // exiting a loop's `end` always resets reachability to whatever it was
    // on entry, regardless of whether every path inside diverged — so the
    // function's own implicit final return is left unsatisfied even though
    // it is genuinely never reached at runtime. A trailing `unreachable`,
    // unnested at the function's true top level, is the standard WASM idiom
    // for exactly this "provably dead, but the validator can't see it"
    // situation, and satisfies validation regardless of whether this
    // particular trace happened to need it.
    f.instruction(&Instruction::Unreachable);
    f.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&f);

    let mut module = Module::new();
    module.section(&types).section(&imports).section(&functions).section(&exports).section(&code);
    module.finish()
}

/// Errors from [`link_jit_function`].
#[derive(Debug)]
pub enum LinkError {
    Instantiate(wasmi::Error),
    MissingExport,
    TableFull,
}

impl core::fmt::Display for LinkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinkError::Instantiate(e) => write!(f, "failed to instantiate JIT module: {e}"),
            LinkError::MissingExport => write!(f, "JIT module has no `{JIT_EXPORT_NAME}` export"),
            LinkError::TableFull => write!(f, "dynamic dispatch table is full"),
        }
    }
}

/// Instantiate a [`compile_pc`]-produced module (resolving its `"env"`
/// imports against `linker`, which must already carry matching
/// `ecall`/`jit_invalidate`/`lookup_stub` definitions), then splice its
/// exported function into `dyn_table` at the first empty slot and record
/// `(pc, slot)` in `dyn_table_mem` — matching the layout
/// `speet_interp::emit_jit_lookup_stub` expects (see [`JitConfig`]'s doc
/// comment: 12-byte entries, `table_slot == -1` marks empty).
pub fn link_jit_function<T>(
    store: &mut wasmi::Store<T>,
    linker: &wasmi::Linker<T>,
    engine: &wasmi::Engine,
    jit: &JitConfig,
    dyn_table: &wasmi::Table,
    dyn_table_mem: &wasmi::Memory,
    pc: u64,
    wasm_module_bytes: &[u8],
) -> Result<u32, LinkError> {
    let module = wasmi::Module::new(engine, wasm_module_bytes).map_err(LinkError::Instantiate)?;
    let instance =
        linker.instantiate_and_start(&mut *store, &module).map_err(LinkError::Instantiate)?;
    let func = instance.get_func(&mut *store, JIT_EXPORT_NAME).ok_or(LinkError::MissingExport)?;

    let slot = find_empty_dyn_slot(store, jit, dyn_table_mem)?;
    dyn_table
        .set(&mut *store, slot as u64, wasmi::Val::from(func))
        .map_err(|e| LinkError::Instantiate(e.into()))?;

    let entry_off = jit.dyn_table_mem_offset as usize + (slot as usize) * 12;
    let mut entry = [0u8; 12];
    entry[0..8].copy_from_slice(&pc.to_le_bytes());
    entry[8..12].copy_from_slice(&slot.to_le_bytes());
    dyn_table_mem
        .write(&mut *store, entry_off, &entry)
        .map_err(|e| LinkError::Instantiate(e.into()))?;

    Ok(slot)
}

fn find_empty_dyn_slot<T>(
    store: &wasmi::Store<T>,
    jit: &JitConfig,
    dyn_table_mem: &wasmi::Memory,
) -> Result<u32, LinkError> {
    let data = dyn_table_mem.data(store);
    for i in 0..jit.dyn_table_capacity {
        let off = jit.dyn_table_mem_offset as usize + (i as usize) * 12 + 8;
        let table_slot = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        if table_slot == -1 {
            return Ok(i);
        }
    }
    Err(LinkError::TableFull)
}

/// Evict `pc`'s entry from the dynamic dispatch side table, if present.
///
/// This is the host-side eviction logic `abi.jit_invalidate_func_idx`
/// (see [`vane_arch::WasmAbiConfig`]) should be wired to: a JIT'd
/// function's `CheckCode` lowering calls it on a self-modifying-code
/// mismatch, before tail-calling back into the lookup stub for the same
/// `pc` — without eviction, the stub's dynamic-table scan would just find
/// the same now-stale entry again and loop. Only clears the side-table
/// row (`table_slot = -1`, `speet_interp::emit_jit_lookup_stub`'s "empty"
/// sentinel); does not need to touch the WASM `Table` itself, since a
/// cleared row is simply never looked up again.
///
/// Returns `true` if an entry for `pc` was found and evicted. Exposed as a
/// plain function (rather than a pre-built `wasmi` closure) so callers can
/// wire it into `Linker::func_wrap` however fits their own `Store` data
/// type — see `evict_dyn_entry` in this crate's `smc_invalidation` test for
/// the `Caller`-based wiring shape.
pub fn invalidate_dyn_entry(
    ctx: impl wasmi::AsContextMut,
    jit: &JitConfig,
    dyn_table_mem: &wasmi::Memory,
    pc: u64,
) -> bool {
    let mut ctx = ctx;
    let cap = jit.dyn_table_capacity;
    let off_base = jit.dyn_table_mem_offset as usize;
    let mut found_off = None;
    {
        let data = dyn_table_mem.data(&ctx);
        for i in 0..cap {
            let off = off_base + (i as usize) * 12;
            let entry_pc = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let table_slot = i32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
            if table_slot != -1 && entry_pc == pc {
                found_off = Some(off);
                break;
            }
        }
    }
    match found_off {
        Some(off) => {
            let _ = dyn_table_mem.write(&mut ctx, off + 8, &(-1i32).to_le_bytes());
            true
        }
        None => false,
    }
}
