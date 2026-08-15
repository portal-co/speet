//! `DispatchMode::FunctionRef`-mode linking, mirroring [`crate::link_jit_function`]
//! but targeting `wasmtime` instead of `wasmi` — `wasmi` has zero function-
//! references support (no `ref.func`/`call_ref`/`return_call_ref`), so
//! [`speet_interp::emit_jit_lookup_stub`]'s `FunctionRef`-mode dispatch tail
//! (`table.get` + `return_call_ref`) can only execute here.
//!
//! Splicing a compiled function into the typed-funcref sibling table
//! (`JitConfig::dyn_funcref_table_slot`) needs no `ref.func` WASM
//! instruction at all: `wasmtime::Table::set` accepts a host-side `Func`
//! handle directly (`Ref::Func(Some(func))`), so the population is a pure
//! host-embedder-API operation, exactly like [`crate::link_jit_function`]'s
//! existing `wasmi::Table::set` call — the two linking paths differ only in
//! which engine executes the result, not in how linking itself works.

use speet_link_core::JitConfig;
use wasmtime::{AsContextMut, Engine, Instance, Linker, Memory, Ref, Table};

use crate::LinkError;

/// Instantiate a [`crate::compile_pc`]-produced module against `linker`
/// (which must already carry matching `ecall`/`jit_invalidate`/`lookup_stub`
/// definitions), then splice its exported function into `dyn_funcref_table`
/// at the first empty slot (tracked the same way as the `TableIndirect`
/// path: via `dyn_table_mem`'s 12-byte `(pc, table_slot)` side-table
/// entries — both dispatch modes share one side-table format and one
/// occupancy-tracking convention).
pub fn link_jit_function_funcref<T: 'static>(
    mut store: impl AsContextMut<Data = T>,
    linker: &Linker<T>,
    engine: &Engine,
    jit: &JitConfig,
    dyn_funcref_table: &Table,
    dyn_table_mem: &Memory,
    pc: u64,
    wasm_module_bytes: &[u8],
) -> Result<u32, LinkError> {
    let module =
        wasmtime::Module::new(engine, wasm_module_bytes).map_err(LinkError::InstantiateWasmtime)?;
    let instance: Instance =
        linker.instantiate(&mut store, &module).map_err(LinkError::InstantiateWasmtime)?;
    let func = instance.get_func(&mut store, crate::JIT_EXPORT_NAME).ok_or(LinkError::MissingExport)?;

    let slot = find_empty_dyn_slot_wasmtime(&mut store, jit, dyn_table_mem)?;
    dyn_funcref_table
        .set(&mut store, slot as u64, Ref::Func(Some(func)))
        .map_err(LinkError::InstantiateWasmtime)?;

    let entry_off = jit.dyn_table_mem_offset as usize + (slot as usize) * 12;
    let mut entry = [0u8; 12];
    entry[0..8].copy_from_slice(&pc.to_le_bytes());
    entry[8..12].copy_from_slice(&slot.to_le_bytes());
    dyn_table_mem
        .write(&mut store, entry_off, &entry)
        .map_err(|e| LinkError::InstantiateWasmtime(e.into()))?;

    Ok(slot)
}

fn find_empty_dyn_slot_wasmtime<T: 'static>(
    mut store: impl AsContextMut<Data = T>,
    jit: &JitConfig,
    dyn_table_mem: &Memory,
) -> Result<u32, LinkError> {
    let data = dyn_table_mem.data(store.as_context_mut());
    for i in 0..jit.dyn_table_capacity {
        let off = jit.dyn_table_mem_offset as usize + (i as usize) * 12 + 8;
        let table_slot = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        if table_slot == -1 {
            return Ok(i);
        }
    }
    Err(LinkError::TableFull)
}

/// `wasmtime`-flavored sibling of [`crate::invalidate_dyn_entry`] — same
/// side-table-only eviction semantics (see that function's doc comment),
/// operating on a `wasmtime::Memory` instead of `wasmi::Memory`.
pub fn invalidate_dyn_entry_wasmtime<T: 'static>(
    mut store: impl AsContextMut<Data = T>,
    jit: &JitConfig,
    dyn_table_mem: &Memory,
    pc: u64,
) -> bool {
    let cap = jit.dyn_table_capacity;
    let off_base = jit.dyn_table_mem_offset as usize;
    let mut found_off = None;
    {
        let data = dyn_table_mem.data(store.as_context_mut());
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
            let _ = dyn_table_mem.write(&mut store, off + 8, &(-1i32).to_le_bytes());
            true
        }
        None => false,
    }
}
