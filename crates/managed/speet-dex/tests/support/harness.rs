//! Drives `DexRecompiler` through `yecta`'s `Reactor`/`ReactorAdapter` (the
//! same machinery `speet-e2e` uses for native recompilers) and assembles a
//! minimal, self-contained, runnable WASM module — no host imports; the
//! `LinearMemoryObjects` allocator functions are written directly as local
//! WASM functions with a trivial bump allocator.

#![allow(dead_code)]

use std::convert::Infallible;

use speet_dex::DexRecompiler;
use speet_link_core::{BaseContext, ReactorAdapter, ReactorContext};
use speet_object::ObjectModel;
use wasm_encoder::{
    CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection, GlobalSection,
    GlobalType, ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection,
    ValType,
};
use yecta::{LocalPool, Pool, Reactor, TableIdx, TypeIdx};

pub type Ctx = ();
pub type Err = Infallible;

/// Translate every method in `dex_bytes` through `DexRecompiler`, returning
/// the compiled functions (one per instruction slot, in program order) and
/// the register-file width (== the exported entry function's param count).
pub fn translate<M>(dex_bytes: &[u8], obj_model: M, base_func_offset: u32) -> (Vec<Function>, u32)
where
    M: ObjectModel<Ctx, Err>,
{
    let mut recompiler =
        DexRecompiler::<Ctx, Err, Function, LocalPool, M>::with_model(dex_bytes, obj_model)
            .expect("parse minimal test dex file");

    let mut reactor: Reactor<Ctx, Err, Function, LocalPool> = Reactor::default();
    static T: TableIdx = TableIdx(0);
    let mut rctx = ReactorAdapter::new(&mut reactor, Pool { handler: &T, ty: TypeIdx(0) });
    rctx.set_base_func_offset(base_func_offset);
    let mut ctx = ();

    let total_params = recompiler.setup_traps(&mut rctx, &mut ctx);

    recompiler
        .translate_all(&mut ctx, &mut rctx, &mut |locals| {
            Function::new(locals.collect::<Vec<_>>())
        })
        .expect("translate_all failed");

    rctx.seal_remaining(&mut ctx).expect("seal_remaining failed");
    (rctx.drain_fns(), total_params)
}

/// Function indices of the three hand-written bump-allocator helpers, in the
/// order they're declared in the assembled module (always 0, 1, 2).
pub const ALLOC_OBJECT_FN: u32 = 0;
pub const ALLOC_ARRAY_FN: u32 = 1;
pub const THROW_CLASS_CAST_FN: u32 = 2;

/// Number of local (non-DEX) helper functions placed before the translated
/// DEX functions — i.e. the `base_func_offset` to pass to [`translate`].
pub const N_HELPERS: u32 = 3;

/// Byte address the bump allocator's heap starts at. Chosen well past any
/// fixed scratch region a test might reserve at the bottom of memory.
pub const HEAP_BASE: i32 = 0x1000;

/// Assemble a runnable module: the three allocator helpers, then the
/// translated DEX functions (function `N_HELPERS` is exported as `_start`),
/// one mutable-i32 global (`$heap_ptr`, initialized to [`HEAP_BASE`]), and a
/// single exported linear memory.
pub fn assemble(dex_fns: Vec<Function>, register_count: u32) -> Vec<u8> {
    let mut types = TypeSection::new();
    // type 0: DEX entry — (i32 x register_count) -> ()
    let regs: Vec<ValType> = (0..register_count).map(|_| ValType::I32).collect();
    types.ty().function(regs, []);
    // type 1: alloc_object_fn — (i64,i64,i64,i64,i32) -> i32
    types.ty().function(
        [ValType::I64, ValType::I64, ValType::I64, ValType::I64, ValType::I32],
        [ValType::I32],
    );
    // type 2: alloc_array_fn — (i32,i64,i64,i64,i64,i32,i32) -> i32
    types.ty().function(
        [
            ValType::I32, ValType::I64, ValType::I64, ValType::I64, ValType::I64, ValType::I32,
            ValType::I32,
        ],
        [ValType::I32],
    );
    // type 3: throw_class_cast_fn — () -> ()
    types.ty().function([], []);

    let imports = ImportSection::new(); // none

    let mut funcs = FunctionSection::new();
    funcs.function(1); // alloc_object_fn
    funcs.function(2); // alloc_array_fn
    funcs.function(3); // throw_class_cast_fn
    for _ in &dex_fns {
        funcs.function(0);
    }

    let mut globals = GlobalSection::new();
    globals.global(
        GlobalType { val_type: ValType::I32, mutable: true, shared: false },
        &ConstExpr::i32_const(HEAP_BASE),
    );
    const HEAP_PTR_GLOBAL: u32 = 0;

    let mut mems = MemorySection::new();
    mems.memory(MemoryType { minimum: 4, maximum: None, memory64: false, shared: false, page_size_log2: None });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("_start", ExportKind::Func, N_HELPERS);

    let mut code = CodeSection::new();
    code.function(&alloc_object_fn_body(HEAP_PTR_GLOBAL));
    code.function(&alloc_array_fn_body(HEAP_PTR_GLOBAL));
    code.function(&throw_class_cast_fn_body());
    for f in &dex_fns {
        code.function(f);
    }

    let mut module = Module::new();
    module.section(&types);
    module.section(&imports);
    module.section(&funcs);
    module.section(&mems);
    module.section(&globals);
    module.section(&exports);
    module.section(&code);
    module.finish()
}

/// `alloc_object_fn(h0, h1, h2, h3, data_bytes) -> i32`
///
/// Bump-allocates `36 + data_bytes` from `$heap_ptr`, writes the 32-byte
/// type hash and a zero `array_dim` into the 36-byte header, and returns the
/// (pre-bump) base address — matching `LinearMemoryObjects`'s documented
/// header layout (`crates/managed/speet-object/src/linear.rs`).
fn alloc_object_fn_body(heap_ptr_global: u32) -> Function {
    // Params: h0:i64=0, h1:i64=1, h2:i64=2, h3:i64=3, data_bytes:i32=4.
    // Locals: ptr:i32 (local index 5).
    let mut f = Function::new([(1, ValType::I32)]);
    const PTR: u32 = 5;
    f.instruction(&Instruction::GlobalGet(heap_ptr_global));
    f.instruction(&Instruction::LocalSet(PTR));

    for (i, param) in [0u32, 1, 2, 3].into_iter().enumerate() {
        f.instruction(&Instruction::LocalGet(PTR));
        f.instruction(&Instruction::LocalGet(param));
        f.instruction(&Instruction::I64Store(MemArg {
            offset: (i as u64) * 8,
            align: 0,
            memory_index: 0,
        }));
    }
    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::I32Store(MemArg { offset: 32, align: 0, memory_index: 0 }));

    // $heap_ptr = ptr + 36 + data_bytes
    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::I32Const(36));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalGet(4));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::GlobalSet(heap_ptr_global));

    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::End);
    f
}

/// `alloc_array_fn(length, h0, h1, h2, h3, dim, elem_bytes) -> i32`
///
/// Header: 32-byte hash, `array_dim` (u32) @32, `length` (u32) @36, elements
/// starting @40 (`ARRAY_DATA_OFFSET`).
fn alloc_array_fn_body(heap_ptr_global: u32) -> Function {
    // Params: length=0, h0=1,h1=2,h2=3,h3=4 (i64), dim=5, elem_bytes=6.
    let mut f = Function::new([(1, ValType::I32)]);
    const PTR: u32 = 7;
    f.instruction(&Instruction::GlobalGet(heap_ptr_global));
    f.instruction(&Instruction::LocalSet(PTR));

    for (i, param) in [1u32, 2, 3, 4].into_iter().enumerate() {
        f.instruction(&Instruction::LocalGet(PTR));
        f.instruction(&Instruction::LocalGet(param));
        f.instruction(&Instruction::I64Store(MemArg {
            offset: (i as u64) * 8,
            align: 0,
            memory_index: 0,
        }));
    }
    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::LocalGet(5)); // dim
    f.instruction(&Instruction::I32Store(MemArg { offset: 32, align: 0, memory_index: 0 }));
    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::LocalGet(0)); // length
    f.instruction(&Instruction::I32Store(MemArg { offset: 36, align: 0, memory_index: 0 }));

    // $heap_ptr = ptr + 40 + length * elem_bytes
    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::I32Const(40));
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::LocalGet(0)); // length
    f.instruction(&Instruction::LocalGet(6)); // elem_bytes
    f.instruction(&Instruction::I32Mul);
    f.instruction(&Instruction::I32Add);
    f.instruction(&Instruction::GlobalSet(heap_ptr_global));

    f.instruction(&Instruction::LocalGet(PTR));
    f.instruction(&Instruction::End);
    f
}

fn throw_class_cast_fn_body() -> Function {
    let mut f = Function::new([]);
    f.instruction(&Instruction::Unreachable);
    f.instruction(&Instruction::End);
    f
}

// ── Execution ──────────────────────────────────────────────────────────────

pub struct RunResult {
    pub memory: Vec<u8>,
    pub trapped: bool,
}

/// Instantiate `wasm` in `wasmi`, call `_start()` with all-zero register
/// params, and snapshot the resulting linear memory.
pub fn run(wasm: &[u8]) -> RunResult {
    use wasmi::{Engine, Linker, Module as WasmiModule, Store};

    let engine = Engine::default();
    let module = WasmiModule::new(&engine, wasm).expect("module should validate");
    let linker: Linker<()> = Linker::new(&engine);
    let mut store = Store::new(&engine, ());
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate_and_start");

    let func = instance.get_func(&store, "_start").expect("_start export");
    let ty = func.ty(&store);
    let params: Vec<wasmi::Val> = ty.params().iter().map(|_| wasmi::Val::I32(0)).collect();
    let mut results = vec![];
    let trapped = func.call(&mut store, &params, &mut results).is_err();

    let mem = instance.get_memory(&store, "memory").expect("memory export");
    let memory = mem.data(&store).to_vec();
    RunResult { memory, trapped }
}

/// Read a little-endian i32 out of a memory snapshot.
pub fn read_i32(memory: &[u8], addr: i32) -> i32 {
    let addr = addr as usize;
    i32::from_le_bytes(memory[addr..addr + 4].try_into().unwrap())
}
