use speet_link_core::{ReactorAdapter, ReactorContext};
use speet_link_core::BaseContext;
use speet_x86_64::X86Recompiler;
use wasmparser::{Operator, Payload};
use yecta::{LocalPool, Reactor, TableIdx, TypeIdx};
use wasm_encoder::{Function, Module, ValType};
use std::convert::Infallible;

fn main() {
    let text = vec![0x44u8, 0x89, 0xFF, 0xC3]; // mov r15d,edi; ret
    let base = 0x10_0000u64;
    let mut reactor: Reactor<(), Infallible, Function, LocalPool> = Reactor::default();
    static T: TableIdx = TableIdx(0);
    let mut rctx = ReactorAdapter {
        reactor: &mut reactor,
        layout: yecta::LocalLayout::empty(),
        locals_mark: yecta::Mark { slot_count: 0, total_locals: 0 },
        injected_start: yecta::Mark { slot_count: 0, total_locals: 0 },
        layout_params: speet_link_core::RuntimeLayoutParams::new(),
        pool: yecta::Pool { handler: &T, ty: TypeIdx(0) },
        escape: yecta::CallEscape::Jump,
    };
    rctx.set_base_func_offset(0);
    let mut ctx = ();
    let mut rc = X86Recompiler::<(), Infallible>::new_with_base_rip(base);
    rc.setup_traps(&mut rctx, &mut ctx);
    rc.translate_bytes(&mut ctx, &mut rctx, &text, base,
        &mut |a| Function::new(a.collect::<Vec<_>>())).unwrap();
    let fns = rctx.drain_fns();
    println!("n functions: {}", fns.len());
    // Wrap in a module so wasmparser can decode bodies.
    let mut types = wasm_encoder::TypeSection::new();
    let mut params: Vec<ValType> = (0..42).map(|_| ValType::I64).collect();
    params.truncate(16);
    let _ = params;
    let mut ty_all = wasm_encoder::TypeSection::new();
    ty_all.ty().function(vec![ValType::I64; 43], vec![ValType::I64; 43]);
    let mut funcs = wasm_encoder::FunctionSection::new();
    for _ in &fns { funcs.function(0); }
    let mut code = wasm_encoder::CodeSection::new();
    for f in &fns { code.function(f); }
    let mut m = Module::new();
    m.section(&ty_all);
    m.section(&funcs);
    m.section(&code);
    let bytes = m.finish();
    let mut fi = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let Payload::CodeSectionEntry(body) = payload.unwrap() {
            println!("== fn {fi} ==");
            let ops = body.get_operators_reader().unwrap();
            for op in ops {
                let op = op.unwrap();
                match op {
                    Operator::End => continue,
                    Operator::LocalGet { local_index } => println!("  get {local_index}"),
                    Operator::LocalSet { local_index } => println!("  set {local_index}"),
                    Operator::LocalTee { local_index } => println!("  tee {local_index}"),
                    Operator::I64Const { value } => println!("  i64c {value:#x}"),
                    Operator::I32Const { value } => println!("  i32c {value:#x}"),
                    other => println!("  {other:?}"),
                }
            }
            fi += 1;
        }
    }
}
