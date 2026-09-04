use wasmparser::{Operator, Payload};
fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(seed);
    let wasm = speet_diff_core::recompiled::build_case_module(&case).expect("build");
    let parser = wasmparser::Parser::new(0);
    let mut fn_idx = 0usize;
    let show_from: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    for payload in parser.parse_all(&wasm) {
        if let Payload::CodeSectionEntry(body) = payload.unwrap() {
            if fn_idx >= show_from {
                println!("== function {fn_idx} ==");
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
                        Operator::BrIf { relative_depth } => println!("  br_if {relative_depth}"),
                        Operator::Br { relative_depth } => println!("  br {relative_depth}"),
                        Operator::If { blockty: _ } => println!("  if"),
                        Operator::Else => println!("  else"),
                        Operator::Call { function_index } => println!("  call {function_index}"),
                        other => println!("  {other:?}"),
                    }
                }
            }
            fn_idx += 1;
        }
    }
}
