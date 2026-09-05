use wasmparser::{Operator, Payload};
fn main() {
    let data = std::fs::read("/tmp/rv.wasm").unwrap();
    let parser = wasmparser::Parser::new(0);
    let mut slot = 0usize;
    for payload in parser.parse_all(&data) {
        if let Payload::CodeSectionEntry(body) = payload.unwrap() {
            if slot == 2 {
                let ops = body.get_operators_reader().unwrap();
                for op in ops {
                    let op = op.unwrap();
                    match op {
                        Operator::End => continue,
                        Operator::I64Const { value } => println!("  i64c {value:#x}"),
                        Operator::I32Const { value } => println!("  i32c {value:#x}"),
                        Operator::Unreachable => println!("  UNREACHABLE"),
                        other => println!("  {other:?}"),
                    }
                }
            }
            slot += 1;
        }
    }
}
