use wasmparser::{Operator, Payload};
fn main() {
    let case = speet_diff_core::tests_common::make_case(
        vec![0x48,0xB8,0x03,0,0,0,0,0,0,0, 0x48,0x83,0xC0,0x03, 0xC3], 1);
    let wasm = speet_diff_core::recompiled::build_case_module(&case).unwrap();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm) {
        if let Payload::CodeSectionEntry(body) = payload.unwrap() {
            let ops = body.get_operators_reader().unwrap();
            for op in ops {
                let op = op.unwrap();
                match op {
                    Operator::End => continue,
                    Operator::LocalSet { local_index } if (17..=21).contains(&local_index) =>
                        println!("  SET FLAG {local_index}"),
                    Operator::I32Const { value } => println!("  i32c {value}"),
                    Operator::I64Const { value } => println!("  i64c {value:#x}"),
                    Operator::Call { function_index } => println!("  call {function_index}"),
                    Operator::Unreachable => println!("  UNREACHABLE"),
                    _ => {}
                }
            }
            break;
        }
    }
}
