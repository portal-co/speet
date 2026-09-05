//! Dump a translated MIPS slot's WASM ops (SPEET_DIFF_DUMP_ON_TRAP artifact).
use wasmparser::{Operator, Payload};
fn main() {
    let path = std::env::args().nth(1).unwrap_or("/tmp/m.wasm".into());
    let want: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let data = std::fs::read(&path).unwrap();
    let parser = wasmparser::Parser::new(0);
    let mut slot = 0usize;
    for payload in parser.parse_all(&data) {
        if let Payload::CodeSectionEntry(body) = payload.unwrap() {
            if slot == want {
                let ops = body.get_operators_reader().unwrap();
                for op in ops {
                    let op = op.unwrap();
                    match op {
                        Operator::End => continue,
                        other => println!("  {other:?}"),
                    }
                }
            }
            slot += 1;
        }
    }
}
