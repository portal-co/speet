/// Dump fn0's operators with a simple type-stack simulation.
fn main() {
    let wasm = std::fs::read("/tmp/m.wasm").unwrap();
    for payload in wasmparser::Parser::new(0).parse_all(&wasm).flatten() {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload {
            let reader = body.get_operators_reader().unwrap();
            let mut stack: Vec<&'static str> = Vec::new();
            for op in reader {
                let o = op.unwrap();
                let desc = match &o {
                    wasmparser::Operator::I32Const { value } => { stack.push("i32"); format!("i32.const {}", value) }
                    wasmparser::Operator::I64Const { value } => { stack.push("i64"); format!("i64.const {}", value) }
                    wasmparser::Operator::LocalGet { local_index } => { stack.push("<local>"); format!("get {}", local_index) }
                    wasmparser::Operator::LocalSet { local_index } => { stack.pop(); format!("set {}", local_index) }
                    wasmparser::Operator::LocalTee { local_index } => format!("tee {}", local_index),
                    wasmparser::Operator::I32Rotl => { stack.pop(); stack.pop(); stack.push("i32"); "rotl".into() }
                    wasmparser::Operator::I64Rotl => { stack.pop(); stack.pop(); stack.push("i64"); "rotl64".into() }
                    wasmparser::Operator::I32Add => { stack.pop(); stack.pop(); stack.push("i32"); "i32.add".into() }
                    wasmparser::Operator::I64ExtendI32U => { stack.pop(); stack.push("i64"); "extend".into() }
                    wasmparser::Operator::I32And | wasmparser::Operator::I32Or
                    | wasmparser::Operator::I32Shl | wasmparser::Operator::I32ShrU
                    | wasmparser::Operator::I32ShrS => { stack.pop(); stack.pop(); stack.push("i32"); "i32.binop".into() }
                    wasmparser::Operator::I64And | wasmparser::Operator::I64Or
                    | wasmparser::Operator::I64Shl | wasmparser::Operator::I64ShrU => { stack.pop(); stack.pop(); stack.push("i64"); "i64.binop".into() }
                    wasmparser::Operator::I32Load { .. } | wasmparser::Operator::I64Load { .. } => { stack.pop(); stack.push("<load>"); "load".into() }
                    wasmparser::Operator::I32Store { .. } => { let v = stack.pop().unwrap_or("?"); let a = stack.pop().unwrap_or("?"); format!("i32.store value={} addr={}", v, a) }
                    wasmparser::Operator::I64Store { .. } => { let v = stack.pop().unwrap_or("?"); let a = stack.pop().unwrap_or("?"); format!("i64.store value={} addr={}", v, a) }
                    wasmparser::Operator::End => format!("END stack={:?}", stack),
                    other => format!("{:?}", other),
                };
                println!("{}", desc);
            }
            return;
        }
    }
}
