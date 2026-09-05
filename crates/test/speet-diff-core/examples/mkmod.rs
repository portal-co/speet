fn main() {
    let path = std::env::args().nth(1).unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let case = speet_diff_core::case_from_json(&v);
    match speet_diff_core::build_case_module(&case) {
        Ok(wasm) => std::fs::write("/tmp/m.wasm", wasm).unwrap(),
        Err(e) => println!("err: {e:?}"),
    }
}
