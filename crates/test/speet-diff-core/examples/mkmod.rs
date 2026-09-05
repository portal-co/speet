/// Rebuild the case-module WASM from an artifact JSON and write /tmp/m.wasm.
/// Falls back to the unvalidated module so mpdump can inspect it.
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let case = speet_diff_core::case_from_json(&v);
    eprintln!("arch = {:?}", case.arch);
    match speet_diff_core::build_case_module(&case) {
        Ok(wasm) => std::fs::write("/tmp/m.wasm", wasm).unwrap(),
        Err(e) => {
            println!("err: {e:?}");
            let (fns, params, _) = speet_diff_core::recompiled::translate_case(&case);
            let wasm = speet_corpus_harness::assemble::assemble_corpus_module(&fns, &params, 0);
            std::fs::write("/tmp/m.wasm", wasm).unwrap();
        }
    }
}
