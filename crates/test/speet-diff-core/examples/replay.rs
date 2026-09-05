fn main() {
    let path = std::env::args().nth(1).unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let record: speet_diff_core::report::CaseRecord =
        serde_json::from_value(v["case"].clone()).unwrap();
    let case = record.to_case();
    let o = speet_diff_core::run_oracle(&case);
    let r = speet_diff_core::run_recompiled(&case);
    println!("oracle: {:?}", o.map(|x| x.exit));
    println!("recomp: {:?}", r.map(|x| x.exit).map_err(|e| format!("{e:?}")));
}
