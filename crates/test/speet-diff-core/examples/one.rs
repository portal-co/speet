fn main() {
    let s: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(s);
    let o = speet_diff_core::run_oracle(&case);
    let r = speet_diff_core::run_recompiled(&case);
    let c = speet_diff_core::compare_outcomes(&case, o.as_ref().ok(), r.as_ref());
    println!("seed {s}: {c:?}");
    if matches!(c, speet_diff_core::Comparison::Divergence(_)) {
        let m = speet_diff_core::minimize(&case);
        println!("minimized code: {}", m.code.iter().map(|b| format!("{b:02x}")).collect::<String>());
        println!("regs g7-g15: {:?}", &m.regs.gprs[7..]);
    }
}
