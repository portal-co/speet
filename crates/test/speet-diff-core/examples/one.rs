fn main() {
    let s: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(s);
    let o = speet_diff_core::run_oracle(&case);
    let r = speet_diff_core::run_recompiled(&case);
    let c = speet_diff_core::compare_outcomes(&case, o.as_ref().ok(), r.as_ref());
    println!("seed {s}: {c:?}");
}
