fn main() {
    let a = speet_diff_core::case::Arch::RiscV64;
    for s in 1u64..400 {
        let case = speet_diff_core::generator::generate_case(a, s);
        if let Err(e) = speet_diff_core::run_oracle(&case) {
            println!("seed {s}: {e}");
            if s > 30 { break; }
        }
    }
}
