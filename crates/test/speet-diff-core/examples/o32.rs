fn main() {
    let a = speet_diff_core::case::Arch::X86_32;
    for s in 1u64..40 {
        let case = speet_diff_core::generate_case(a, s);
        if let Err(e) = speet_diff_core::run_oracle(&case) {
            println!("seed {s}: {e}");
            println!("{}", case.code.iter().map(|b| format!("{b:02x}")).collect::<String>());
            return;
        }
    }
}
