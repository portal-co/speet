fn main() {
    let a = match std::env::args().nth(1).as_deref() {
        Some("aarch64") => speet_diff_core::case::Arch::AArch64,
        Some("riscv64") => speet_diff_core::case::Arch::RiscV64,
        _ => speet_diff_core::case::Arch::X86_64,
    };
    use std::str::FromStr;
    for s in 1u64..60 {
        let case = speet_diff_core::generator::generate_case(a, s);
        if let Err(e) = speet_diff_core::run_recompiled(&case) {
            println!("seed {s}: {e:?}");
        }
    }
}
