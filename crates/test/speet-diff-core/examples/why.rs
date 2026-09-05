
fn parse_arch(s: Option<&String>) -> speet_diff_core::case::Arch {
    match s.map(|x| x.as_str()) {
        Some("aarch64") => speet_diff_core::case::Arch::AArch64,
        Some("riscv64") => speet_diff_core::case::Arch::RiscV64,
        _ => speet_diff_core::case::Arch::X86_64,
    }
}
fn main() {
    let s: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(parse_arch(std::env::args().nth(2).as_ref()), s);
    match speet_diff_core::run_recompiled(&case) {
        Ok(_) => println!("seed {s}: ok"),
        Err(e) => println!("seed {s}: {e:?}"),
    }
    println!("unsupported at translation: {:?}", speet_diff_core::unsupported_for(&case));
}
