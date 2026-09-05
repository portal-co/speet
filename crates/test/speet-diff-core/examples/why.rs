fn main() {
    let s: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(speet_diff_core::parse_arch_name(std::env::args().nth(2).as_deref()), s);
    match speet_diff_core::run_recompiled(&case) {
        Ok(_) => println!("seed {s}: ok"),
        Err(e) => println!("seed {s}: {e:?}"),
    }
    println!("unsupported at translation: {:?}", speet_diff_core::unsupported_for(&case));
}
