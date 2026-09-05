fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generate_case(speet_diff_core::case::Arch::Arm, seed);
    for i in (0..case.code.len()).step_by(4) {
        let w = u32::from_le_bytes(case.code[i..i+4].try_into().unwrap());
        println!("{:#x}: {:08x}", case.entry_pc + i as u64, w);
    }
}
