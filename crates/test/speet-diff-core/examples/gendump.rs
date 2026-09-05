fn main() {
    let arch = speet_diff_core::parse_arch_name(std::env::args().nth(1).as_deref());
    let n: u64 = std::env::args().nth(2).and_then(|x| x.parse().ok()).unwrap_or(5);
    if arch.code_align() == 4 {
        // word-arch mode: one 8-hex-digit word per line
        let mut all = std::collections::HashSet::new();
        for s in 1..=n * 20 {
            let case = speet_diff_core::generate_case(arch, s);
            for i in (0..case.code.len()).step_by(4) {
                let w = u32::from_le_bytes(case.code[i..i+4].try_into().unwrap());
                all.insert(format!("{w:08x}"));
            }
            if all.len() > 400 { break; }
        }
        for w in all { println!("{w}"); }
    } else {
        // byte-arch mode: one whole-case hex blob per line
        for s in 1..=n {
            let case = speet_diff_core::generate_case(arch, s);
            println!("{}", case.code.iter().map(|b| format!("{b:02x}")).collect::<String>());
        }
    }
}
