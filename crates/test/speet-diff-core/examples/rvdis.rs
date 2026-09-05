fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generate_case(speet_diff_core::case::Arch::RiscV64, seed);
    for i in (0..case.code.len()).step_by(4) {
        let w = u32::from_le_bytes(case.code[i..i+4].try_into().unwrap());
        println!("{:#x}: {:08x} {}", case.entry_pc + i as u64, w,
            match rv_asm::Inst::decode(w, rv_asm::Xlen::Rv64) {
                Ok((inst, _)) => format!("{inst:?}"),
                Err(e) => format!("BAD {e:?}"),
            });
    }
}
