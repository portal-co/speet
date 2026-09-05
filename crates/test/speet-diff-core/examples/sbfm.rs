//! Probe: SBFM/UBFM/ASR/LSL forms vs the oracle (finding #2 regression check).
fn main() {
    use speet_diff_core::case::{Arch, FuzzCase, RegState};
    use speet_diff_core::{run_oracle, run_recompiled, compare_outcomes};
    // (word, x13_initial) pairs exercising bitfield edge combos
    let words: Vec<(u32, u64)> = vec![
        // 64-bit, dst X4
        (0xd379e1a4, 0x8000000000000000), // UBFM X4,X13,#57,#56 = LSL #7
        (0xd379e1a4, 0xffffffffffffff80), // LSL #7 of small negative
        (0x9379fda4, 0x8000000000000000), // SBFM X4,X13,#57,#63 = ASR #57
        (0x9379fda4, 0xffffffffffffff80), // ASR #57 of small negative
        (0xd343fda4, 0xdeadbeefcafebabe), // UBFM X4,X13,#3,#63 = LSR #3
        (0x9343fc04, 0x8000000000000000), // SBFM X4,X0,#3,#63 = ASR #3
        (0xd340fc04, 0xdeadbeefcafebabe), // UBFM X4,X0,#0,#63 = move
        (0x93401c04, 0xdeadbeefcafebabe), // SBFM X4,X0,#0,#7 = SXTB
        (0xd3407c04, 0xdeadbeefcafebabe), // UBFM X4,X0,#0,#31 = UXTW (N=1, 32-bit extract)
        // 32-bit (N=0), dst W4, src W0
        (0x13077c04, 0xffffffff12345678), // SBFM W4,W0,#7,#31 = ASR W4,W0,#7
        (0x53077c04, 0xffffffff12345678), // UBFM W4,W0,#7,#31 = LSR W4,W0,#7
        (0x53001c04, 0xffffffff12345678), // UBFM W4,W0,#0,#7 = UXTB
        (0x13001c04, 0xffffffff12345678), // SBFM W4,W0,#0,#7 = SXTB w-form
        (0x531f7804, 0xffffffff12345678), // UBFM W4,W0,#31,#30 = LSL W4,W0,#1
    ];
    for (word, x13) in words {
        let code: Vec<u8> = word.to_le_bytes().iter().chain(0xd65f03c0u32.to_le_bytes().iter()).copied().collect();
        let mut regs = RegState::default();
        regs.gprs[13] = x13;
        regs.gprs[0] = x13;
        regs.gprs[30] = 0x100008; // halt sentinel
        let case = FuzzCase {
            arch: Arch::AArch64,
            code,
            entry_pc: 0x100000,
            regs,
            data: vec![0u8; 0x2000],
            data_base: 0x200000,
            stack: vec![0u8; 0x8000],
            stack_base: 0x300000,
            read_only: vec![],
            step_budget: 4000,
            seed: 0,
        };
        let o = run_oracle(&case);
        let r = run_recompiled(&case);
        let v = compare_outcomes(&case, o.as_ref().ok(), r.as_ref());
        let label = match v { speet_diff_core::Comparison::Match => "ok".to_string(),
            speet_diff_core::Comparison::Divergence(d) => format!("DIVERGE {d}"),
            speet_diff_core::Comparison::Skip(s) => format!("SKIP {s:?}") };
        let ov = o.map(|x| x.regs.gprs[4]).unwrap_or(0);
        let rv = r.map(|x| x.regs.gprs[4]).unwrap_or(0);
        println!("{word:08x} x13={x13:#018x} → oracle dst={ov:#x} recomp dst={rv:#x} {label}");
    }
}
