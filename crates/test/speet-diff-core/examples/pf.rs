use speet_diff_core::tests_common::make_case;
fn probe(code: Vec<u8>, name: &str) {
    let case = make_case(code, 1);
    let o = speet_diff_core::run_oracle(&case).expect("oracle");
    let r = speet_diff_core::run_recompiled(&case).expect("recompiled");
    println!("{name}: pf oracle={} recompiled={} zf o={} r={}",
        o.regs.pf, r.regs.pf, o.regs.zf, r.regs.zf);
}
fn main() {
    // mov rax, 1; add rax, 3 → result 4 (1 bit) → PF=0 (odd parity)... PF set when even number of 1 bits: 4=100b → one 1 → PF=0
    probe(vec![0x48,0xB8,0x01,0,0,0,0,0,0,0, 0x48,0x83,0xC0,0x03, 0xC3], "add 1+3");
    // result with 2 ones: 3+3=6 (110b) two ones → PF=1
    probe(vec![0x48,0xB8,0x03,0,0,0,0,0,0,0, 0x48,0x83,0xC0,0x03, 0xC3], "add 3+3");
    // sub: 10-8=2 → one one → PF=0
    probe(vec![0x48,0xB8,0x0A,0,0,0,0,0,0,0, 0x48,0x83,0xE8,0x08, 0xC3], "sub 10-8");
    // or: 1|2=3 → two ones → PF=1
    probe(vec![0x48,0xB8,0x01,0,0,0,0,0,0,0, 0x48,0x83,0xC8,0x02, 0xC3], "or 1|2");
    // and: 3&1=1 → PF=0
    probe(vec![0x48,0xB8,0x03,0,0,0,0,0,0,0, 0x48,0x83,0xE0,0x01, 0xC3], "and 3&1");
    // xor: 3^1=2 → PF=0
    probe(vec![0x48,0xB8,0x03,0,0,0,0,0,0,0, 0x48,0x83,0xF0,0x01, 0xC3], "xor 3^1");
    // 16-bit op: mov ax,3; or ax,1 → 3 → PF=1
    probe(vec![0x66,0xB8,0x03,0, 0x66,0x83,0xC8,0x01, 0xC3], "or ax 3|1");
}
