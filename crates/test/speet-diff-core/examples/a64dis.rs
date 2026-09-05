//! Disassemble a few words via disarm64 (the recompiler's own decoder).
fn main() {
    for arg in std::env::args().skip(1) {
        let w = u32::from_str_radix(arg.trim_start_matches("0x"), 16).unwrap();
        match disarm64::decoder::decode(w) {
            Some(inst) => println!("{w:08x} → {inst}"),
            None => println!("{w:08x} → DECODE FAIL"),
        }
    }
}
