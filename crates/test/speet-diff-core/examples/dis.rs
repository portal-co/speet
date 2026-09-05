use iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};
fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let case = speet_diff_core::generator::generate_case(speet_diff_core::parse_arch_name(std::env::args().nth(2).as_deref()), seed);
    let mut dec = Decoder::with_ip(64, &case.code, case.entry_pc, DecoderOptions::NONE);
    let mut out = String::new();
    let mut f = NasmFormatter::new();
    while dec.can_decode() {
        let i = dec.decode();
        out.clear();
        f.format(&i, &mut out);
        println!("{:#x}: {} (len {})", i.ip(), out, i.len());
    }
}
