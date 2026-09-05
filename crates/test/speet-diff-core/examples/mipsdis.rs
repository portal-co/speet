//! Disassemble big-endian MIPS words via rabbitizer.
fn main() {
    for arg in std::env::args().skip(1) {
        let w = u32::from_str_radix(arg.trim_start_matches("0x"), 16).unwrap();
        let inst = rabbitizer::Instruction::new(w, 0x100000, rabbitizer::InstrCategory::CPU);
        println!("{w:08x} → {}", inst.disassemble(None, 0));
    }
}
