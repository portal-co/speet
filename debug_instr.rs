use rabbitizer::{Instruction, InstrCategory};

fn main() {
    // Test different branch instructions
    let beq = Instruction::new(0x11290004, 0x1000, InstrCategory::CPU);
    println!("BEQ: unique_id = {:?}", beq.unique_id);

    let bne = Instruction::new(0x15290004, 0x1000, InstrCategory::CPU);
    println!("BNE: unique_id = {:?}", bne.unique_id);

    let blez = Instruction::new(0x19200004, 0x1000, InstrCategory::CPU);
    println!("BLEZ: unique_id = {:?}", blez.unique_id);

    let bgtz = Instruction::new(0x1D200004, 0x1000, InstrCategory::CPU);
    println!("BGTZ: unique_id = {:?}", bgtz.unique_id);

    let bltz = Instruction::new(0x05200004, 0x1000, InstrCategory::CPU);
    println!("BLTZ: unique_id = {:?}", bltz.unique_id);

    let bgez = Instruction::new(0x05210004, 0x1000, InstrCategory::CPU);
    println!("BGEZ: unique_id = {:?}", bgez.unique_id);
}