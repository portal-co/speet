fn main() {
    use unicorn_engine::unicorn_const::{Arch, Mode, Prot};
    let mut uc = unicorn_engine::Unicorn::new(Arch::ARM, Mode::ARM).unwrap();
    uc.mem_map(0x100000, 0x1000, Prot::ALL).unwrap();
    uc.mem_write(0x100000, &0xE12FFF1Eu32.to_le_bytes()).unwrap();
    let two = [0xE1A11000u32.to_le_bytes(), 0xE12FFF1Eu32.to_le_bytes()].concat();
    uc.mem_write(0x100000, &two).unwrap();
    uc.reg_write(unicorn_engine::RegisterARM::LR, 0x100004u64).unwrap();
    uc.reg_write(unicorn_engine::RegisterARM::CPSR, 0u64).unwrap();
    match uc.emu_start(0x100000, 0x100004, 0, 10) {
        Ok(()) => println!("OK"),
        Err(e) => println!("ERR {e:?}"),
    }
}
