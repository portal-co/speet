fn main() {
    let mut uc = unicorn_engine::Unicorn::new(unicorn_engine::unicorn_const::Arch::ARM64, unicorn_engine::unicorn_const::Mode::LITTLE_ENDIAN).unwrap();
    uc.mem_map(0x100000, 0x1000, unicorn_engine::unicorn_const::Prot::ALL).unwrap();
    uc.mem_write(0x100000, &0xD65F03C0u32.to_le_bytes()).unwrap();
    uc.reg_write(unicorn_engine::RegisterARM64::X30, 0x100004u64).unwrap();
    uc.reg_write(unicorn_engine::RegisterARM64::NZCV, 0u64).unwrap();
    println!("pre-run NZCV = {:#x}", uc.reg_read(unicorn_engine::RegisterARM64::NZCV).unwrap());
    uc.emu_start(0x100000, 0x100004, 0, 10).unwrap();
    println!("post-run NZCV = {:#x}", uc.reg_read(unicorn_engine::RegisterARM64::NZCV).unwrap());
}
