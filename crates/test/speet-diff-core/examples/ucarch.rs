fn main() {
    let uc = unicorn_engine::Unicorn::new(unicorn_engine::unicorn_const::Arch::ARM64, unicorn_engine::unicorn_const::Mode::LITTLE_ENDIAN).unwrap();
    println!("aarch64 OK");
    let uc2 = unicorn_engine::Unicorn::new(unicorn_engine::unicorn_const::Arch::RISCV, unicorn_engine::unicorn_const::Mode::RISCV64).unwrap();
    println!("riscv64 OK");
}
