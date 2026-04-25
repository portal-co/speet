// Simple arithmetic with HINT markers (addi x0, x0, N).
// Compiled with: clang --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32
//                -nostdlib -ffreestanding -O0 -c

__attribute__((noinline))
static int add(int a, int b) { return a + b; }

__attribute__((noinline))
static int mul(int a, int b) { return a * b; }

void _start(void) {
    int r1 = add(3, 4);           // r1 = 7
    __asm__ volatile("addi x0, x0, 1"); // HINT 1: a0 should be 7
    (void)r1;

    int r2 = mul(6, 7);           // r2 = 42
    __asm__ volatile("addi x0, x0, 2"); // HINT 2: a0 should be 42
    (void)r2;
}
