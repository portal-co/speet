// Simple arithmetic for x86_64 smoke/run tests.
// Compiled with: clang --target=x86_64-unknown-none
//                -ffreestanding -nostdlib -O0 -c
// -O0 minimises the instruction variety the (incomplete) x86_64 recompiler must handle.

__attribute__((noinline))
static long add(long a, long b) { return a + b; }

__attribute__((noinline))
static long sub(long a, long b) { return a - b; }

void _start(void) {
    volatile long r1 = add(10, 32); // r1 = 42
    volatile long r2 = sub(r1, 7); // r2 = 35
    (void)r1;
    (void)r2;
}
