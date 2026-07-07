__attribute__((noinline))
static int add(int a, int b) { return a + b; }

__attribute__((noinline))
static int mul(int a, int b) { return a * b; }

int main(void) {
    int r1 = add(3, 4);
    int r2 = mul(6, 7);
    return r1 + r2 - 5;
}
