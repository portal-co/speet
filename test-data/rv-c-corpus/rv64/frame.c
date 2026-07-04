__attribute__((noinline))
static int sink(volatile int *vals, int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) {
        acc += vals[i];
    }
    return acc;
}

int main(void) {
    volatile int a = 1, b = 2, c = 3, d = 4, e = 5, f = 6;
    volatile int buf[4] = {a, b, c, d};
    return sink(buf, 4) + e + f;
}
