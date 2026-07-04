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
    volatile int g = 7, h = 8, i = 9, j = 10, k = 11, l = 12;
    volatile int buf[6] = {a, b, c, d, e, f};
    return sink(buf, 6) + g + h + i + j + k + l;
}
