// Struct copy / pair return — compiler stp/ldp pressure.

typedef struct { int a; int b; } Pair;

__attribute__((noinline))
static Pair make_pair(int x, int y) {
    Pair p;
    p.a = x;
    p.b = y;
    return p;
}

__attribute__((noinline))
static int sum_pair(Pair p) { return p.a + p.b; }

int main(void) {
    Pair p = make_pair(10, 32);
    Pair q = make_pair(p.a, p.b);
    return sum_pair(q);
}
