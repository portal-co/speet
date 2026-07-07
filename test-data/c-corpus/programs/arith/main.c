#include "speet_corpus.h"

int main(void) {
    int a = speet_corpus_add(10, 6);
    int b = speet_corpus_mul(5, 6);
    return speet_corpus_add(a, b);
}
