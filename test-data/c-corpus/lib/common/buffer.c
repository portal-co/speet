#include "speet_corpus.h"

void speet_corpus_fill_buffer(uint8_t *buf, size_t len, uint8_t seed) {
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)(seed + (uint8_t)i);
    }
}

int speet_corpus_add(int a, int b) { return a + b; }

int speet_corpus_mul(int a, int b) { return a * b; }
