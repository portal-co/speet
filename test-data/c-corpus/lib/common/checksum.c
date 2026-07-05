#include "speet_corpus.h"

uint32_t speet_corpus_checksum(const uint8_t *data, size_t len) {
    uint32_t acc = 0x811c9dc5u;
    for (size_t i = 0; i < len; i++) {
        acc ^= data[i];
        acc *= 0x01000193u;
    }
    return acc;
}
