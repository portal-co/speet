#include "speet_corpus.h"

static uint8_t data[16];

int main(void) {
    speet_corpus_fill_buffer(data, sizeof data, 7);
    return (int)(speet_corpus_checksum(data, sizeof data) & 0x7fffffffu);
}
