#ifndef SPEET_CORPUS_H
#define SPEET_CORPUS_H

#include <stddef.h>
#include <stdint.h>

int speet_corpus_add(int a, int b);
int speet_corpus_mul(int a, int b);
uint32_t speet_corpus_checksum(const uint8_t *data, size_t len);
void speet_corpus_fill_buffer(uint8_t *buf, size_t len, uint8_t seed);
int speet_corpus_write_hello(void);

#endif
