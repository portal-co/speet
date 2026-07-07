#include <unistd.h>

#include "speet_corpus.h"

int speet_corpus_write_hello(void) {
    const char msg[] = "hello\n";
    return (int)write(1, msg, sizeof msg - 1);
}
