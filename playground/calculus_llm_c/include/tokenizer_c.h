#ifndef TOKENIZER_C_H
#define TOKENIZER_C_H

#include "math_c.h"

typedef struct {
    char word[64];
    State vec;
} Token;

typedef struct {
    Token* tokens;
    int size;
    int capacity;
} Tokenizer;

Tokenizer* create_tokenizer();
void free_tokenizer(Tokenizer* t);
State encode_c(Tokenizer* t, const char* word);
const char* decode_c(Tokenizer* t, State s);
void load_vocab_c(Tokenizer* t, const char* path);

#endif
