#include "../include/tokenizer_c.h"
#include <string.h>

Tokenizer* create_tokenizer() {
    Tokenizer* t = malloc(sizeof(Tokenizer));
    if (!t) return NULL;
    t->size = 0;
    t->capacity = 100;
    t->tokens = malloc(sizeof(Token) * t->capacity);
    if (!t->tokens) {
        free(t);
        return NULL;
    }
    return t;
}

void free_tokenizer(Tokenizer* t) {
    if (!t) return;
    free(t->tokens);
    free(t);
}

void load_vocab_c(Tokenizer* t, const char* path) {
    FILE* f;
    if (!path) {
        const char* alt_paths[] = {
            "vocabulario_es.txt",
            "../calculus_llm/vocabulario_es.txt",
            "../calculus_llm/build/vocabulario_es.txt",
            NULL
        };
        for (int i = 0; alt_paths[i]; i++) {
            f = fopen(alt_paths[i], "r");
            if (f) { path = alt_paths[i]; break; }
        }
        if (!f) return;
    } else {
        f = fopen(path, "r");
        if (!f) return;
    }
    char word[64];
    while (fscanf(f, "%63s", word) == 1) {
        if (t->size >= t->capacity) {
            int new_cap = t->capacity * 2;
            Token* new_tokens = realloc(t->tokens, sizeof(Token) * new_cap);
            if (!new_tokens) {
                fprintf(stderr, "[ERROR] No hay memoria para el vocabulario.\n");
                return;
            }
            t->tokens = new_tokens;
            t->capacity = new_cap;
        }
        strcpy(t->tokens[t->size].word, word);
        // Embedding determinista
        unsigned long hash = 5381;
        for (int i = 0; word[i]; i++) hash = ((hash << 5) + hash) + (unsigned char)word[i];
        for (int j = 0; j < DIM; j++) {
            t->tokens[t->size].vec.data[j] = sin(hash * (j + 1) * 0.1);
        }
        t->size++;
    }
    fclose(f);
}

State encode_c(Tokenizer* t, const char* word) {
    for (int i = 0; i < t->size; i++) {
        if (strcmp(t->tokens[i].word, word) == 0) return t->tokens[i].vec;
    }
    State zero = {{0}};
    return zero;
}

const char* decode_c(Tokenizer* t, State s) {
    int best = -1;
    double min_dist = 1e18;
    for (int i = 0; i < t->size; i++) {
        double d = norm(sub(s, t->tokens[i].vec));
        if (d < min_dist) {
            min_dist = d;
            best = i;
        }
    }
    return (best != -1) ? t->tokens[best].word : "unknown";
}
