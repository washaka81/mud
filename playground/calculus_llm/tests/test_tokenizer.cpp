#include "../src/nlp/tokenizer.h"
#include "../src/config.h"
#include <iostream>
#include <cassert>
#include <cmath>

static int test_failures = 0;

#define TEST_ASSERT(condition, msg) do { \
    if (!(condition)) { \
        std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        test_failures++; \
    } else { \
        std::cout << "  [OK] " << msg << std::endl; \
    } \
} while(0)

void test_encode_decode_roundtrip() {
    std::cout << ">>> Test: Tokenizer Encode/Decode Roundtrip\n";
    nlp::Tokenizer tok(config::EMBEDDING_DIM);
    std::string word = "verdad";
    math::State vec = tok.encode(word);
    std::string decoded = tok.decode(vec);
    TEST_ASSERT(word == decoded, "Decoded word should match encoded word if exact vector is used.");
}

void test_update_vector() {
    std::cout << ">>> Test: Tokenizer Update Vector\n";
    nlp::Tokenizer tok(config::EMBEDDING_DIM);
    std::string word = "verdad";
    math::State original = tok.encode(word);
    
    math::State new_vec = original;
    new_vec[0] += 10.0;
    tok.update_vector(word, new_vec);
    
    math::State updated = tok.encode(word);
    TEST_ASSERT(std::abs(updated[0] - original[0] - 10.0) < 1e-6, "Vector should be updated.");
}

void test_save_load_roundtrip() {
    std::cout << ">>> Test: Tokenizer Save/Load Roundtrip\n";
    nlp::Tokenizer tok1(config::EMBEDDING_DIM);
    tok1.update_vector("test_word", math::State::Constant(config::EMBEDDING_DIM, 0.42));
    tok1.save_to_file("test_vocab.txt");

    nlp::Tokenizer tok2(config::EMBEDDING_DIM);
    bool loaded = tok2.load_from_file("test_vocab.txt");
    TEST_ASSERT(loaded, "Vocabulary should load successfully.");
    
    math::State vec = tok2.encode("test_word");
    TEST_ASSERT(std::abs(vec[0] - 0.42) < 1e-6, "Loaded vector should match saved vector.");
}

void test_sample_returns_valid() {
    std::cout << ">>> Test: Tokenizer Sample\n";
    nlp::Tokenizer tok(config::EMBEDDING_DIM);
    math::State vec = tok.encode("verdad");
    std::string sampled = tok.sample(vec, 1.0);
    TEST_ASSERT(sampled != "unknown" && sampled != "", "Sample should return a valid word.");
}

void test_unknown_word_handling() {
    std::cout << ">>> Test: Tokenizer Unknown Word\n";
    nlp::Tokenizer tok(config::EMBEDDING_DIM);
    math::State vec = tok.encode("palabra_inexistente_123");
    TEST_ASSERT(vec.norm() > 1e-3, "Unknown word should encode to a non-zero deterministic hash vector.");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Test Suite: NLP Tokenizer            \n";
    std::cout << "========================================\n\n";

    test_encode_decode_roundtrip();
    test_update_vector();
    test_save_load_roundtrip();
    test_sample_returns_valid();
    test_unknown_word_handling();

    if (test_failures > 0) {
        std::cerr << "\n[RESULT] " << test_failures << " tests failed!\n";
        return 1;
    }
    
    std::cout << "\n[RESULT] All tests passed.\n";
    return 0;
}
