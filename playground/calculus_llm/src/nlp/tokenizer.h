#ifndef CALCULUS_LLM_NLP_TOKENIZER_H
#define CALCULUS_LLM_NLP_TOKENIZER_H

#include "../math/calculus.h"
#include "../config.h"
#include "bpe_tokenizer.h"
#include "embedding.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>

namespace nlp {

class Tokenizer {
public:
    Tokenizer(size_t dim = config::EMBEDDING_DIM);

    // BPE-based encoding/decoding
    math::State encode(const std::string& word) const;
    std::string decode(const math::State& state) const;

    // Find closest word in vocabulary (Levenshtein)
    std::string find_closest_word(const std::string& word) const;
    static size_t levenshtein_distance(const std::string& s1, const std::string& s2);

    // Probabilistic sampling
    std::string sample(const math::State& state, double temperature = 1.0) const;
    std::string sample_restricted(const math::State& state,
                                   const std::vector<std::string>& candidates,
                                   const std::vector<std::string>& taboo,
                                   double temperature = 1.0) const;

    // Vocabulary access
    std::vector<std::string> get_vocabulary() const;
    math::State get_vector(const std::string& word) const;
    void update_vector(const std::string& word, const math::State& new_vec);

    // Persistence
    bool save_to_file(const std::string& path) const;
    bool load_from_file(const std::string& path);

    // BPE tokenizer access
    BpeTokenizer& bpe() { return bpe_; }
    const BpeTokenizer& bpe() const { return bpe_; }

    // Embedding matrix access
    EmbeddingMatrix& embeddings() { return *embed_; }
    const EmbeddingMatrix& embeddings() const { return *embed_; }

private:
    size_t dimension;
    std::map<std::string, math::State> word_to_vec;
    std::unordered_map<std::string, math::State> word_embed_cache_;
    BpeTokenizer bpe_;
    std::unique_ptr<EmbeddingMatrix> embed_;
    mutable std::mt19937 rng_sample;

    void initialize_vocabulary();
    bool load_word_list(const std::string& path);
    math::State compute_word_embedding(const std::string& word) const;
    void sync_word_to_vec_from_embeddings();
};

} // namespace nlp

#endif // CALCULUS_LLM_NLP_TOKENIZER_H
