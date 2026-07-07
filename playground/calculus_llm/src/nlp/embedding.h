#ifndef CALCULUS_LLM_NLP_EMBEDDING_H
#define CALCULUS_LLM_NLP_EMBEDDING_H

#include "../math/calculus.h"
#include "../config.h"
#include "bpe_tokenizer.h"
#include <string>
#include <vector>
#include <random>

namespace nlp {

class EmbeddingMatrix {
public:
    EmbeddingMatrix(size_t vocab_size, size_t dim = config::EMBEDDING_DIM);

    math::State forward(token_id_t id) const;
    std::vector<math::State> forward_batch(const std::vector<token_id_t>& ids) const;

    // Output projection: state → logits over vocab
    Eigen::VectorXd logits(const math::State& state) const;
    token_id_t sample(const math::State& state, double temperature = 1.0,
                      const std::vector<token_id_t>& taboo = {}) const;

    void sgd_step(double lr);
    void zero_grad();
    void accumulate_grad(token_id_t id, const math::State& grad);
    void accumulate_grad_batch(const std::vector<token_id_t>& ids,
                                const std::vector<math::State>& grads);

    // Save/load full matrix
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    size_t vocab_size() const { return vocab_size_; }
    size_t dim() const { return dim_; }

    void update_row(token_id_t id, const math::State& grad, double lr);

    math::State& weight(token_id_t id);
    const math::State& weight(token_id_t id) const;

private:
    size_t vocab_size_;
    size_t dim_;
    math::Matrix weights_;
    math::Matrix grad_accum_;
    mutable std::mt19937 rng_;
};

} // namespace nlp

#endif // CALCULUS_LLM_NLP_EMBEDDING_H
