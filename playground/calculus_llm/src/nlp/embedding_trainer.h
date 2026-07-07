#ifndef CALCULUS_LLM_NLP_EMBEDDING_TRAINER_H
#define CALCULUS_LLM_NLP_EMBEDDING_TRAINER_H

#include "embedding.h"
#include "bpe_tokenizer.h"
#include <string>
#include <vector>
#include <random>

namespace nlp {

struct TrainingStats {
    int steps = 0;
    double total_loss = 0.0;
    int correct_top1 = 0;
    int correct_top5 = 0;
};

class EmbeddingTrainer {
public:
    EmbeddingTrainer(EmbeddingMatrix& embed, BpeTokenizer& tokenizer,
                     size_t context_size = 4, double learning_rate = 0.01,
                     int negative_samples = 5);

    void train(const std::vector<std::vector<token_id_t>>& sequences, int epochs = 5);
    void train_on_text(const std::string& corpus_path, int epochs = 5);
    void train_step(const std::vector<token_id_t>& context, token_id_t target);

    const TrainingStats& stats() const { return stats_; }
    void reset_stats() { stats_ = TrainingStats{}; }

private:
    EmbeddingMatrix& embed_;
    BpeTokenizer& tokenizer_;
    size_t context_size_;
    double learning_rate_;
    int negative_samples_;
    TrainingStats stats_;
    mutable std::mt19937 rng_;
};

} // namespace nlp

#endif // CALCULUS_LLM_NLP_EMBEDDING_TRAINER_H
