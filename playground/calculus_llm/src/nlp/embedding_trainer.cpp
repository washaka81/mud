#include "embedding_trainer.h"
#include "../config.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace nlp {

EmbeddingTrainer::EmbeddingTrainer(EmbeddingMatrix& embed, BpeTokenizer& tokenizer,
                                   size_t context_size, double learning_rate,
                                   int negative_samples)
    : embed_(embed), tokenizer_(tokenizer),
      context_size_(context_size), learning_rate_(learning_rate),
      negative_samples_(negative_samples), rng_(42) {}

void EmbeddingTrainer::train_step(const std::vector<token_id_t>& context, token_id_t target) {
    size_t dim = embed_.dim();
    size_t vocab_size = embed_.vocab_size();
    size_t t = static_cast<size_t>(target);
    if (t >= vocab_size) return;

    // Average context embeddings → h
    math::State h = math::State::Zero(dim);
    size_t valid_count = 0;
    for (auto id : context) {
        if (id >= 0 && static_cast<size_t>(id) < vocab_size && id != BpeTokenizer::UNK_ID) {
            h += embed_.forward(id);
            valid_count++;
        }
    }
    if (valid_count == 0) return;
    double inv_count = 1.0 / static_cast<double>(valid_count);
    for (size_t j = 0; j < dim; ++j) h[j] *= inv_count;

    // Fetch target embedding once
    math::State w_target = embed_.forward(target);

    // Target logit = w_target · h
    double target_logit = 0.0;
    for (size_t j = 0; j < dim; ++j) target_logit += w_target[j] * h[j];

    // Sample negative tokens (uniform over non-special tokens)
    std::vector<token_id_t> neg_samples;
    {
        std::uniform_int_distribution<size_t> dist(5, vocab_size - 1);
        for (int k = 0; k < negative_samples_; ++k) {
            token_id_t neg = static_cast<token_id_t>(dist(rng_));
            if (neg != target) neg_samples.push_back(neg);
        }
    }

    // Compute negative logits
    std::vector<double> neg_logits(neg_samples.size(), 0.0);
    for (size_t k = 0; k < neg_samples.size(); ++k) {
        math::State w_neg = embed_.forward(neg_samples[k]);
        for (size_t j = 0; j < dim; ++j) neg_logits[k] += w_neg[j] * h[j];
    }

    // Negative sampling loss (binary cross-entropy over sigmoids)
    double target_sig = 1.0 / (1.0 + std::exp(-target_logit));
    double loss = -std::log(std::max(target_sig, 1e-15));
    for (auto nl : neg_logits) {
        double neg_sig = 1.0 / (1.0 + std::exp(-nl));
        loss -= std::log(std::max(1.0 - neg_sig, 1e-15));
    }
    stats_.total_loss += loss;
    stats_.steps++;

    // Count top-1 accuracy (target should have highest logit among sampled)
    bool correct = true;
    for (auto nl : neg_logits) if (nl > target_logit) { correct = false; break; }
    if (correct) stats_.correct_top1++;

    // Apply SGD updates:
    //   w_target -= lr * dL/d_w = w_target - lr * (sigmoid - 1) * h
    //   w_neg    -= lr * dL/d_w = w_neg    - lr * sigmoid * h
    math::State target_grad = math::mul(h, target_sig - 1.0);
    embed_.update_row(target, target_grad, learning_rate_);

    for (size_t k = 0; k < neg_samples.size(); ++k) {
        double neg_sig = 1.0 / (1.0 + std::exp(-neg_logits[k]));
        math::State neg_grad = math::mul(h, neg_sig);
        embed_.update_row(neg_samples[k], neg_grad, learning_rate_);
    }
}

void EmbeddingTrainer::train(const std::vector<std::vector<token_id_t>>& sequences, int epochs) {
    std::cout << "[EmbedTrainer] Training on " << sequences.size()
              << " sequences, context=" << context_size_
              << ", lr=" << learning_rate_
              << ", neg_samples=" << negative_samples_ << "\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        reset_stats();
        int total_examples = 0;

        for (const auto& seq : sequences) {
            if (seq.size() < context_size_ + 1) continue;
            for (size_t pos = context_size_; pos < seq.size(); ++pos) {
                std::vector<token_id_t> ctx(seq.begin() + pos - context_size_,
                                           seq.begin() + pos);
                token_id_t target = seq[pos];
                if (target == BpeTokenizer::PAD_ID || target == BpeTokenizer::UNK_ID ||
                    target == BpeTokenizer::BOS_ID || target == BpeTokenizer::EOS_ID ||
                    target == BpeTokenizer::MASK_ID) continue;
                train_step(ctx, target);
                total_examples++;
            }
        }

        double avg_loss = stats_.total_loss / std::max(1, stats_.steps);
        double acc = 100.0 * stats_.correct_top1 / std::max(1, stats_.steps);
        std::cout << "[EmbedTrainer] Epoch " << (epoch + 1) << "/" << epochs
                  << " | examples: " << total_examples
                  << " | avg_loss: " << avg_loss
                  << " | acc: " << acc << "%\n";
    }
}

void EmbeddingTrainer::train_on_text(const std::string& corpus_path, int epochs) {
    std::ifstream file(corpus_path);
    if (!file.is_open()) {
        std::cerr << "[EmbedTrainer] Cannot open " << corpus_path << "\n";
        return;
    }

    std::vector<std::vector<token_id_t>> sequences;
    sequences.reserve(12000);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.size() < 3) continue;
        auto ids = tokenizer_.encode(line);
        if (ids.size() >= context_size_ + 1) {
            sequences.push_back(std::move(ids));
        }
    }

    std::cout << "[EmbedTrainer] Tokenized " << sequences.size()
              << " sequences from " << corpus_path << "\n";
    train(sequences, epochs);

    embed_.save(config::EMBEDDING_WEIGHTS_FILENAME);
    std::cout << "[EmbedTrainer] Saved embeddings to "
              << config::EMBEDDING_WEIGHTS_FILENAME << "\n";
}

} // namespace nlp
