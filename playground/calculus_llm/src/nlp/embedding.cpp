#include "embedding.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

namespace nlp {

EmbeddingMatrix::EmbeddingMatrix(size_t vocab_size, size_t dim)
    : vocab_size_(vocab_size), dim_(dim),
      weights_(math::Matrix::Zero(vocab_size, dim)),
      grad_accum_(math::Matrix::Zero(vocab_size, dim)),
      rng_(42)
{
    // Xavier-like initialization
    double scale = std::sqrt(2.0 / dim);
    std::normal_distribution<double> dist(0.0, scale);
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            weights_(i, j) = dist(rng_);
        }
    }
}

math::State EmbeddingMatrix::forward(token_id_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= vocab_size_) {
        return math::State::Zero();
    }
    math::State result;
    for (size_t j = 0; j < dim_; ++j) {
        result[j] = weights_(id, j);
    }
    return result;
}

std::vector<math::State> EmbeddingMatrix::forward_batch(const std::vector<token_id_t>& ids) const {
    std::vector<math::State> result;
    result.reserve(ids.size());
    for (auto id : ids) {
        result.push_back(forward(id));
    }
    return result;
}

Eigen::VectorXd EmbeddingMatrix::logits(const math::State& state) const {
    // logits = weights * state  (vocab_size × dim * dim × 1 → vocab_size × 1)
    Eigen::VectorXd result(vocab_size_);
    for (size_t i = 0; i < vocab_size_; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < dim_; ++j) {
            dot += weights_(i, j) * state[j];
        }
        result[i] = dot;
    }
    return result;
}

token_id_t EmbeddingMatrix::sample(const math::State& state, double temperature,
                                     const std::vector<token_id_t>& taboo) const {
    Eigen::VectorXd logits_vec = logits(state);

    // Find max logit for numerical stability
    double max_logit = logits_vec[0];
    for (size_t i = 1; i < vocab_size_; ++i) {
        if (logits_vec[i] > max_logit) max_logit = logits_vec[i];
    }

    // Build taboo mask + softmax
    std::vector<double> probs(vocab_size_);
    double sum = 0.0;
    for (size_t i = 0; i < vocab_size_; ++i) {
        bool is_taboo = !taboo.empty() &&
            std::find(taboo.begin(), taboo.end(), static_cast<token_id_t>(i)) != taboo.end();
        if (is_taboo) {
            probs[i] = 0.0;
        } else {
            probs[i] = std::exp((logits_vec[i] - max_logit) / temperature);
        }
        sum += probs[i];
    }

    if (sum <= 0.0) {
        // Fallback: argmax
        size_t best = 0;
        double best_val = logits_vec[0];
        for (size_t i = 1; i < vocab_size_; ++i) {
            if (logits_vec[i] > best_val) {
                best_val = logits_vec[i];
                best = i;
            }
        }
        return static_cast<token_id_t>(best);
    }

    // Normalize
    for (size_t i = 0; i < vocab_size_; ++i) probs[i] /= sum;

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return static_cast<token_id_t>(dist(rng_));
}

void EmbeddingMatrix::zero_grad() {
    grad_accum_.setZero();
}

void EmbeddingMatrix::accumulate_grad(token_id_t id, const math::State& grad) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_size_) return;
    for (size_t j = 0; j < dim_; ++j) {
        grad_accum_(id, j) += grad[j];
    }
}

void EmbeddingMatrix::accumulate_grad_batch(const std::vector<token_id_t>& ids,
                                              const std::vector<math::State>& grads) {
    for (size_t k = 0; k < ids.size() && k < grads.size(); ++k) {
        accumulate_grad(ids[k], grads[k]);
    }
}

void EmbeddingMatrix::sgd_step(double lr) {
    for (size_t i = 0; i < vocab_size_; ++i) {
        for (size_t j = 0; j < dim_; ++j) {
            weights_(i, j) -= lr * grad_accum_(i, j);
        }
    }
}

void EmbeddingMatrix::update_row(token_id_t id, const math::State& grad, double lr) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_size_) return;
    for (size_t j = 0; j < dim_; ++j) {
        weights_(id, j) -= lr * grad[j];
    }
}

bool EmbeddingMatrix::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Embedding] Error: cannot write " << path << "\n";
        return false;
    }
    int rows = static_cast<int>(vocab_size_);
    int cols = static_cast<int>(dim_);
    f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char*>(weights_.data()),
            rows * cols * sizeof(double));
    return true;
}

bool EmbeddingMatrix::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    int rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (static_cast<size_t>(rows) != vocab_size_ || static_cast<size_t>(cols) != dim_) {
        std::cerr << "[Embedding] Dimension mismatch: expected "
                  << vocab_size_ << "x" << dim_ << ", got "
                  << rows << "x" << cols << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(weights_.data()),
           rows * cols * sizeof(double));
    return true;
}

math::State& EmbeddingMatrix::weight(token_id_t id) {
    // Return a view as State (we need to map the row to a State)
    // For Eigen map, we create a copy since the matrix is column-major
    static math::State tmp;
    if (id < 0 || static_cast<size_t>(id) >= vocab_size_) {
        tmp = math::State::Zero();
        return tmp;
    }
    tmp = math::State();
    for (size_t j = 0; j < dim_; ++j) {
        tmp[j] = weights_(id, j);
    }
    return tmp;
}

const math::State& EmbeddingMatrix::weight(token_id_t id) const {
    static math::State tmp;
    if (id < 0 || static_cast<size_t>(id) >= vocab_size_) {
        tmp = math::State::Zero();
        return tmp;
    }
    tmp = math::State();
    for (size_t j = 0; j < dim_; ++j) {
        tmp[j] = weights_(id, j);
    }
    return tmp;
}

} // namespace nlp
