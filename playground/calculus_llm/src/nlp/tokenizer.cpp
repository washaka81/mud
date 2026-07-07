#include "tokenizer.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <numeric>
#include <filesystem>

namespace nlp {

Tokenizer::Tokenizer(size_t dim) : dimension(dim), rng_sample(std::random_device{}()) {
    // Load BPE tokenizer
    std::string bpe_vocab = config::BPE_VOCAB_FILENAME;
    std::string bpe_merges = config::BPE_MERGES_FILENAME;
    std::string embed_path = config::EMBEDDING_WEIGHTS_FILENAME;

    // Search paths for BPE files
    std::vector<std::string> search_prefixes = {"", "../", "../../", "build/", "../build/", "../../build/"};

    bool bpe_loaded = false;
    for (const auto& prefix : search_prefixes) {
        std::string vp = prefix + config::BPE_VOCAB_FILENAME;
        std::string mp = prefix + config::BPE_MERGES_FILENAME;
        if (std::filesystem::exists(vp) && std::filesystem::exists(mp)) {
            bpe_loaded = bpe_.load(vp, mp);
            if (bpe_loaded) break;
        }
    }

    if (!bpe_loaded) {
        std::cerr << "[Tokenizer] BPE files not found. Creating fallback vocab.\n";
        // Minimal BPE: just single chars + special tokens
    }

    // Create embedding matrix
    size_t vocab_size = bpe_.is_loaded() ? bpe_.vocab_size() : config::BPE_VOCAB_SIZE;
    embed_ = std::make_unique<EmbeddingMatrix>(vocab_size, dimension);

    // Try loading trained embeddings
    if (std::filesystem::exists(embed_path)) {
        embed_->load(embed_path);
        std::cout << "[Tokenizer] Loaded trained embeddings from " << embed_path << "\n";
    }

    // Try loading trained word vectors
    bool trained_loaded = load_from_file(config::TRAINED_VOCAB_FILENAME);
    if (!trained_loaded) {
        for (const auto& prefix : search_prefixes) {
            std::string p = prefix + config::TRAINED_VOCAB_FILENAME;
            if (std::filesystem::exists(p)) {
                trained_loaded = load_from_file(p);
                if (trained_loaded) break;
            }
        }
    }

    // Fallback: load word list with hash-based 128D vectors (fast)
    if (!trained_loaded) {
        initialize_vocabulary();
    }

    // Sync word_to_vec from trained embeddings if available
    if (std::filesystem::exists(embed_path)) {
        sync_word_to_vec_from_embeddings();
    }

    // Pre-populate word_embed_cache from word_to_vec for fast encode()
    for (const auto& [w, v] : word_to_vec) {
        word_embed_cache_[w] = v;
    }

    if (!word_to_vec.empty()) {
        std::cout << "[Tokenizer] Ready: " << word_to_vec.size()
                  << " words, " << (bpe_.is_loaded() ? bpe_.vocab_size() : 0)
                  << " BPE tokens, " << bpe_vocab << "\n";
    }
}

bool Tokenizer::load_word_list(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string word;
    size_t count = 0;
    while (file >> word) {
        if (word.empty() || word[0] == '<') continue;
        // Use fast hash-based 128D vector for bulk loading
        math::State vec = math::State::Zero();
        unsigned long hash_val = 5381;
        for (char c : word) hash_val = ((hash_val << 5) + hash_val) + static_cast<unsigned char>(c);
        for (size_t j = 0; j < config::EMBEDDING_DIM; ++j) {
            vec[j] = std::sin(static_cast<double>(hash_val) * (j + 1) * 0.1);
        }
        word_to_vec[word] = vec;
        count++;
    }
    if (count > 0) {
        std::cout << "[INFO] Cargadas " << count << " palabras desde " << path << "\n";
        return true;
    }
    return false;
}

math::State Tokenizer::compute_word_embedding(const std::string& word) const {
    // Fast hash-based fallback (deterministic, 128D)
    math::State vec = math::State::Zero();
    unsigned long hash_val = 5381;
    for (char c : word) hash_val = ((hash_val << 5) + hash_val) + static_cast<unsigned char>(c);
    for (size_t j = 0; j < config::EMBEDDING_DIM; ++j) {
        vec[j] = std::sin(static_cast<double>(hash_val) * (j + 1) * 0.1);
    }
    return vec;
}

void Tokenizer::sync_word_to_vec_from_embeddings() {
    size_t updated = 0;
    for (auto& [word, vec] : word_to_vec) {
        auto ids = bpe_.encode(word);
        if (!ids.empty()) {
            math::State avg = math::State::Zero();
            size_t valid = 0;
            for (auto id : ids) {
                if (id >= 0 && static_cast<size_t>(id) < bpe_.vocab_size() && id != BpeTokenizer::UNK_ID && id != BpeTokenizer::PAD_ID) {
                    avg += embed_->forward(id);
                    valid++;
                }
            }
            if (valid > 0) {
                avg *= (1.0 / valid);
                vec = avg;
                updated++;
            }
        }
    }
    if (updated > 0) {
        std::cout << "[Tokenizer] Synced " << updated << " word vectors from trained embeddings\n";
    }
}

void Tokenizer::initialize_vocabulary() {
    std::vector<std::string> paths = {
        config::VOCAB_FILENAME,
        "../" + config::VOCAB_FILENAME,
        "../../" + config::VOCAB_FILENAME,
    };

    // Check in build directory too
    for (const auto& prefix : std::vector<std::string>{"", "../", "../../"}) {
        paths.push_back(prefix + "build/" + config::VOCAB_FILENAME);
    }

    for (const auto& p : paths) {
        if (load_word_list(p)) return;
    }

    std::cerr << "[ERROR] No se encontró vocabulario_es.txt. Usando fallback mínimo.\n";
    // Minimal fallback using BPE if available
    if (bpe_.is_loaded()) {
        math::State v = embed_->forward(bpe_.token_to_id("respuesta"));
        if (v.norm() > 0) word_to_vec["respuesta"] = v;
        v = embed_->forward(bpe_.token_to_id("mundo"));
        if (v.norm() > 0) word_to_vec["mundo"] = v;
        v = embed_->forward(bpe_.token_to_id("verdad"));
        if (v.norm() > 0) word_to_vec["verdad"] = v;
    }
    if (word_to_vec.empty()) {
        word_to_vec["respuesta"] = math::State::Constant(0.5);
        word_to_vec["mundo"] = math::State::Constant(0.3);
        word_to_vec["verdad"] = math::State::Constant(0.7);
    }
}

math::State Tokenizer::encode(const std::string& word) const {
    auto it = word_to_vec.find(word);
    if (it != word_to_vec.end()) {
        return it->second;
    }

    // Check BPE for subword encoding
    if (bpe_.is_loaded()) {
        auto ids = bpe_.encode(word);
        if (!ids.empty()) {
            math::State vec = math::State::Zero();
            for (auto id : ids) {
                vec += embed_->forward(id);
            }
            vec *= (1.0 / ids.size());
            return vec;
        }
    }

    // Hash fallback
    math::State vec = math::State::Zero();
    unsigned long hash_val = 5381;
    for (char c : word) hash_val = ((hash_val << 5) + hash_val) + static_cast<unsigned char>(c);
    for (size_t j = 0; j < config::EMBEDDING_DIM; ++j) {
        vec[j] = std::sin(static_cast<double>(hash_val) * (j + 1) * 0.1);
    }
    return vec;
}

size_t Tokenizer::levenshtein_distance(const std::string& s1, const std::string& s2) {
    const size_t m(s1.size());
    const size_t n(s2.size());
    if (m == 0) return n;
    if (n == 0) return m;

    std::vector<size_t> costs(n + 1);
    std::iota(costs.begin(), costs.end(), 0);

    for (size_t i = 0; i < m; ++i) {
        costs[0] = i + 1;
        size_t corner = i;
        for (size_t j = 0; j < n; ++j) {
            size_t upper = costs[j + 1];
            if (s1[i] == s2[j]) {
                costs[j + 1] = corner;
            } else {
                size_t t(upper < corner ? upper : corner);
                costs[j + 1] = (costs[j] < t ? costs[j] : t) + 1;
            }
            corner = upper;
        }
    }
    return costs[n];
}

std::string Tokenizer::find_closest_word(const std::string& word) const {
    if (word_to_vec.empty()) return word;
    if (word.size() <= 2) return word;
    std::string best_match = word_to_vec.begin()->first;
    size_t min_dist = std::numeric_limits<size_t>::max();

    for (const auto& [w, v] : word_to_vec) {
        size_t d = levenshtein_distance(word, w);
        if (d < min_dist) {
            min_dist = d;
            best_match = w;
            if (d == 0) break;
        }
    }

    size_t threshold = std::max<size_t>(1, word.size() / 3);
    if (min_dist <= threshold) {
        return best_match;
    }
    return word;
}

std::string Tokenizer::decode(const math::State& state) const {
    std::string best_word = "unknown";
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& [word, vec] : word_to_vec) {
        double dist = (state - vec).norm();
        if (dist < min_dist) {
            min_dist = dist;
            best_word = word;
        }
    }
    return best_word;
}

std::string Tokenizer::sample(const math::State& state, double temperature) const {
    if (word_to_vec.empty()) return "unknown";

    std::vector<std::string> words;
    std::vector<double> weights;
    double max_weight = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& [word, vec] : word_to_vec) {
        double d = (state - vec).norm();
        if (d < min_dist) min_dist = d;
    }

    for (const auto& [word, vec] : word_to_vec) {
        double dist = (state - vec).norm();
        double w = std::exp(-(dist - min_dist) / temperature);
        weights.push_back(w);
        words.push_back(word);
        if (w > max_weight) max_weight = w;
    }

    if (max_weight == 0) {
        return decode(state);
    }

    std::discrete_distribution<> dist_gen(weights.begin(), weights.end());
    return words[dist_gen(rng_sample)];
}

std::string Tokenizer::sample_restricted(const math::State& state,
                                          const std::vector<std::string>& candidates,
                                          const std::vector<std::string>& taboo,
                                          double temperature) const {
    if (candidates.empty()) return "unknown";

    std::vector<std::string> valid_words;
    std::vector<double> weights;
    double max_weight = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& w : candidates) {
        auto it = word_to_vec.find(w);
        if (it != word_to_vec.end()) {
            bool is_taboo = (std::find(taboo.begin(), taboo.end(), w) != taboo.end());
            double d = (state - it->second).norm();
            if (is_taboo) d += 15.0;
            if (d < min_dist) min_dist = d;
            valid_words.push_back(w);
        }
    }

    if (valid_words.empty()) {
        return candidates[0];
    }

    double polarized_temp = temperature;
    if (temperature > 0.001) {
        polarized_temp = temperature / 1.5;
    }

    for (const auto& w : valid_words) {
        auto it = word_to_vec.find(w);
        bool is_taboo = (std::find(taboo.begin(), taboo.end(), w) != taboo.end());
        double dist = (state - it->second).norm();
        if (is_taboo) dist += 15.0;
        double weight = std::exp(-(dist - min_dist) / polarized_temp);
        weights.push_back(weight);
        if (weight > max_weight) max_weight = weight;
    }

    if (max_weight == 0) {
        return valid_words[0];
    }

    std::discrete_distribution<> dist_gen(weights.begin(), weights.end());
    return valid_words[dist_gen(rng_sample)];
}

std::vector<std::string> Tokenizer::get_vocabulary() const {
    std::vector<std::string> vocab;
    for (const auto& [word, vec] : word_to_vec) {
        vocab.push_back(word);
    }
    return vocab;
}

void Tokenizer::update_vector(const std::string& word, const math::State& new_vec) {
    word_to_vec[word] = new_vec;
}

bool Tokenizer::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    for (const auto& [word, vec] : word_to_vec) {
        file << word;
        for (double v : vec) file << " " << v;
        file << "\n";
    }
    return true;
}

bool Tokenizer::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        if (word.empty()) continue;
        math::State vec;
        double val;
        int i = 0;
        while (ss >> val && static_cast<size_t>(i) < config::EMBEDDING_DIM) {
            vec(i++) = val;
        }
        if (static_cast<size_t>(i) == config::EMBEDDING_DIM) {
            word_to_vec[word] = vec;
            count++;
        } else if (i > 0) {
            std::cerr << "[WARN] Dimension mismatch for '" << word
                      << "': expected " << config::EMBEDDING_DIM << ", got " << i << ". Skipping.\n";
        }
    }
    return count > 0;
}

math::State Tokenizer::get_vector(const std::string& word) const {
    return encode(word);
}

} // namespace nlp
