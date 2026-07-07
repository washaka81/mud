#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace nlp {

BpeTokenizer::BpeTokenizer() {}

bool BpeTokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) {
        std::cerr << "[BPE] Error: cannot open " << vocab_path << "\n";
        return false;
    }

    token_to_id_.clear();
    id_to_token_.clear();

    std::string line;
    while (std::getline(vf, line)) {
        if (line.empty()) continue;
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        token_id_t id = std::stoi(line.substr(0, space));
        std::string token = line.substr(space + 1);
        token_to_id_[token] = id;
        if (static_cast<size_t>(id) >= id_to_token_.size()) {
            id_to_token_.resize(id + 1);
        }
        id_to_token_[id] = token;
    }

    if (merges_path.empty()) return true;

    std::ifstream mf(merges_path);
    if (!mf.is_open()) {
        std::cerr << "[BPE] Error: cannot open " << merges_path << "\n";
        return false;
    }

    merges_.clear();
    pair_rank_.clear();
    size_t rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        std::string left = line.substr(0, space);
        std::string right = line.substr(space + 1);
        merges_.emplace_back(left, right);
        pair_rank_[left + "|" + right] = rank++;
    }

    std::cout << "[BPE] Loaded " << id_to_token_.size() << " tokens, "
              << merges_.size() << " merges from "
              << vocab_path << " / " << merges_path << "\n";
    return true;
}

token_id_t BpeTokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return UNK_ID;
}

std::string BpeTokenizer::id_to_token(token_id_t id) const {
    if (static_cast<size_t>(id) < id_to_token_.size()) return id_to_token_[id];
    return "<UNK>";
}

std::vector<token_id_t> BpeTokenizer::encode(const std::string& text) const {
    if (!is_loaded()) return {};

    std::vector<token_id_t> result;

    // Split by whitespace
    std::istringstream iss(text);
    std::string word;
    bool first = true;
    while (iss >> word) {
        if (!first) {
            // Add space token if present in vocab
            auto space_it = token_to_id_.find(" ");
            if (space_it != token_to_id_.end()) {
                result.push_back(space_it->second);
            }
        }
        first = false;

        auto word_ids = encode_word(word);
        result.insert(result.end(), word_ids.begin(), word_ids.end());
    }

    return result;
}

std::vector<token_id_t> BpeTokenizer::encode_word(const std::string& word) const {
    // Split into individual characters
    std::vector<std::string> pieces;
    for (size_t i = 0; i < word.size(); ) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        pieces.push_back(word.substr(i, len));
        i += len;
    }

    // Direct single-char lookup
    if (pieces.size() <= 1) {
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end()) return {it->second};
        std::vector<token_id_t> ids;
        for (const auto& p : pieces) {
            auto pit = token_to_id_.find(p);
            ids.push_back(pit != token_to_id_.end() ? pit->second : UNK_ID);
        }
        return ids;
    }

    // Apply BPE merges using pair_rank_ for O(1) priority lookups
    // Track the current best pair to merge each iteration
    while (pieces.size() > 1) {
        size_t best_idx = pieces.size();
        size_t best_rank = pair_rank_.size();  // larger than any valid rank

        // Scan adjacent pairs, find highest-priority (lowest rank) merge
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string key = pieces[i] + "|" + pieces[i + 1];
            auto it = pair_rank_.find(key);
            if (it != pair_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                if (best_rank == 0) break;  // highest possible priority
            }
        }

        if (best_idx >= pieces.size()) break;  // no more merges apply

        // Apply the merge
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }

    // Convert pieces to token IDs
    std::vector<token_id_t> ids;
    ids.reserve(pieces.size());
    for (const auto& p : pieces) {
        auto it = token_to_id_.find(p);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Fallback to individual characters
            for (size_t i = 0; i < p.size(); ) {
                unsigned char c = static_cast<unsigned char>(p[i]);
                size_t len = 1;
                if ((c & 0x80) == 0) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                std::string ch = p.substr(i, len);
                auto cit = token_to_id_.find(ch);
                ids.push_back(cit != token_to_id_.end() ? cit->second : UNK_ID);
                i += len;
            }
        }
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<token_id_t>& ids) const {
    std::string result;
    for (size_t i = 0; i < ids.size(); ++i) {
        std::string token = id_to_token(ids[i]);
        if (token == "<PAD>" || token == "<UNK>" || token == "<BOS>" ||
            token == "<EOS>" || token == "<MASK>") {
            continue;
        }
        if (!result.empty() && token != " " &&
            result.back() != ' ' && token[0] != ' ') {
            result += ' ';
        }
        result += token;
    }
    return result;
}

} // namespace nlp
