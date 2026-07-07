#ifndef CALCULUS_LLM_NLP_BPE_TOKENIZER_H
#define CALCULUS_LLM_NLP_BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace nlp {

using token_id_t = int32_t;

class BpeTokenizer {
public:
    static constexpr token_id_t PAD_ID = 0;
    static constexpr token_id_t UNK_ID = 1;
    static constexpr token_id_t BOS_ID = 2;
    static constexpr token_id_t EOS_ID = 3;
    static constexpr token_id_t MASK_ID = 4;

    BpeTokenizer();

    bool load(const std::string& vocab_path, const std::string& merges_path);

    std::vector<token_id_t> encode(const std::string& text) const;
    std::string decode(const std::vector<token_id_t>& ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }

    token_id_t token_to_id(const std::string& token) const;
    std::string id_to_token(token_id_t id) const;

    bool is_loaded() const { return !id_to_token_.empty(); }

private:
    std::unordered_map<std::string, token_id_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    // pair_hash_: "left|right" → merge_rank (lower = higher priority)
    mutable std::unordered_map<std::string, size_t> pair_rank_;

    std::vector<token_id_t> encode_word(const std::string& word) const;
};

} // namespace nlp

#endif // CALCULUS_LLM_NLP_BPE_TOKENIZER_H
