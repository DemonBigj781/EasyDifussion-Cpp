#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace edcpp::inference {

using TokenId = std::int32_t;

struct TokenizerConfig {
    TokenId unknown_token = -1;
    TokenId beginning_token = -1;
    TokenId end_token = -1;
    TokenId padding_token = -1;
    bool add_beginning = true;
    bool add_end = true;
    bool lowercase_ascii = true;
    bool use_end_of_word_suffix = true;
    std::size_t maximum_tokens = 0;
};

struct TokenizationResult {
    bool ok = false;
    std::string message{};
    std::vector<TokenId> tokens{};
    std::size_t unknown_pieces = 0;
    bool truncated = false;
};

class Tokenizer {
  public:
    virtual ~Tokenizer() = default;
    virtual TokenizationResult encode(const std::string& text) const = 0;
};

class BpeTokenizer final : public Tokenizer {
  public:
    using Vocabulary = std::unordered_map<std::string, TokenId>;
    using Merge = std::pair<std::string, std::string>;

    BpeTokenizer(Vocabulary vocabulary, std::vector<Merge> merges,
                 TokenizerConfig config = {});

    TokenizationResult encode(const std::string& text) const override;

  private:
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> merge_word(const std::string& word) const;

    Vocabulary vocabulary_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    TokenizerConfig config_;
};

} // namespace edcpp::inference
