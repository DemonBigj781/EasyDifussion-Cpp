#pragma once

#include "features/token/common/tokenizer.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace edcpp::inference {

struct UnigramPiece {
    std::string text{};
    float score = 0.0f;
};

enum class MetaspacePrependScheme : std::uint8_t { always, never, first };

struct T5UnigramConfig {
    TokenId unknown_token = 2;
    TokenId beginning_token = -1;
    TokenId end_token = 1;
    TokenId padding_token = 0;
    bool add_beginning = false;
    bool add_end = true;
    MetaspacePrependScheme prepend_scheme = MetaspacePrependScheme::always;
    std::string metaspace = u8"\u2581";
    float unknown_penalty = 10.0f;
    std::size_t maximum_tokens = 0;
};

class T5UnigramTokenizer final : public Tokenizer {
  public:
    explicit T5UnigramTokenizer(std::vector<UnigramPiece> pieces,
                                T5UnigramConfig config = {});

    TokenizationResult encode(const std::string& text) const override;

  private:
    TokenizationResult encode_segment(const std::string& text,
                                      bool prepend_metaspace) const;
    std::string normalize(const std::string& text) const;
    std::string pretokenize(const std::string& text,
                            bool prepend_metaspace) const;

    std::vector<UnigramPiece> pieces_{};
    std::array<std::vector<std::size_t>, 256> candidates_{};
    std::vector<std::pair<std::string, TokenId>> special_tokens_{};
    T5UnigramConfig config_{};
    float minimum_score_ = 0.0f;
};

} // namespace edcpp::inference
