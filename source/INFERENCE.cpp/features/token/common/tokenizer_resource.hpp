#pragma once

#include "features/token/common/tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace edcpp::inference {

enum class TokenizerResourceFormat : std::uint8_t {
    clip_merges,
    hugging_face_unigram_json,
};

struct TokenizerResourceView {
    TokenizerResourceFormat format = TokenizerResourceFormat::clip_merges;
    std::string_view identifier{};
    std::string_view serialized{};
    std::size_t maximum_tokens = 0;
    TokenId padding_token = -1;
};

struct TokenizerSpecialTokens {
    TokenId unknown = -1;
    TokenId beginning = -1;
    TokenId end = -1;
    TokenId padding = -1;
    bool adds_beginning = false;
    bool adds_end = false;
};

struct TokenizerResourceLoadResult {
    bool ok = false;
    std::string message{};
    std::unique_ptr<Tokenizer> tokenizer{};
    TokenizerSpecialTokens special_tokens{};
};

TokenizerResourceLoadResult load_tokenizer_resource(const TokenizerResourceView& resource);

} // namespace edcpp::inference
