#pragma once

#include "features/conditioning/common/prompt_mode.hpp"
#include "features/token/common/tokenizer_resource.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace edcpp::inference {

enum class AttentionMaskFormat : std::uint8_t { binary, additive };

struct PromptSequenceConfig {
    std::size_t minimum_length = 0;
    std::size_t maximum_length = 0;
    bool allow_overflow_chunks = false;
    bool pad_left = false;
    AttentionMaskFormat mask_format = AttentionMaskFormat::binary;
    PromptMode prompt_mode = PromptMode::natural_language;
};

struct PromptSequenceResult {
    bool ok = false;
    std::string message{};
    std::vector<TokenId> tokens{};
    std::vector<float> weights{};
    std::vector<float> attention_mask{};
    std::vector<std::uint32_t> fusion_groups{};
    std::vector<std::size_t> branch_offsets{};
    std::size_t unknown_pieces = 0;
    std::size_t chunk_count = 0;
    std::size_t branch_count = 0;
    bool truncated = false;
};

PromptSequenceResult prepare_prompt_sequence(
    std::string_view prompt, const Tokenizer& tokenizer,
    const TokenizerSpecialTokens& special_tokens,
    const PromptSequenceConfig& config = {});

} // namespace edcpp::inference
