#include "features/conditioning/common/token_sequence.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace {

class LetterTokenizer final : public edcpp::inference::Tokenizer {
  public:
    edcpp::inference::TokenizationResult encode(
        const std::string& text) const override {
        edcpp::inference::TokenizationResult result;
        result.tokens.push_back(100);
        for (const char value : text) {
            if (value >= 'a' && value <= 'z') {
                result.tokens.push_back(value - 'a' + 1);
            } else if (value != ' ') {
                result.tokens.push_back(999);
                ++result.unknown_pieces;
            }
        }
        result.tokens.push_back(101);
        result.ok = true;
        return result;
    }
};

edcpp::inference::TokenizerSpecialTokens special_tokens() {
    edcpp::inference::TokenizerSpecialTokens result;
    result.unknown = 999;
    result.beginning = 100;
    result.end = 101;
    result.padding = 0;
    result.adds_beginning = true;
    result.adds_end = true;
    return result;
}

bool near(float left, float right) {
    return std::fabs(left - right) < 0.0001f;
}

} // namespace

int main() {
    using namespace edcpp::inference;

    LetterTokenizer tokenizer;
    PromptSequenceConfig padded_config;
    padded_config.minimum_length = 8;
    const auto padded = prepare_prompt_sequence(
        "(ab) c", tokenizer, special_tokens(), padded_config);
    assert(padded.ok && !padded.truncated && padded.chunk_count == 1);
    assert((padded.tokens ==
            std::vector<TokenId>{100, 1, 2, 3, 101, 0, 0, 0}));
    assert(padded.weights.size() == padded.tokens.size());
    assert(padded.fusion_groups.size() == padded.tokens.size());
    assert(padded.branch_count == 1);
    assert((padded.branch_offsets == std::vector<std::size_t>{0}));
    assert(near(padded.weights[0], 1.0f));
    assert(near(padded.weights[1], 1.1f));
    assert(near(padded.weights[2], 1.1f));
    assert(near(padded.weights[3], 1.0f));
    assert((padded.attention_mask ==
            std::vector<float>{1, 1, 1, 1, 1, 0, 0, 0}));

    PromptSequenceConfig left_config;
    left_config.minimum_length = 5;
    left_config.pad_left = true;
    const auto left =
        prepare_prompt_sequence("a", tokenizer, special_tokens(), left_config);
    assert(left.ok);
    assert((left.tokens == std::vector<TokenId>{0, 0, 100, 1, 101}));
    assert((left.attention_mask == std::vector<float>{0, 0, 1, 1, 1}));

    PromptSequenceConfig truncate_config;
    truncate_config.maximum_length = 4;
    const auto truncated = prepare_prompt_sequence(
        "abcdef", tokenizer, special_tokens(), truncate_config);
    assert(truncated.ok && truncated.truncated && truncated.chunk_count == 1);
    assert((truncated.tokens == std::vector<TokenId>{100, 1, 2, 101}));

    PromptSequenceConfig chunk_config;
    chunk_config.minimum_length = 5;
    chunk_config.maximum_length = 5;
    chunk_config.allow_overflow_chunks = true;
    const auto chunked = prepare_prompt_sequence(
        "abcdef", tokenizer, special_tokens(), chunk_config);
    assert(chunked.ok && !chunked.truncated && chunked.chunk_count == 2);
    assert((chunked.tokens ==
            std::vector<TokenId>{100, 1, 2, 3, 101,
                                 100, 4, 5, 6, 101}));

    const auto separated = prepare_prompt_sequence(
        "ab BREAK cd", tokenizer, special_tokens(), chunk_config);
    assert(separated.ok && separated.chunk_count == 2);
    assert((separated.tokens ==
            std::vector<TokenId>{100, 1, 2, 101, 0,
                                 100, 3, 4, 101, 0}));
    assert((separated.attention_mask ==
            std::vector<float>{1, 1, 1, 1, 0,
                               1, 1, 1, 1, 0}));

    PromptSequenceConfig tag_config;
    tag_config.prompt_mode = PromptMode::tags;
    const auto fused = prepare_prompt_sequence(
        "red:fur AND green:eyes", tokenizer, special_tokens(), tag_config);
    assert(fused.ok && fused.branch_count == 2 && fused.chunk_count == 2);
    assert((fused.branch_offsets == std::vector<std::size_t>{0, 8}));
    assert(fused.tokens.size() == fused.weights.size());
    assert(fused.tokens.size() == fused.fusion_groups.size());
    for (std::size_t index = 1; index < 7; ++index)
        assert(fused.fusion_groups[index] == 1);
    for (std::size_t index = 9; index < 18; ++index)
        assert(fused.fusion_groups[index] == 2);

    const auto bare_weight = prepare_prompt_sequence(
        "xyz:1.2", tokenizer, special_tokens(), tag_config);
    assert(bare_weight.ok);
    assert(near(bare_weight.weights[1], 1.2f));
    assert(near(bare_weight.weights[2], 1.2f));
    assert(near(bare_weight.weights[3], 1.2f));

    auto no_break_chunks = chunk_config;
    no_break_chunks.allow_overflow_chunks = false;
    const auto rejected_break = prepare_prompt_sequence(
        "ab BREAK cd", tokenizer, special_tokens(), no_break_chunks);
    assert(!rejected_break.ok);

    PromptSequenceConfig additive_config;
    additive_config.minimum_length = 5;
    additive_config.mask_format = AttentionMaskFormat::additive;
    const auto additive = prepare_prompt_sequence(
        "a", tokenizer, special_tokens(), additive_config);
    assert(additive.ok);
    assert(near(additive.attention_mask[0], 0.0f));
    assert(near(additive.attention_mask[1], 0.0f));
    assert(near(additive.attention_mask[2], 0.0f));
    assert(std::isinf(additive.attention_mask[3]) &&
           additive.attention_mask[3] < 0.0f);
    assert(std::isinf(additive.attention_mask[4]) &&
           additive.attention_mask[4] < 0.0f);

    const auto unknown = prepare_prompt_sequence(
        "!", tokenizer, special_tokens(), PromptSequenceConfig{});
    assert(unknown.ok && unknown.unknown_pieces == 1);
    assert((unknown.tokens == std::vector<TokenId>{100, 999, 101}));

    auto missing_padding = special_tokens();
    missing_padding.padding = -1;
    const auto rejected_padding = prepare_prompt_sequence(
        "a", tokenizer, missing_padding, padded_config);
    assert(!rejected_padding.ok);

    PromptSequenceConfig invalid_lengths;
    invalid_lengths.minimum_length = 6;
    invalid_lengths.maximum_length = 5;
    const auto rejected_lengths = prepare_prompt_sequence(
        "a", tokenizer, special_tokens(), invalid_lengths);
    assert(!rejected_lengths.ok);

    auto wrong_beginning = special_tokens();
    wrong_beginning.beginning = 200;
    const auto rejected_contract = prepare_prompt_sequence(
        "a", tokenizer, wrong_beginning, PromptSequenceConfig{});
    assert(!rejected_contract.ok);
}
