#include "features/conditioning/common/token_sequence.hpp"

#include "features/conditioning/common/prompt_attention.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace edcpp::inference {
namespace {

struct WeightedPayload {
    std::vector<TokenId> tokens{};
    std::vector<float> weights{};
    std::vector<std::uint32_t> fusion_groups{};
};

struct BranchPayload {
    std::vector<WeightedPayload> groups{1};
    bool saw_break = false;
};

float active_mask_value(AttentionMaskFormat format) {
    return format == AttentionMaskFormat::binary ? 1.0f : 0.0f;
}

float padding_mask_value(AttentionMaskFormat format) {
    return format == AttentionMaskFormat::binary
               ? 0.0f
               : -std::numeric_limits<float>::infinity();
}

} // namespace

PromptSequenceResult prepare_prompt_sequence(
    std::string_view prompt, const Tokenizer& tokenizer,
    const TokenizerSpecialTokens& special_tokens,
    const PromptSequenceConfig& config) {
    PromptSequenceResult result;
    const std::size_t special_count =
        (special_tokens.adds_beginning ? 1U : 0U) +
        (special_tokens.adds_end ? 1U : 0U);

    if ((special_tokens.adds_beginning && special_tokens.beginning < 0) ||
        (special_tokens.adds_end && special_tokens.end < 0)) {
        result.message = "tokenizer special-token policy is incomplete";
        return result;
    }
    if (config.maximum_length != 0 &&
        config.maximum_length < special_count) {
        result.message = "maximum sequence length cannot hold required special tokens";
        return result;
    }
    if (config.maximum_length != 0 &&
        config.minimum_length > config.maximum_length) {
        result.message = "minimum sequence length exceeds maximum sequence length";
        return result;
    }

    const auto parsed = parse_prompt_attention(prompt, config.prompt_mode);
    if (!parsed.ok) {
        result.message = parsed.message;
        return result;
    }

    std::vector<BranchPayload> branches(1);
    for (std::size_t segment_index = 0;
         segment_index < parsed.segments.size(); ++segment_index) {
        const auto& segment = parsed.segments[segment_index];
        if (segment.kind == PromptSegmentKind::break_marker) {
            auto& branch = branches.back();
            if (!branch.groups.back().tokens.empty()) {
                branch.groups.emplace_back();
                branch.saw_break = true;
            }
            continue;
        }
        if (segment.kind == PromptSegmentKind::conjunction_marker) {
            auto& branch = branches.back();
            if (branch.groups.size() > 1 && branch.groups.back().tokens.empty())
                branch.groups.pop_back();
            bool has_tokens = false;
            for (const auto& group : branch.groups)
                has_tokens = has_tokens || !group.tokens.empty();
            if (has_tokens) branches.emplace_back();
            continue;
        }

        const auto encoded = tokenizer.encode(segment.text);
        if (!encoded.ok) {
            result.message = "prompt segment " + std::to_string(segment_index) +
                             " failed tokenization: " + encoded.message;
            return result;
        }
        result.unknown_pieces += encoded.unknown_pieces;
        result.truncated = result.truncated || encoded.truncated;

        std::size_t begin = 0;
        std::size_t end = encoded.tokens.size();
        if (special_tokens.adds_beginning) {
            if (begin == end ||
                encoded.tokens[begin] != special_tokens.beginning) {
                result.message =
                    "tokenizer did not emit its declared beginning token";
                return result;
            }
            ++begin;
        }
        if (special_tokens.adds_end) {
            if (begin == end || encoded.tokens[end - 1] != special_tokens.end) {
                result.message = "tokenizer did not emit its declared end token";
                return result;
            }
            --end;
        }

        auto& group = branches.back().groups.back();
        group.tokens.insert(group.tokens.end(),
                            encoded.tokens.begin() +
                                static_cast<std::ptrdiff_t>(begin),
                            encoded.tokens.begin() +
                                static_cast<std::ptrdiff_t>(end));
        group.weights.insert(group.weights.end(), end - begin, segment.weight);
        group.fusion_groups.insert(group.fusion_groups.end(), end - begin,
                                   segment.fusion_group);
    }
    for (auto& branch : branches) {
        if (branch.groups.size() > 1 && branch.groups.back().tokens.empty())
            branch.groups.pop_back();
    }
    if (branches.size() > 1 && branches.back().groups.size() == 1 &&
        branches.back().groups.front().tokens.empty())
        branches.pop_back();

    auto append_chunk = [&](const WeightedPayload& payload, std::size_t begin,
                            std::size_t count, std::size_t target_length) {
        if (count > std::numeric_limits<std::size_t>::max() - special_count) {
            result.message = "prompt sequence length overflows host size limits";
            return false;
        }
        const auto base_length = count + special_count;
        const auto final_length = std::max(base_length, target_length);
        const auto padding_count = final_length - base_length;
        if (padding_count != 0 && special_tokens.padding < 0) {
            result.message = "padding was requested without a padding token";
            return false;
        }

        std::vector<TokenId> chunk_tokens;
        std::vector<float> chunk_weights;
        std::vector<float> chunk_mask;
        std::vector<std::uint32_t> chunk_fusion_groups;
        chunk_tokens.reserve(final_length);
        chunk_weights.reserve(final_length);
        chunk_mask.reserve(final_length);
        chunk_fusion_groups.reserve(final_length);

        const auto append_padding = [&]() {
            chunk_tokens.push_back(special_tokens.padding);
            chunk_weights.push_back(1.0f);
            chunk_mask.push_back(padding_mask_value(config.mask_format));
            chunk_fusion_groups.push_back(0);
        };
        for (std::size_t index = 0;
             config.pad_left && index < padding_count; ++index)
            append_padding();

        if (special_tokens.adds_beginning) {
            chunk_tokens.push_back(special_tokens.beginning);
            chunk_weights.push_back(1.0f);
            chunk_mask.push_back(active_mask_value(config.mask_format));
            chunk_fusion_groups.push_back(0);
        }
        for (std::size_t index = 0; index < count; ++index) {
            chunk_tokens.push_back(payload.tokens[begin + index]);
            chunk_weights.push_back(payload.weights[begin + index]);
            chunk_mask.push_back(active_mask_value(config.mask_format));
            chunk_fusion_groups.push_back(payload.fusion_groups[begin + index]);
        }
        if (special_tokens.adds_end) {
            chunk_tokens.push_back(special_tokens.end);
            chunk_weights.push_back(1.0f);
            chunk_mask.push_back(active_mask_value(config.mask_format));
            chunk_fusion_groups.push_back(0);
        }

        for (std::size_t index = 0;
             !config.pad_left && index < padding_count; ++index)
            append_padding();

        result.tokens.insert(result.tokens.end(), chunk_tokens.begin(),
                             chunk_tokens.end());
        result.weights.insert(result.weights.end(), chunk_weights.begin(),
                              chunk_weights.end());
        result.attention_mask.insert(result.attention_mask.end(),
                                     chunk_mask.begin(), chunk_mask.end());
        result.fusion_groups.insert(result.fusion_groups.end(),
                                    chunk_fusion_groups.begin(),
                                    chunk_fusion_groups.end());
        ++result.chunk_count;
        return true;
    };

    auto append_fixed_chunks = [&](const WeightedPayload& payload,
                                   std::size_t target_length) {
        if (target_length <= special_count) {
            result.message =
                "chunk length must leave room for at least one prompt token";
            return false;
        }
        const auto payload_capacity = target_length - special_count;
        std::size_t offset = 0;
        do {
            const auto count =
                std::min(payload_capacity, payload.tokens.size() - offset);
            if (!append_chunk(payload, offset, count, target_length))
                return false;
            offset += count;
        } while (offset < payload.tokens.size());
        return true;
    };

    for (const auto& branch : branches) {
        result.branch_offsets.push_back(result.tokens.size());
        ++result.branch_count;
        if (branch.saw_break) {
            if (!config.allow_overflow_chunks) {
                result.message = "BREAK requires overflow chunking";
                return result;
            }
            if (config.minimum_length <= special_count) {
                result.message =
                    "BREAK requires a fixed minimum sequence length";
                return result;
            }
            for (const auto& group : branch.groups) {
                if (!append_fixed_chunks(group, config.minimum_length))
                    return result;
            }
            continue;
        }

        const auto& payload = branch.groups.front();
        if (payload.tokens.size() >
            std::numeric_limits<std::size_t>::max() - special_count) {
            result.message = "prompt sequence length overflows host size limits";
            return result;
        }
        const auto unpadded_length = payload.tokens.size() + special_count;
        const bool exceeds_maximum =
            config.maximum_length != 0 &&
            unpadded_length > config.maximum_length;
        if (!exceeds_maximum) {
            if (!append_chunk(payload, 0, payload.tokens.size(),
                              config.minimum_length))
                return result;
        } else {
            if (config.allow_overflow_chunks) {
                if (!append_fixed_chunks(payload, config.minimum_length))
                    return result;
            } else {
                const auto payload_capacity =
                    config.maximum_length - special_count;
                if (!append_chunk(payload, 0, payload_capacity,
                                  config.maximum_length))
                    return result;
                result.truncated = true;
            }
        }
    }

    result.ok = true;
    return result;
}

} // namespace edcpp::inference
