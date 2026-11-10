#pragma once

#include "features/conditioning/common/prompt_mode.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace edcpp::inference {

enum class PromptSegmentKind : std::uint8_t {
    text,
    break_marker,
    conjunction_marker,
};

struct PromptSegment {
    PromptSegmentKind kind = PromptSegmentKind::text;
    std::string text{};
    float weight = 1.0f;
    std::uint32_t fusion_group = 0;
};

struct PromptAttentionParseResult {
    bool ok = false;
    std::string message{};
    std::vector<PromptSegment> segments{};
};

PromptAttentionParseResult parse_prompt_attention(
    std::string_view prompt,
    PromptMode mode = PromptMode::natural_language);

} // namespace edcpp::inference
