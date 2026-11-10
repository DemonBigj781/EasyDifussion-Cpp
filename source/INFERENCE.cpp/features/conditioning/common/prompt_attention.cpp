#include "features/conditioning/common/prompt_attention.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace edcpp::inference {
namespace {

constexpr float round_multiplier = 1.1f;
constexpr float square_multiplier = 1.0f / round_multiplier;

bool is_ascii_word(char value) {
    return (value >= 'a' && value <= 'z') ||
           (value >= 'A' && value <= 'Z') ||
           (value >= '0' && value <= '9') || value == '_';
}

bool is_ascii_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\f' || value == '\v';
}

bool is_marker_at(std::string_view prompt, std::size_t position,
                  std::string_view marker) {
    if (prompt.substr(position, marker.size()) != marker) return false;
    const bool left_boundary = position == 0 || !is_ascii_word(prompt[position - 1]);
    const auto after = position + marker.size();
    const bool right_boundary = after == prompt.size() || !is_ascii_word(prompt[after]);
    return left_boundary && right_boundary;
}

enum class WeightStatus : std::uint8_t { not_a_weight, valid, invalid };

struct WeightCandidate {
    WeightStatus status = WeightStatus::not_a_weight;
    float value = 1.0f;
    std::size_t next_position = 0;
};

WeightCandidate scan_weight(std::string_view prompt, std::size_t colon_position) {
    const auto close = prompt.find(')', colon_position + 1);
    if (close == std::string_view::npos) return {};
    auto number = prompt.substr(colon_position + 1, close - colon_position - 1);
    if (number.empty()) return {};

    std::size_t position = 0;
    if (number[position] == '+' || number[position] == '-') ++position;
    bool has_digit = false;
    while (position < number.size() && number[position] >= '0' &&
           number[position] <= '9') {
        has_digit = true;
        ++position;
    }
    if (position < number.size() && number[position] == '.') {
        ++position;
        while (position < number.size() && number[position] >= '0' &&
               number[position] <= '9') {
            has_digit = true;
            ++position;
        }
    }
    if (!has_digit || position != number.size()) return {};

    if (number.front() == '+') number.remove_prefix(1);
    float value = 0.0f;
    const auto converted =
        std::from_chars(number.data(), number.data() + number.size(), value);
    if (converted.ec != std::errc{} ||
        converted.ptr != number.data() + number.size() || !std::isfinite(value)) {
        return {WeightStatus::invalid, 1.0f, close + 1};
    }
    return {WeightStatus::valid, value, close + 1};
}

bool multiply_range(std::vector<PromptSegment>& segments, std::size_t begin,
                    float multiplier) {
    for (std::size_t index = begin; index < segments.size(); ++index) {
        if (segments[index].kind != PromptSegmentKind::text) continue;
        segments[index].weight *= multiplier;
        if (!std::isfinite(segments[index].weight)) return false;
    }
    return true;
}

void trim_space_before_marker(std::vector<PromptSegment>& segments) {
    while (!segments.empty() &&
           segments.back().kind == PromptSegmentKind::text) {
        auto& text = segments.back().text;
        while (!text.empty() && is_ascii_space(text.back())) text.pop_back();
        if (!text.empty()) return;
        segments.pop_back();
    }
}

WeightCandidate parse_tag_weight(std::string_view number) {
    if (number.empty()) return {};
    std::size_t position = 0;
    if (number[position] == '+' || number[position] == '-') ++position;
    bool has_digit = false;
    while (position < number.size() && number[position] >= '0' &&
           number[position] <= '9') {
        has_digit = true;
        ++position;
    }
    if (position < number.size() && number[position] == '.') {
        ++position;
        while (position < number.size() && number[position] >= '0' &&
               number[position] <= '9') {
            has_digit = true;
            ++position;
        }
    }
    if (!has_digit || position != number.size()) return {};

    if (number.front() == '+') number.remove_prefix(1);
    float value = 0.0f;
    const auto converted =
        std::from_chars(number.data(), number.data() + number.size(), value);
    if (converted.ec != std::errc{} ||
        converted.ptr != number.data() + number.size() || !std::isfinite(value))
        return {WeightStatus::invalid, 1.0f, 0};
    return {WeightStatus::valid, value, 0};
}

bool has_weightable_name(std::string_view text) {
    for (const auto value : text) {
        if ((value >= 'a' && value <= 'z') ||
            (value >= 'A' && value <= 'Z') || value == '_')
            return true;
    }
    return false;
}

bool normalize_segments(PromptAttentionParseResult& result,
                        bool fuse_relations) {
    std::vector<PromptSegment> normalized;
    std::uint32_t next_fusion_group = 1;

    auto append = [&](std::string text, float weight,
                      std::uint32_t fusion_group) {
        if (text.empty()) return;
        if (!normalized.empty() &&
            normalized.back().kind == PromptSegmentKind::text &&
            normalized.back().weight == weight &&
            normalized.back().fusion_group == fusion_group) {
            normalized.back().text += text;
            return;
        }
        normalized.push_back({PromptSegmentKind::text, std::move(text), weight,
                              fusion_group});
    };

    for (auto& segment : result.segments) {
        if (segment.kind != PromptSegmentKind::text) {
            normalized.push_back(std::move(segment));
            continue;
        }

        std::size_t position = 0;
        while (position < segment.text.size()) {
            if (is_ascii_space(segment.text[position]) ||
                segment.text[position] == ',' || segment.text[position] == ';') {
                const auto begin = position;
                while (position < segment.text.size() &&
                       (is_ascii_space(segment.text[position]) ||
                        segment.text[position] == ',' ||
                        segment.text[position] == ';'))
                    ++position;
                append(segment.text.substr(begin, position - begin),
                       segment.weight, 0);
                continue;
            }

            const auto begin = position;
            while (position < segment.text.size() &&
                   !is_ascii_space(segment.text[position]) &&
                   segment.text[position] != ',' &&
                   segment.text[position] != ';')
                ++position;
            auto atom = segment.text.substr(begin, position - begin);
            float weight = segment.weight;

            std::size_t plus_count = 0;
            while (plus_count < atom.size() &&
                   atom[atom.size() - plus_count - 1] == '+')
                ++plus_count;
            if (plus_count != 0 && plus_count != atom.size() &&
                has_weightable_name(std::string_view(atom).substr(
                    0, atom.size() - plus_count))) {
                atom.erase(atom.size() - plus_count);
                weight *= std::pow(round_multiplier,
                                   static_cast<float>(plus_count));
                if (!std::isfinite(weight)) {
                    result.message =
                        "combined tag emphasis is outside the finite float range";
                    return false;
                }
            } else {
                const auto colon = atom.rfind(':');
                if (colon != std::string::npos && colon + 1 < atom.size()) {
                    const auto candidate = parse_tag_weight(
                        std::string_view(atom).substr(colon + 1));
                    if (candidate.status == WeightStatus::invalid) {
                        result.message =
                            "tag weight is outside the finite float range";
                        return false;
                    }
                    if (candidate.status == WeightStatus::valid &&
                        has_weightable_name(
                            std::string_view(atom).substr(0, colon))) {
                        atom.erase(colon);
                        weight *= candidate.value;
                        if (!std::isfinite(weight)) {
                            result.message =
                                "combined tag weight is outside the finite float range";
                            return false;
                        }
                    }
                }
            }

            std::uint32_t fusion_group = 0;
            if (fuse_relations && atom.find(':') != std::string::npos &&
                atom.front() != ':' && atom.back() != ':' &&
                atom.find("::") == std::string::npos) {
                for (auto& value : atom) {
                    if (value == ':') value = ' ';
                }
                fusion_group = next_fusion_group++;
            }
            append(std::move(atom), weight, fusion_group);
        }
    }

    if (normalized.empty())
        normalized.push_back({PromptSegmentKind::text, {}, 1.0f, 0});
    result.segments = std::move(normalized);
    return true;
}

} // namespace

PromptAttentionParseResult parse_prompt_attention(std::string_view prompt,
                                                  PromptMode mode) {
    PromptAttentionParseResult result;
    std::vector<std::size_t> round_openings;
    std::vector<std::size_t> square_openings;
    std::string buffered;

    auto flush = [&]() {
        if (buffered.empty()) return;
        result.segments.push_back(
            {PromptSegmentKind::text, std::move(buffered), 1.0f, 0});
        buffered.clear();
    };

    std::size_t position = 0;
    while (position < prompt.size()) {
        const char current = prompt[position];

        if (current == '\\' && position + 1 < prompt.size()) {
            const char escaped = prompt[position + 1];
            if (escaped == '(' || escaped == ')' || escaped == '[' ||
                escaped == ']' || escaped == '\\') {
                buffered.push_back(escaped);
                position += 2;
                continue;
            }
        }

        if (is_marker_at(prompt, position, "BREAK")) {
            flush();
            trim_space_before_marker(result.segments);
            result.segments.push_back(
                {PromptSegmentKind::break_marker, "BREAK", 1.0f, 0});
            position += 5;
            while (position < prompt.size() && is_ascii_space(prompt[position]))
                ++position;
            continue;
        }
        if (mode == PromptMode::tags && is_marker_at(prompt, position, "AND")) {
            flush();
            trim_space_before_marker(result.segments);
            result.segments.push_back(
                {PromptSegmentKind::conjunction_marker, "AND", 1.0f, 0});
            position += 3;
            while (position < prompt.size() && is_ascii_space(prompt[position]))
                ++position;
            continue;
        }

        if (current == '(') {
            flush();
            round_openings.push_back(result.segments.size());
            ++position;
            continue;
        }
        if (current == '[') {
            flush();
            square_openings.push_back(result.segments.size());
            ++position;
            continue;
        }
        if (current == ':' && !round_openings.empty()) {
            const auto candidate = scan_weight(prompt, position);
            if (candidate.status == WeightStatus::invalid) {
                result.message = "explicit prompt weight is outside the finite float range";
                return result;
            }
            if (candidate.status == WeightStatus::valid) {
                flush();
                if (!multiply_range(result.segments, round_openings.back(),
                                    candidate.value)) {
                    result.message = "combined prompt weight is outside the finite float range";
                    return result;
                }
                round_openings.pop_back();
                position = candidate.next_position;
                continue;
            }
        }
        if (current == ')' && !round_openings.empty()) {
            flush();
            if (!multiply_range(result.segments, round_openings.back(),
                                round_multiplier)) {
                result.message = "combined prompt weight is outside the finite float range";
                return result;
            }
            round_openings.pop_back();
            ++position;
            continue;
        }
        if (current == ']' && !square_openings.empty()) {
            flush();
            if (!multiply_range(result.segments, square_openings.back(),
                                square_multiplier)) {
                result.message = "combined prompt weight is outside the finite float range";
                return result;
            }
            square_openings.pop_back();
            ++position;
            continue;
        }

        buffered.push_back(current);
        ++position;
    }
    flush();

    for (const auto opening : round_openings) {
        if (!multiply_range(result.segments, opening, round_multiplier)) {
            result.message = "combined prompt weight is outside the finite float range";
            return result;
        }
    }
    for (const auto opening : square_openings) {
        if (!multiply_range(result.segments, opening, square_multiplier)) {
            result.message = "combined prompt weight is outside the finite float range";
            return result;
        }
    }

    std::vector<PromptSegment> merged;
    merged.reserve(result.segments.size());
    for (auto& segment : result.segments) {
        if (segment.kind == PromptSegmentKind::text && segment.text.empty())
            continue;
        if (segment.kind == PromptSegmentKind::text && !merged.empty() &&
            merged.back().kind == PromptSegmentKind::text &&
            merged.back().weight == segment.weight) {
            merged.back().text += segment.text;
        } else {
            merged.push_back(std::move(segment));
        }
    }
    if (merged.empty())
        merged.push_back({PromptSegmentKind::text, {}, 1.0f, 0});

    result.segments = std::move(merged);
    if (!normalize_segments(result, mode == PromptMode::tags))
        return result;
    result.ok = true;
    return result;
}

} // namespace edcpp::inference
