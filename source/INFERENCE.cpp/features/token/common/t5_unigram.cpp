#include "features/token/common/t5_unigram.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace edcpp::inference {
namespace {

std::size_t utf8_character_length(const std::string& text, std::size_t position) {
    const auto lead = static_cast<unsigned char>(text[position]);
    std::size_t length = 1;
    if ((lead & 0xe0) == 0xc0)
        length = 2;
    else if ((lead & 0xf0) == 0xe0)
        length = 3;
    else if ((lead & 0xf8) == 0xf0)
        length = 4;

    if (position + length > text.size()) return 1;
    for (std::size_t offset = 1; offset < length; ++offset) {
        const auto continuation = static_cast<unsigned char>(text[position + offset]);
        if ((continuation & 0xc0) != 0x80) return 1;
    }
    return length;
}

bool is_single_utf8_character(const std::string& text) {
    if (text.empty()) return false;
    const auto lead = static_cast<unsigned char>(text[0]);
    std::size_t length = 0;
    if (lead <= 0x7f)
        length = 1;
    else if (lead >= 0xc2 && lead <= 0xdf)
        length = 2;
    else if (lead >= 0xe0 && lead <= 0xef)
        length = 3;
    else if (lead >= 0xf0 && lead <= 0xf4)
        length = 4;
    else
        return false;
    if (length != text.size()) return false;
    for (std::size_t index = 1; index < length; ++index) {
        const auto continuation = static_cast<unsigned char>(text[index]);
        if ((continuation & 0xc0) != 0x80) return false;
    }
    const auto second = length > 1 ? static_cast<unsigned char>(text[1]) : 0;
    if ((lead == 0xe0 && second < 0xa0) || (lead == 0xed && second >= 0xa0) ||
        (lead == 0xf0 && second < 0x90) || (lead == 0xf4 && second > 0x8f)) {
        return false;
    }
    return true;
}

bool valid_token_id(TokenId id, std::size_t piece_count) {
    return id >= 0 && static_cast<std::size_t>(id) < piece_count;
}

} // namespace

T5UnigramTokenizer::T5UnigramTokenizer(std::vector<UnigramPiece> pieces,
                                       T5UnigramConfig config)
    : pieces_(std::move(pieces)), config_(std::move(config)) {
    if (pieces_.empty()) throw std::invalid_argument("T5 Unigram vocabulary is empty");
    if (!is_single_utf8_character(config_.metaspace))
        throw std::invalid_argument("T5 metaspace must be exactly one Unicode character");
    if (!std::isfinite(config_.unknown_penalty) || config_.unknown_penalty < 0.0f)
        throw std::invalid_argument("T5 unknown penalty must be finite and non-negative");
    if (!valid_token_id(config_.unknown_token, pieces_.size()))
        throw std::invalid_argument("T5 unknown token ID is outside the vocabulary");
    if (config_.add_beginning && !valid_token_id(config_.beginning_token, pieces_.size()))
        throw std::invalid_argument("T5 beginning token ID is outside the vocabulary");
    if (config_.add_end && !valid_token_id(config_.end_token, pieces_.size()))
        throw std::invalid_argument("T5 end token ID is outside the vocabulary");
    if (config_.padding_token >= 0 &&
        !valid_token_id(config_.padding_token, pieces_.size()))
        throw std::invalid_argument("T5 padding token ID is outside the vocabulary");

    minimum_score_ = std::numeric_limits<float>::infinity();
    std::unordered_map<std::string, TokenId> ids;
    for (std::size_t index = 0; index < pieces_.size(); ++index) {
        const auto& piece = pieces_[index];
        if (piece.text.empty()) throw std::invalid_argument("T5 Unigram piece is empty");
        if (!std::isfinite(piece.score))
            throw std::invalid_argument("T5 Unigram piece score is not finite");
        if (!ids.emplace(piece.text, static_cast<TokenId>(index)).second)
            throw std::invalid_argument("T5 Unigram piece is duplicated: " + piece.text);
        minimum_score_ = std::min(minimum_score_, piece.score);
        candidates_[static_cast<unsigned char>(piece.text.front())].push_back(index);
    }

    for (auto& candidates : candidates_) {
        std::sort(candidates.begin(), candidates.end(), [&](std::size_t left,
                                                            std::size_t right) {
            if (pieces_[left].text.size() != pieces_[right].text.size())
                return pieces_[left].text.size() < pieces_[right].text.size();
            return left < right;
        });
    }

    const auto add_special = [&](TokenId id) {
        if (!valid_token_id(id, pieces_.size())) return;
        const auto& text = pieces_[static_cast<std::size_t>(id)].text;
        const auto found = std::find_if(special_tokens_.begin(), special_tokens_.end(),
                                        [&](const auto& entry) { return entry.first == text; });
        if (found == special_tokens_.end()) special_tokens_.emplace_back(text, id);
    };
    add_special(config_.padding_token);
    add_special(config_.end_token);
    add_special(config_.unknown_token);
    add_special(config_.beginning_token);
}

std::string T5UnigramTokenizer::normalize(const std::string& text) const {
    std::string normalized;
    normalized.reserve(text.size());
    bool previous_space = false;
    for (char value : text) {
        if (value == ' ') {
            if (!previous_space) normalized.push_back(value);
            previous_space = true;
        } else {
            normalized.push_back(value);
            previous_space = false;
        }
    }
    return normalized;
}

std::string T5UnigramTokenizer::pretokenize(const std::string& text,
                                            bool prepend_metaspace) const {
    std::string result;
    result.reserve(text.size() + config_.metaspace.size());
    if (prepend_metaspace) result += config_.metaspace;
    for (char value : text) {
        if (value == ' ')
            result += config_.metaspace;
        else
            result.push_back(value);
    }
    return result;
}

TokenizationResult T5UnigramTokenizer::encode_segment(const std::string& text,
                                                      bool prepend_metaspace) const {
    struct PathNode {
        float score = -std::numeric_limits<float>::infinity();
        std::size_t starts_at = std::string::npos;
        TokenId token = -1;
        bool unknown = false;
    };

    const auto input = pretokenize(text, prepend_metaspace);
    if (input.empty()) {
        TokenizationResult empty;
        empty.ok = true;
        return empty;
    }
    std::vector<PathNode> best(input.size() + 1);
    best[0].score = 0.0f;

    std::size_t starts_at = 0;
    while (starts_at < input.size()) {
        const auto character_length = utf8_character_length(input, starts_at);
        bool has_single_character_piece = false;
        const auto& candidates = candidates_[static_cast<unsigned char>(input[starts_at])];
        for (std::size_t piece_index : candidates) {
            const auto& piece = pieces_[piece_index];
            if (starts_at + piece.text.size() > input.size() ||
                input.compare(starts_at, piece.text.size(), piece.text) != 0) {
                continue;
            }

            const auto ends_at = starts_at + piece.text.size();
            auto& target = best[ends_at];
            const auto candidate_score = best[starts_at].score + piece.score;
            if (target.starts_at == std::string::npos || candidate_score > target.score) {
                target.score = candidate_score;
                target.starts_at = starts_at;
                target.token = static_cast<TokenId>(piece_index);
                target.unknown = false;
            }
            if (piece.text.size() == character_length) has_single_character_piece = true;
        }

        if (!has_single_character_piece) {
            const auto ends_at = starts_at + character_length;
            auto& target = best[ends_at];
            const auto candidate_score = best[starts_at].score + minimum_score_ -
                                         config_.unknown_penalty;
            if (target.starts_at == std::string::npos || candidate_score > target.score) {
                target.score = candidate_score;
                target.starts_at = starts_at;
                target.token = config_.unknown_token;
                target.unknown = true;
            }
        }
        starts_at += character_length;
    }

    TokenizationResult result;
    if (best.back().starts_at == std::string::npos) {
        result.message = "T5 Unigram could not cover the normalized input";
        return result;
    }

    std::size_t ends_at = input.size();
    while (ends_at != 0) {
        const auto& node = best[ends_at];
        result.tokens.push_back(node.token);
        if (node.unknown) ++result.unknown_pieces;
        ends_at = node.starts_at;
    }
    std::reverse(result.tokens.begin(), result.tokens.end());
    result.ok = true;
    return result;
}

TokenizationResult T5UnigramTokenizer::encode(const std::string& text) const {
    TokenizationResult result;
    const auto normalized = normalize(text);
    if (config_.add_beginning) result.tokens.push_back(config_.beginning_token);

    std::size_t cursor = 0;
    bool found_special = false;
    bool prepended_once = false;
    const auto encode_text = [&](const std::string& segment) {
        const bool prepend = config_.prepend_scheme == MetaspacePrependScheme::always ||
                             (config_.prepend_scheme == MetaspacePrependScheme::first &&
                              !prepended_once);
        auto encoded = encode_segment(segment, prepend);
        if (prepend) prepended_once = true;
        return encoded;
    };
    while (cursor < normalized.size()) {
        std::size_t next_position = std::string::npos;
        std::size_t next_special = 0;
        for (std::size_t index = 0; index < special_tokens_.size(); ++index) {
            const auto position = normalized.find(special_tokens_[index].first, cursor);
            if (position < next_position ||
                (position == next_position && position != std::string::npos &&
                 special_tokens_[index].first.size() >
                     special_tokens_[next_special].first.size())) {
                next_position = position;
                next_special = index;
            }
        }

        if (next_position == std::string::npos) break;
        found_special = true;
        if (next_position > cursor) {
            auto segment = encode_text(normalized.substr(cursor, next_position - cursor));
            if (!segment.ok) return segment;
            result.tokens.insert(result.tokens.end(), segment.tokens.begin(), segment.tokens.end());
            result.unknown_pieces += segment.unknown_pieces;
        }
        result.tokens.push_back(special_tokens_[next_special].second);
        cursor = next_position + special_tokens_[next_special].first.size();
    }

    if (cursor < normalized.size() || (!found_special && normalized.empty())) {
        auto segment = encode_text(normalized.substr(cursor));
        if (!segment.ok) return segment;
        result.tokens.insert(result.tokens.end(), segment.tokens.begin(), segment.tokens.end());
        result.unknown_pieces += segment.unknown_pieces;
    }

    if (config_.add_end) result.tokens.push_back(config_.end_token);
    if (config_.maximum_tokens != 0 && result.tokens.size() > config_.maximum_tokens) {
        result.tokens.resize(config_.maximum_tokens);
        if (config_.add_end && !result.tokens.empty()) result.tokens.back() = config_.end_token;
        result.truncated = true;
    }
    result.ok = true;
    return result;
}

} // namespace edcpp::inference
