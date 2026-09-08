#include "features/token/common/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <limits>

namespace edcpp::inference {
namespace {

std::string pair_key(const std::string& left, const std::string& right) {
    return left + '\0' + right;
}

bool is_ascii_word(unsigned char value) {
    return value < 128 && (std::isalnum(value) != 0 || value == '_');
}

} // namespace

BpeTokenizer::BpeTokenizer(Vocabulary vocabulary, std::vector<Merge> merges,
                           TokenizerConfig config)
    : vocabulary_(std::move(vocabulary)), config_(config) {
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        merge_ranks_.emplace(pair_key(merges[rank].first, merges[rank].second), rank);
    }
}

std::vector<std::string> BpeTokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> words;
    std::string current;
    bool current_is_word = false;

    const auto flush = [&]() {
        if (!current.empty()) words.push_back(std::move(current));
        current.clear();
    };

    for (unsigned char byte : text) {
        if (byte < 128 && std::isspace(byte) != 0) {
            flush();
            continue;
        }
        const bool word = is_ascii_word(byte) || byte >= 128;
        if (!current.empty() && word != current_is_word) flush();
        current_is_word = word;
        char value = static_cast<char>(byte);
        if (config_.lowercase_ascii && byte < 128)
            value = static_cast<char>(std::tolower(byte));
        current.push_back(value);
    }
    flush();
    return words;
}

std::vector<std::string> BpeTokenizer::merge_word(const std::string& word) const {
    std::vector<std::string> pieces;
    pieces.reserve(word.size());
    for (unsigned char byte : word) pieces.emplace_back(1, static_cast<char>(byte));
    if (pieces.empty()) return pieces;
    if (config_.use_end_of_word_suffix) pieces.back() += "</w>";

    while (pieces.size() > 1) {
        std::size_t best_position = 0;
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto found = merge_ranks_.find(pair_key(pieces[i], pieces[i + 1]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_position = i;
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) break;
        pieces[best_position] += pieces[best_position + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_position + 1));
    }
    return pieces;
}

TokenizationResult BpeTokenizer::encode(const std::string& text) const {
    TokenizationResult result;
    if (vocabulary_.empty()) {
        result.message = "tokenizer vocabulary is empty";
        return result;
    }

    const auto append = [&](TokenId id) {
        if (config_.maximum_tokens != 0 && result.tokens.size() >= config_.maximum_tokens) {
            result.truncated = true;
            return false;
        }
        result.tokens.push_back(id);
        return true;
    };

    if (config_.add_beginning && config_.beginning_token >= 0)
        append(config_.beginning_token);

    bool stop = result.truncated;
    for (const auto& word : pretokenize(text)) {
        if (stop) break;
        for (const auto& piece : merge_word(word)) {
            const auto found = vocabulary_.find(piece);
            if (found != vocabulary_.end()) {
                stop = !append(found->second);
            } else if (config_.unknown_token >= 0) {
                ++result.unknown_pieces;
                stop = !append(config_.unknown_token);
            } else {
                result.message = "tokenizer encountered a piece absent from the vocabulary";
                return result;
            }
            if (stop) break;
        }
    }

    if (config_.add_end && config_.end_token >= 0) {
        if (config_.maximum_tokens != 0 && result.tokens.size() >= config_.maximum_tokens) {
            result.tokens.back() = config_.end_token;
            result.truncated = true;
        } else {
            append(config_.end_token);
        }
    }
    result.ok = true;
    return result;
}

} // namespace edcpp::inference
