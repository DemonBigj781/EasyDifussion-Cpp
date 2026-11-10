#include "features/token/common/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace edcpp::inference {
namespace {

std::string pair_key(const std::string& left, const std::string& right) {
    return left + '\0' + right;
}

bool is_ascii_word(unsigned char value) {
    return value < 128 && (std::isalnum(value) != 0 || value == '_');
}

std::string utf8(std::uint32_t value) {
    std::string encoded;
    if (value <= 0x7f) {
        encoded.push_back(static_cast<char>(value));
    } else if (value <= 0x7ff) {
        encoded.push_back(static_cast<char>(0xc0 | (value >> 6)));
        encoded.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        encoded.push_back(static_cast<char>(0xe0 | (value >> 12)));
        encoded.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        encoded.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    return encoded;
}

struct ClipByteTable {
    std::array<std::string, 256> by_byte{};
    std::vector<std::pair<unsigned char, std::string>> vocabulary_order{};
};

const ClipByteTable& clip_byte_table() {
    static const ClipByteTable table = [] {
        ClipByteTable value;
        std::array<bool, 256> direct{};

        const auto add_direct_range = [&](unsigned int first, unsigned int last) {
            for (unsigned int byte = first; byte <= last; ++byte) {
                direct[byte] = true;
                value.by_byte[byte] = utf8(byte);
                value.vocabulary_order.emplace_back(static_cast<unsigned char>(byte),
                                                     value.by_byte[byte]);
            }
        };
        add_direct_range('!', '~');
        add_direct_range(161, 172);
        add_direct_range(174, 255);

        std::uint32_t replacement = 256;
        for (unsigned int byte = 0; byte < direct.size(); ++byte) {
            if (direct[byte]) continue;
            value.by_byte[byte] = utf8(replacement++);
            value.vocabulary_order.emplace_back(static_cast<unsigned char>(byte),
                                                 value.by_byte[byte]);
        }
        return value;
    }();
    return table;
}

bool ascii_alpha(unsigned char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

bool ascii_digit(unsigned char value) {
    return value >= '0' && value <= '9';
}

bool ascii_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\v' || value == '\f';
}

char lower_ascii(char value) {
    const auto byte = static_cast<unsigned char>(value);
    return byte >= 'A' && byte <= 'Z' ? static_cast<char>(byte + ('a' - 'A')) : value;
}

bool begins_clip_contraction(const std::string& text, std::size_t position,
                             std::size_t& length) {
    static constexpr std::array<const char*, 7> suffixes = {
        "'re", "'ve", "'ll", "'s", "'t", "'m", "'d",
    };
    for (const char* suffix : suffixes) {
        const std::string candidate(suffix);
        if (position + candidate.size() > text.size()) continue;
        bool matches = true;
        for (std::size_t offset = 0; offset < candidate.size(); ++offset) {
            if (lower_ascii(text[position + offset]) != candidate[offset]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            length = candidate.size();
            return true;
        }
    }
    return false;
}

std::vector<std::string> clip_pretokenize(const std::string& text,
                                          bool lowercase_ascii) {
    std::vector<std::string> tokens;
    std::size_t position = 0;
    while (position < text.size()) {
        const auto byte = static_cast<unsigned char>(text[position]);
        if (ascii_space(byte)) {
            ++position;
            continue;
        }

        std::size_t contraction_length = 0;
        if (text[position] == '\'' &&
            begins_clip_contraction(text, position, contraction_length)) {
            std::string token = text.substr(position, contraction_length);
            if (lowercase_ascii)
                std::transform(token.begin(), token.end(), token.begin(), lower_ascii);
            tokens.push_back(std::move(token));
            position += contraction_length;
            continue;
        }

        const auto start = position;
        if (ascii_alpha(byte)) {
            while (position < text.size()) {
                const auto current = static_cast<unsigned char>(text[position]);
                if (!ascii_alpha(current)) break;
                ++position;
            }
        } else if (ascii_digit(byte)) {
            ++position;
        } else {
            while (position < text.size()) {
                const auto current = static_cast<unsigned char>(text[position]);
                if (ascii_space(current) || ascii_alpha(current) || ascii_digit(current)) {
                    break;
                }
                std::size_t ignored = 0;
                if (text[position] == '\'' &&
                    begins_clip_contraction(text, position, ignored)) {
                    break;
                }
                ++position;
            }
        }

        std::string token = text.substr(start, position - start);
        if (lowercase_ascii)
            std::transform(token.begin(), token.end(), token.begin(), lower_ascii);
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<BpeTokenizer::Merge> parse_clip_merges(const std::string& serialized) {
    std::vector<BpeTokenizer::Merge> merges;
    std::istringstream input(serialized);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto content = line.find_first_not_of(" \t");
        if (content == std::string::npos || line[content] == '#') continue;

        std::istringstream fields(line.substr(content));
        std::string left;
        std::string right;
        std::string extra;
        if (!(fields >> left >> right) || (fields >> extra))
            throw std::invalid_argument("invalid CLIP merge line: " + line);
        merges.emplace_back(std::move(left), std::move(right));
    }
    return merges;
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
    if (config_.pretokenizer == Pretokenizer::clip)
        return clip_pretokenize(text, config_.lowercase_ascii);

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
    for (unsigned char byte : word) {
        if (config_.byte_encoding == ByteEncoding::clip)
            pieces.push_back(clip_byte_table().by_byte[byte]);
        else
            pieces.emplace_back(1, static_cast<char>(byte));
    }
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

ClipTokenizer::ClipTokenizer(const std::string& serialized_merges,
                             std::size_t maximum_tokens,
                             TokenId padding_token) {
    auto merges = parse_clip_merges(serialized_merges);
    BpeTokenizer::Vocabulary vocabulary;
    TokenId next_id = 0;
    for (const auto& entry : clip_byte_table().vocabulary_order)
        vocabulary[entry.second] = next_id++;
    for (const auto& entry : clip_byte_table().vocabulary_order)
        vocabulary[entry.second + "</w>"] = next_id++;
    for (const auto& merge : merges)
        vocabulary[merge.first + merge.second] = next_id++;

    beginning_token_ = next_id++;
    end_token_ = next_id++;
    padding_token_ = padding_token < 0 ? end_token_ : padding_token;
    vocabulary["<|startoftext|>"] = beginning_token_;
    vocabulary["<|endoftext|>"] = end_token_;

    TokenizerConfig config;
    config.unknown_token = end_token_;
    config.beginning_token = beginning_token_;
    config.end_token = end_token_;
    config.padding_token = padding_token_;
    config.pretokenizer = Pretokenizer::clip;
    config.byte_encoding = ByteEncoding::clip;
    config.maximum_tokens = maximum_tokens;
    tokenizer_ = std::make_unique<BpeTokenizer>(std::move(vocabulary),
                                                std::move(merges), config);
}

TokenizationResult ClipTokenizer::encode(const std::string& text) const {
    return tokenizer_->encode(text);
}

} // namespace edcpp::inference
