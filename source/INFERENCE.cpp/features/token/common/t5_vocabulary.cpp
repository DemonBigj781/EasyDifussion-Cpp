#include "features/token/common/t5_vocabulary.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

namespace edcpp::inference {
namespace {

class JsonReader {
  public:
    explicit JsonReader(std::string_view input) : input_(input) {}

    bool consume(char expected) {
        skip_space();
        if (position_ >= input_.size() || input_[position_] != expected)
            return fail(std::string("expected '") + expected + "'");
        ++position_;
        return true;
    }

    bool consume_if(char expected) {
        skip_space();
        if (position_ >= input_.size() || input_[position_] != expected) return false;
        ++position_;
        return true;
    }

    bool string(std::string& value) {
        skip_space();
        if (position_ >= input_.size() || input_[position_] != '"')
            return fail("expected JSON string");
        ++position_;
        value.clear();
        while (position_ < input_.size()) {
            const auto byte = static_cast<unsigned char>(input_[position_++]);
            if (byte == '"') return true;
            if (byte < 0x20) return fail("unescaped control byte in JSON string");
            if (byte != '\\') {
                value.push_back(static_cast<char>(byte));
                continue;
            }

            if (position_ >= input_.size()) return fail("unfinished JSON escape");
            const char escape = input_[position_++];
            switch (escape) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    std::uint32_t code_point = 0;
                    if (!hex_quad(code_point)) return false;
                    if (code_point >= 0xd800 && code_point <= 0xdbff) {
                        if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                            input_[position_ + 1] != 'u') {
                            return fail("missing low surrogate in JSON string");
                        }
                        position_ += 2;
                        std::uint32_t low = 0;
                        if (!hex_quad(low)) return false;
                        if (low < 0xdc00 || low > 0xdfff)
                            return fail("invalid low surrogate in JSON string");
                        code_point = 0x10000 + ((code_point - 0xd800) << 10) +
                                     (low - 0xdc00);
                    } else if (code_point >= 0xdc00 && code_point <= 0xdfff) {
                        return fail("unexpected low surrogate in JSON string");
                    }
                    append_utf8(value, code_point);
                    break;
                }
                default: return fail("invalid JSON escape");
            }
        }
        return fail("unfinished JSON string");
    }

    bool boolean(bool& value) {
        skip_space();
        if (literal("true")) {
            value = true;
            return true;
        }
        if (literal("false")) {
            value = false;
            return true;
        }
        return fail("expected JSON boolean");
    }

    bool number(float& value) {
        std::string_view token;
        if (!number_token(token)) return false;
        const auto converted = std::from_chars(token.data(), token.data() + token.size(),
                                               value, std::chars_format::general);
        if (converted.ec != std::errc{} || converted.ptr != token.data() + token.size() ||
            !std::isfinite(value)) {
            return fail("JSON number is outside the float range");
        }
        return true;
    }

    bool integer(TokenId& value) {
        std::string_view token;
        if (!number_token(token)) return false;
        if (token.find_first_of(".eE") != std::string_view::npos)
            return fail("expected integer JSON number");
        std::int64_t parsed = 0;
        const auto converted = std::from_chars(token.data(), token.data() + token.size(), parsed);
        if (converted.ec != std::errc{} || converted.ptr != token.data() + token.size() ||
            parsed < std::numeric_limits<TokenId>::min() ||
            parsed > std::numeric_limits<TokenId>::max()) {
            return fail("JSON integer is outside the token ID range");
        }
        value = static_cast<TokenId>(parsed);
        return true;
    }

    bool skip_value(std::size_t depth = 0) {
        if (depth > 128) return fail("JSON nesting exceeds loader limit");
        skip_space();
        if (position_ >= input_.size()) return fail("expected JSON value");

        if (input_[position_] == '"') {
            std::string ignored;
            return string(ignored);
        }
        if (input_[position_] == '{') {
            ++position_;
            skip_space();
            if (consume_if('}')) return true;
            while (true) {
                std::string key;
                if (!string(key) || !consume(':') || !skip_value(depth + 1)) return false;
                if (consume_if('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (input_[position_] == '[') {
            ++position_;
            skip_space();
            if (consume_if(']')) return true;
            while (true) {
                if (!skip_value(depth + 1)) return false;
                if (consume_if(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (literal("true") || literal("false") || literal("null")) return true;
        std::string_view ignored;
        return number_token(ignored);
    }

    bool finish() {
        skip_space();
        return position_ == input_.size() || fail("unexpected content after JSON document");
    }

    const std::string& error() const { return error_; }

  private:
    bool fail(const std::string& message) {
        if (error_.empty())
            error_ = message + " at byte " + std::to_string(position_);
        return false;
    }

    void skip_space() {
        while (position_ < input_.size()) {
            const char value = input_[position_];
            if (value != ' ' && value != '\t' && value != '\n' && value != '\r') break;
            ++position_;
        }
    }

    bool literal(std::string_view value) {
        if (input_.substr(position_, value.size()) != value) return false;
        position_ += value.size();
        return true;
    }

    bool number_token(std::string_view& token) {
        skip_space();
        const auto start = position_;
        if (position_ < input_.size() && input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) return fail("unfinished JSON number");
        if (input_[position_] == '0') {
            ++position_;
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9') {
                ++position_;
            }
        } else {
            return fail("invalid JSON number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const auto digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9') {
                ++position_;
            }
            if (digits == position_) return fail("missing JSON fractional digits");
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            const auto digits = position_;
            while (position_ < input_.size() && input_[position_] >= '0' &&
                   input_[position_] <= '9') {
                ++position_;
            }
            if (digits == position_) return fail("missing JSON exponent digits");
        }
        token = input_.substr(start, position_ - start);
        return true;
    }

    bool hex_quad(std::uint32_t& value) {
        if (position_ + 4 > input_.size()) return fail("unfinished JSON Unicode escape");
        value = 0;
        for (int index = 0; index < 4; ++index) {
            const char digit = input_[position_++];
            value <<= 4;
            if (digit >= '0' && digit <= '9')
                value += static_cast<std::uint32_t>(digit - '0');
            else if (digit >= 'a' && digit <= 'f')
                value += static_cast<std::uint32_t>(digit - 'a' + 10);
            else if (digit >= 'A' && digit <= 'F')
                value += static_cast<std::uint32_t>(digit - 'A' + 10);
            else
                return fail("invalid JSON Unicode escape");
        }
        return true;
    }

    static void append_utf8(std::string& output, std::uint32_t value) {
        if (value <= 0x7f) {
            output.push_back(static_cast<char>(value));
        } else if (value <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (value >> 6)));
            output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
        } else if (value <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (value >> 12)));
            output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (value >> 18)));
            output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
        }
    }

    std::string_view input_{};
    std::size_t position_ = 0;
    std::string error_{};
};

struct ParsedVocabulary {
    std::vector<UnigramPiece> pieces{};
    TokenId unknown_token = -1;
    bool has_unknown_token = false;
    std::string metaspace{};
    MetaspacePrependScheme prepend_scheme = MetaspacePrependScheme::always;
    bool has_metaspace = false;
    bool has_prepend_scheme = false;
};

bool parse_vocab(JsonReader& reader, ParsedVocabulary& parsed) {
    if (!reader.consume('[')) return false;
    if (reader.consume_if(']')) return true;
    while (true) {
        if (!reader.consume('[')) return false;
        UnigramPiece piece;
        if (!reader.string(piece.text) || !reader.consume(',') || !reader.number(piece.score) ||
            !reader.consume(']')) {
            return false;
        }
        if (piece.text.empty()) piece.text = "<empty_token>";
        parsed.pieces.push_back(std::move(piece));
        if (reader.consume_if(']')) return true;
        if (!reader.consume(',')) return false;
    }
}

bool parse_model(JsonReader& reader, ParsedVocabulary& parsed) {
    if (!reader.consume('{')) return false;
    if (reader.consume_if('}')) return true;
    bool found_vocab = false;
    while (true) {
        std::string key;
        if (!reader.string(key) || !reader.consume(':')) return false;
        if (key == "vocab") {
            if (found_vocab) return false;
            found_vocab = true;
            if (!parse_vocab(reader, parsed)) return false;
        } else if (key == "unk_id") {
            if (parsed.has_unknown_token) return false;
            parsed.has_unknown_token = true;
            if (!reader.integer(parsed.unknown_token)) return false;
        } else if (!reader.skip_value()) {
            return false;
        }
        if (reader.consume_if('}')) break;
        if (!reader.consume(',')) return false;
    }
    return found_vocab;
}

bool parse_pre_tokenizer(JsonReader& reader, ParsedVocabulary& parsed) {
    if (!reader.consume('{')) return false;
    if (reader.consume_if('}')) return true;
    while (true) {
        std::string key;
        if (!reader.string(key) || !reader.consume(':')) return false;
        if (key == "replacement") {
            if (parsed.has_metaspace) return false;
            parsed.has_metaspace = true;
            if (!reader.string(parsed.metaspace)) return false;
        } else if (key == "add_prefix_space") {
            if (parsed.has_prepend_scheme) return false;
            bool add_prefix_space = false;
            if (!reader.boolean(add_prefix_space)) return false;
            parsed.prepend_scheme = add_prefix_space ? MetaspacePrependScheme::always
                                                     : MetaspacePrependScheme::never;
            parsed.has_prepend_scheme = true;
        } else if (key == "prepend_scheme") {
            if (parsed.has_prepend_scheme) return false;
            std::string scheme;
            if (!reader.string(scheme)) return false;
            if (scheme == "always")
                parsed.prepend_scheme = MetaspacePrependScheme::always;
            else if (scheme == "never")
                parsed.prepend_scheme = MetaspacePrependScheme::never;
            else if (scheme == "first")
                parsed.prepend_scheme = MetaspacePrependScheme::first;
            else
                return false;
            parsed.has_prepend_scheme = true;
        } else if (!reader.skip_value()) {
            return false;
        }
        if (reader.consume_if('}')) return true;
        if (!reader.consume(',')) return false;
    }
}

bool parse_root(JsonReader& reader, ParsedVocabulary& parsed) {
    if (!reader.consume('{')) return false;
    if (reader.consume_if('}')) return false;
    bool found_model = false;
    bool found_pre_tokenizer = false;
    while (true) {
        std::string key;
        if (!reader.string(key) || !reader.consume(':')) return false;
        if (key == "model") {
            if (found_model) return false;
            found_model = true;
            if (!parse_model(reader, parsed)) return false;
        } else if (key == "pre_tokenizer") {
            if (found_pre_tokenizer) return false;
            found_pre_tokenizer = true;
            if (!parse_pre_tokenizer(reader, parsed)) return false;
        } else if (!reader.skip_value()) {
            return false;
        }
        if (reader.consume_if('}')) break;
        if (!reader.consume(',')) return false;
    }
    return found_model && found_pre_tokenizer;
}

TokenId find_piece(const std::vector<UnigramPiece>& pieces, std::string_view text) {
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        if (pieces[index].text == text) return static_cast<TokenId>(index);
    }
    return -1;
}

bool is_single_utf8_character(std::string_view text) {
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
    return !((lead == 0xe0 && second < 0xa0) ||
             (lead == 0xed && second >= 0xa0) ||
             (lead == 0xf0 && second < 0x90) ||
             (lead == 0xf4 && second > 0x8f));
}

} // namespace

T5VocabularyLoadResult load_t5_unigram_json(std::string_view serialized) {
    T5VocabularyLoadResult result;
    JsonReader reader(serialized);
    ParsedVocabulary parsed;
    if (!parse_root(reader, parsed) || !reader.finish()) {
        result.message = reader.error().empty() ? "invalid T5 tokenizer JSON structure"
                                                : reader.error();
        return result;
    }
    if (parsed.pieces.empty()) {
        result.message = "T5 tokenizer vocabulary is empty";
        return result;
    }
    if (parsed.pieces.size() >
        static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
        result.message = "T5 tokenizer vocabulary exceeds the token ID range";
        return result;
    }
    std::unordered_set<std::string> unique_pieces;
    for (const auto& piece : parsed.pieces) {
        if (!unique_pieces.insert(piece.text).second) {
            result.message = "T5 tokenizer vocabulary contains duplicate piece: " + piece.text;
            return result;
        }
    }
    if (!parsed.has_metaspace || !is_single_utf8_character(parsed.metaspace)) {
        result.message = "T5 tokenizer JSON metaspace must be one Unicode character";
        return result;
    }
    if (!parsed.has_prepend_scheme) {
        result.message = "T5 tokenizer JSON has no metaspace prepend policy";
        return result;
    }

    result.config.padding_token = find_piece(parsed.pieces, "<pad>");
    result.config.end_token = find_piece(parsed.pieces, "</s>");
    result.config.beginning_token = find_piece(parsed.pieces, "<s>");
    result.config.unknown_token = parsed.has_unknown_token
                                      ? parsed.unknown_token
                                      : find_piece(parsed.pieces, "<unk>");
    if (result.config.padding_token < 0 || result.config.end_token < 0 ||
        result.config.unknown_token < 0 ||
        static_cast<std::size_t>(result.config.unknown_token) >= parsed.pieces.size()) {
        result.message = "T5 tokenizer JSON has invalid special-token IDs";
        return result;
    }
    result.config.metaspace = std::move(parsed.metaspace);
    result.config.prepend_scheme = parsed.prepend_scheme;
    result.pieces = std::move(parsed.pieces);
    result.ok = true;
    return result;
}

} // namespace edcpp::inference
