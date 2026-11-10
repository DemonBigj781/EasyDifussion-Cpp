#include "features/token/common/tokenizer_resource.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool read_file(const std::string& path, std::string& contents) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    contents.assign(std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>());
    return input.good() || input.eof();
}

bool parse_token_ids(std::string_view text,
                     std::vector<edcpp::inference::TokenId>& tokens) {
    if (text.empty()) return true;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto delimiter = text.find(',', offset);
        const auto end = delimiter == std::string_view::npos ? text.size() : delimiter;
        const auto token = text.substr(offset, end - offset);
        if (token.empty()) return false;

        edcpp::inference::TokenId value = 0;
        const auto converted = std::from_chars(token.data(), token.data() + token.size(), value);
        if (converted.ec != std::errc{} || converted.ptr != token.data() + token.size() ||
            value < 0) {
            return false;
        }
        tokens.push_back(value);
        if (delimiter == std::string_view::npos) return true;
        offset = delimiter + 1;
    }
    return false;
}

void print_tokens(const std::vector<edcpp::inference::TokenId>& tokens) {
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) std::cerr << ',';
        std::cerr << tokens[index];
    }
}

} // namespace

int main(int argc, char** argv) {
    using namespace edcpp::inference;

    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " <clip-merges|t5-json> <tokenizer-resource> <vectors.tsv>\n";
        return 2;
    }

    TokenizerResourceFormat format;
    const std::string format_name = argv[1];
    if (format_name == "clip-merges")
        format = TokenizerResourceFormat::clip_merges;
    else if (format_name == "t5-json")
        format = TokenizerResourceFormat::hugging_face_unigram_json;
    else {
        std::cerr << "unsupported tokenizer resource format: " << format_name << '\n';
        return 2;
    }

    std::string serialized;
    if (!read_file(argv[2], serialized)) {
        std::cerr << "could not read tokenizer resource: " << argv[2] << '\n';
        return 2;
    }

    TokenizerResourceView resource;
    resource.format = format;
    resource.identifier = argv[2];
    resource.serialized = serialized;
    auto loaded = load_tokenizer_resource(resource);
    if (!loaded.ok) {
        std::cerr << loaded.message << '\n';
        return 2;
    }

    std::ifstream vectors(argv[3]);
    if (!vectors) {
        std::cerr << "could not read golden vectors: " << argv[3] << '\n';
        return 2;
    }

    std::size_t checked = 0;
    std::size_t line_number = 0;
    std::string line;
    while (std::getline(vectors, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;

        const auto separator = line.find('\t');
        if (separator == std::string::npos) {
            std::cerr << argv[3] << ':' << line_number
                      << ": expected comma-separated token IDs, a tab, and prompt text\n";
            return 2;
        }

        std::vector<TokenId> expected;
        if (!parse_token_ids(std::string_view(line).substr(0, separator), expected)) {
            std::cerr << argv[3] << ':' << line_number << ": invalid token ID list\n";
            return 2;
        }

        const auto actual = loaded.tokenizer->encode(line.substr(separator + 1));
        if (!actual.ok) {
            std::cerr << argv[3] << ':' << line_number
                      << ": tokenization failed: " << actual.message << '\n';
            return 1;
        }
        if (actual.tokens != expected) {
            std::cerr << argv[3] << ':' << line_number << ": expected ";
            print_tokens(expected);
            std::cerr << " but received ";
            print_tokens(actual.tokens);
            std::cerr << '\n';
            return 1;
        }
        ++checked;
    }

    if (!vectors.eof()) {
        std::cerr << "could not finish reading golden vectors: " << argv[3] << '\n';
        return 2;
    }
    if (checked == 0) {
        std::cerr << "golden-vector file contains no test cases\n";
        return 2;
    }

    std::cout << "verified " << checked << " tokenizer golden vectors\n";
    return 0;
}
