#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "tokenizers/clip_tokenizer.h"

namespace {

bool parse_count(const char* value, size_t& count) {
    errno          = 0;
    char* end      = nullptr;
    const auto raw = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || raw > std::numeric_limits<size_t>::max()) {
        return false;
    }
    count = static_cast<size_t>(raw);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    size_t expected_count = 0;
    const char* prompt    = nullptr;
    if (argc == 2) {
        prompt = argv[1];
    } else if (argc == 4 && std::string(argv[1]) == "--expect" && parse_count(argv[2], expected_count)) {
        prompt = argv[3];
    } else {
        std::cerr << "usage: clip-token-count [--expect COUNT] PROMPT\n";
        return 2;
    }

    CLIPTokenizer tokenizer;
    const size_t count = tokenizer.encode(prompt).size();
    std::cout << count << '\n';
    if (argc == 4 && count != expected_count) {
        std::cerr << "expected " << expected_count << " payload tokens, got " << count << '\n';
        return 1;
    }
    return 0;
}
