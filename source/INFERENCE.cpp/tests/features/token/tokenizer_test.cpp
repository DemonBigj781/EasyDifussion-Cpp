#include "features/token/common/tokenizer.hpp"

#include <cassert>
#include <vector>

int main() {
    using namespace edcpp::inference;

    BpeTokenizer::Vocabulary vocabulary = {
        {"h", 1}, {"e", 2}, {"l", 3}, {"o</w>", 4}, {"he", 5},
        {"ll", 6}, {"hello</w>", 7}, {"!</w>", 8}, {"w", 9},
        {"r", 10}, {"d</w>", 11}, {"wo", 12}, {"wor", 13},
        {"world</w>", 14},
    };
    std::vector<BpeTokenizer::Merge> merges = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o</w>"},
        {"w", "o"}, {"wo", "r"}, {"wor", "l"}, {"worl", "d</w>"},
    };
    TokenizerConfig config;
    config.beginning_token = 100;
    config.end_token = 101;
    config.unknown_token = 0;
    BpeTokenizer tokenizer(std::move(vocabulary), std::move(merges), config);

    const auto encoded = tokenizer.encode("Hello world!");
    assert(encoded.ok);
    assert((encoded.tokens == std::vector<TokenId>{100, 7, 14, 8, 101}));
    assert(encoded.unknown_pieces == 0);

    const auto unknown = tokenizer.encode("?");
    assert(unknown.ok && unknown.unknown_pieces == 1);
    assert((unknown.tokens == std::vector<TokenId>{100, 0, 101}));

    auto limited_config = config;
    limited_config.maximum_tokens = 3;
    BpeTokenizer limited({{"a</w>", 20}, {"b</w>", 21}}, {}, limited_config);
    const auto truncated = limited.encode("a b a");
    assert(truncated.ok && truncated.truncated);
    assert((truncated.tokens == std::vector<TokenId>{100, 20, 101}));

    BpeTokenizer missing({{"a</w>", 20}}, {}, TokenizerConfig{});
    const auto failed = missing.encode("b");
    assert(!failed.ok);
}
