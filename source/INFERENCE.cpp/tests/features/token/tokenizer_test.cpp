#include "features/token/common/tokenizer.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
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

    const std::string clip_merges =
        "#version: 0.2\r\n"
        "h e\r\n"
        "he l\r\n"
        "hel l\r\n"
        "hell o</w>\r\n";
    ClipTokenizer clip(clip_merges);
    assert(clip.beginning_token() == 516);
    assert(clip.end_token() == 517);
    assert(clip.padding_token() == clip.end_token());

    const auto clip_word = clip.encode("  HELLO\n");
    assert(clip_word.ok && !clip_word.truncated);
    assert((clip_word.tokens == std::vector<TokenId>{516, 515, 517}));

    const auto contraction = clip.encode("can't 12");
    assert(contraction.ok);
    assert((contraction.tokens ==
            std::vector<TokenId>{516, 66, 64, 333, 6, 339, 272, 273, 517}));

    const auto unicode = clip.encode("\xc3\xa9");
    assert(unicode.ok);
    assert((unicode.tokens == std::vector<TokenId>{516, 127, 358, 517}));

    const std::string control(1, '\0');
    const auto encoded_control = clip.encode(control);
    assert(encoded_control.ok);
    assert((encoded_control.tokens == std::vector<TokenId>{516, 444, 517}));

    ClipTokenizer clipped(clip_merges, 3);
    const auto clipped_result = clipped.encode("hello hello");
    assert(clipped_result.ok && clipped_result.truncated);
    assert((clipped_result.tokens == std::vector<TokenId>{516, 515, 517}));

    bool rejected_invalid_merges = false;
    try {
        ClipTokenizer invalid("one-field\n");
    } catch (const std::invalid_argument&) {
        rejected_invalid_merges = true;
    }
    assert(rejected_invalid_merges);
}
