#include "features/token/common/t5_unigram.hpp"
#include "features/token/common/t5_vocabulary.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::vector<edcpp::inference::UnigramPiece> test_pieces() {
    return {
        {"<pad>", 0.0f}, {"</s>", 0.0f}, {"<unk>", -10.0f},
        {u8"\u2581", -1.0f}, {"h", -2.0f}, {"e", -2.0f},
        {"l", -2.0f}, {"o", -2.0f}, {u8"\u2581hello", 5.0f},
        {u8"\u2581world", 4.0f}, {"w", -2.0f}, {"r", -2.0f},
        {"d", -2.0f}, {"ab", 10.0f}, {u8"\u2581a", 5.0f},
        {"b", -5.0f},
    };
}

} // namespace

int main() {
    using namespace edcpp::inference;

    T5UnigramTokenizer tokenizer(test_pieces());

    const auto words = tokenizer.encode("hello   world");
    assert(words.ok && !words.truncated && words.unknown_pieces == 0);
    assert((words.tokens == std::vector<TokenId>{8, 9, 1}));

    const auto global_path = tokenizer.encode("ab");
    assert(global_path.ok);
    assert((global_path.tokens == std::vector<TokenId>{3, 13, 1}));

    const auto unknown = tokenizer.encode("hello!");
    assert(unknown.ok && unknown.unknown_pieces == 1);
    assert((unknown.tokens == std::vector<TokenId>{8, 2, 1}));

    const auto unicode_unknown = tokenizer.encode("\xf0\x9f\x99\x82");
    assert(unicode_unknown.ok && unicode_unknown.unknown_pieces == 1);
    assert((unicode_unknown.tokens == std::vector<TokenId>{3, 2, 1}));

    const auto explicit_special = tokenizer.encode("hello</s>world");
    assert(explicit_special.ok);
    assert((explicit_special.tokens == std::vector<TokenId>{8, 1, 9, 1}));

    T5UnigramConfig first_prefix_config;
    first_prefix_config.prepend_scheme = MetaspacePrependScheme::first;
    T5UnigramTokenizer first_prefix(test_pieces(), first_prefix_config);
    const auto first_prefix_special = first_prefix.encode("hello</s>world");
    assert(first_prefix_special.ok);
    assert((first_prefix_special.tokens ==
            std::vector<TokenId>{8, 1, 10, 7, 11, 6, 12, 1}));

    const auto empty = tokenizer.encode("");
    assert(empty.ok);
    assert((empty.tokens == std::vector<TokenId>{3, 1}));

    T5UnigramConfig limited_config;
    limited_config.maximum_tokens = 3;
    T5UnigramTokenizer limited(test_pieces(), limited_config);
    const auto truncated = limited.encode("hello world hello");
    assert(truncated.ok && truncated.truncated);
    assert((truncated.tokens == std::vector<TokenId>{8, 9, 1}));

    auto umt5_pieces = test_pieces();
    umt5_pieces.insert(umt5_pieces.begin() + 2, {"<s>", 0.0f});
    T5UnigramConfig umt5_config;
    umt5_config.unknown_token = 3;
    umt5_config.beginning_token = 2;
    umt5_config.end_token = 1;
    umt5_config.add_beginning = true;
    T5UnigramTokenizer umt5(std::move(umt5_pieces), umt5_config);
    const auto umt5_empty = umt5.encode("");
    assert(umt5_empty.ok);
    assert((umt5_empty.tokens == std::vector<TokenId>{2, 4, 1}));

    bool rejected_duplicate = false;
    try {
        T5UnigramTokenizer duplicate({{"<pad>", 0.0f}, {"</s>", 0.0f},
                                      {"<pad>", -1.0f}});
    } catch (const std::invalid_argument&) {
        rejected_duplicate = true;
    }
    assert(rejected_duplicate);

    bool rejected_score = false;
    try {
        auto pieces = test_pieces();
        pieces.back().score = std::numeric_limits<float>::infinity();
        T5UnigramTokenizer invalid_score(std::move(pieces));
    } catch (const std::invalid_argument&) {
        rejected_score = true;
    }
    assert(rejected_score);

    const auto loaded = load_t5_unigram_json(R"json(
        {
          "metadata": {"emoji": "\uD83D\uDE42", "ignored": [true, null, 4]},
          "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always",
            "split": true
          },
          "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": [
              ["<pad>", 0],
              ["</s>", 0.0],
              ["<unk>", -1e1],
              ["\u2581", -1.0],
              ["h", -2.0],
              ["e", -2.0],
              ["l", -2.0],
              ["o", -2.0],
              ["\u2581hello", 5.0]
            ]
          }
        }
    )json");
    assert(loaded.ok);
    assert(loaded.config.padding_token == 0);
    assert(loaded.config.end_token == 1);
    assert(loaded.config.unknown_token == 2);
    assert(loaded.config.beginning_token == -1);
    assert(loaded.config.metaspace == u8"\u2581");
    assert(loaded.config.prepend_scheme == MetaspacePrependScheme::always);
    T5UnigramTokenizer loaded_tokenizer(loaded.pieces, loaded.config);
    const auto loaded_tokens = loaded_tokenizer.encode("hello");
    assert(loaded_tokens.ok);
    assert((loaded_tokens.tokens == std::vector<TokenId>{8, 1}));

    const auto malformed_tuple = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","add_prefix_space":true},
         "model":{"unk_id":2,"vocab":[["<pad>",0,1],["</s>",0],["<unk>",-1]]}}
    )json");
    assert(!malformed_tuple.ok);

    const auto malformed_unicode = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\uD800","add_prefix_space":true},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1]]}}
    )json");
    assert(!malformed_unicode.ok);

    const auto fractional_id = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","add_prefix_space":true},
         "model":{"unk_id":2.5,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1]]}}
    )json");
    assert(!fractional_id.ok);

    const auto trailing_content = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","add_prefix_space":true},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1]]}} false
    )json");
    assert(!trailing_content.ok);

    const auto duplicate_json_piece = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","add_prefix_space":true},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1],
                                          ["<unk>",-2]]}}
    )json");
    assert(!duplicate_json_piece.ok);

    const auto legacy_schema = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","add_prefix_space":false},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1],
                                          ["h",1]]}}
    )json");
    assert(legacy_schema.ok);
    assert(legacy_schema.config.prepend_scheme == MetaspacePrependScheme::never);
    T5UnigramTokenizer legacy_tokenizer(legacy_schema.pieces, legacy_schema.config);
    const auto legacy_empty = legacy_tokenizer.encode("");
    assert(legacy_empty.ok);
    assert((legacy_empty.tokens == std::vector<TokenId>{1}));

    const auto invalid_scheme = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"\u2581","prepend_scheme":"sometimes"},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1]]}}
    )json");
    assert(!invalid_scheme.ok);

    const auto invalid_metaspace = load_t5_unigram_json(R"json(
        {"pre_tokenizer":{"replacement":"two","prepend_scheme":"always"},
         "model":{"unk_id":2,"vocab":[["<pad>",0],["</s>",0],["<unk>",-1]]}}
    )json");
    assert(!invalid_metaspace.ok);
}
