#include "features/token/common/tokenizer_resource.hpp"

#include <cassert>
#include <string>
#include <vector>

int main() {
    using namespace edcpp::inference;

    const std::string clip_merges =
        "#version: 0.2\n"
        "h e\n"
        "he l\n"
        "hel l\n"
        "hell o</w>\n";
    TokenizerResourceView clip_resource;
    clip_resource.format = TokenizerResourceFormat::clip_merges;
    clip_resource.identifier = "clip/merges.txt";
    clip_resource.serialized = clip_merges;
    clip_resource.maximum_tokens = 3;
    clip_resource.padding_token = 42;

    auto clip = load_tokenizer_resource(clip_resource);
    assert(clip.ok && clip.tokenizer);
    assert(clip.special_tokens.unknown == 517);
    assert(clip.special_tokens.beginning == 516);
    assert(clip.special_tokens.end == 517);
    assert(clip.special_tokens.padding == 42);
    assert(clip.special_tokens.adds_beginning && clip.special_tokens.adds_end);
    const auto clip_tokens = clip.tokenizer->encode("hello hello");
    assert(clip_tokens.ok && clip_tokens.truncated);
    assert((clip_tokens.tokens == std::vector<TokenId>{516, 515, 517}));

    const std::string t5_json = R"json(
        {
          "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always"
          },
          "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": [
              ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", -10.0],
              ["\u2581", -1.0], ["\u2581hello", 5.0]
            ]
          }
        }
    )json";
    TokenizerResourceView t5_resource;
    t5_resource.format = TokenizerResourceFormat::hugging_face_unigram_json;
    t5_resource.identifier = "text_encoder/tokenizer.json";
    t5_resource.serialized = t5_json;
    t5_resource.maximum_tokens = 2;
    t5_resource.padding_token = 3;

    auto t5 = load_tokenizer_resource(t5_resource);
    assert(t5.ok && t5.tokenizer);
    assert(t5.special_tokens.unknown == 2);
    assert(t5.special_tokens.beginning == -1);
    assert(t5.special_tokens.end == 1);
    assert(t5.special_tokens.padding == 3);
    assert(!t5.special_tokens.adds_beginning && t5.special_tokens.adds_end);
    const auto t5_tokens = t5.tokenizer->encode("hello hello");
    assert(t5_tokens.ok && t5_tokens.truncated);
    assert((t5_tokens.tokens == std::vector<TokenId>{4, 1}));

    TokenizerResourceView empty_resource;
    empty_resource.identifier = "empty";
    const auto empty = load_tokenizer_resource(empty_resource);
    assert(!empty.ok && !empty.tokenizer);
    assert(empty.message.find("empty") != std::string::npos);

    auto malformed_resource = t5_resource;
    malformed_resource.identifier = "malformed";
    malformed_resource.serialized = "{}";
    const auto malformed = load_tokenizer_resource(malformed_resource);
    assert(!malformed.ok && !malformed.tokenizer);
    assert(malformed.message.find("malformed") != std::string::npos);

    auto invalid_format = clip_resource;
    invalid_format.identifier = "future-format";
    invalid_format.format = static_cast<TokenizerResourceFormat>(255);
    const auto unsupported = load_tokenizer_resource(invalid_format);
    assert(!unsupported.ok && !unsupported.tokenizer);
    assert(unsupported.message.find("not supported") != std::string::npos);
}
