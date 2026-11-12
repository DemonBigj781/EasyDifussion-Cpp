#include <iostream>
#include <stdexcept>
#include <vector>

#include "conditioning/encoder_chunks.h"

// Exercise production padding without loading tokenizer vocabularies or models.
class PaddingTokenizer : public Tokenizer {
    std::string decode_token(int) const override { return ""; }
public:
    PaddingTokenizer(bool clip, int padding) {
        add_bos_token = clip;
        add_eos_token = true;
        BOS_TOKEN_ID = 49406;
        EOS_TOKEN_ID = clip ? 49407 : 1;
        PAD_TOKEN_ID = padding;
    }
    std::vector<int> encode(const std::string&, on_new_token_cb_t) override { return {}; }
};

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check(PaddingTokenizer& tokenizer, bool clip, int padding, size_t chunks) {
    std::vector<int> tokens{5, 6};
    std::vector<float> weights{0.5f, 1.5f};
    tokenizer.pad_tokens(tokens, &weights, nullptr, 77, 77, false);
    auto original_tokens = tokens;
    auto original_weights = weights;
    pad_encoder_chunks(tokenizer, tokens, weights, chunks * 77, 77);
    require(tokens.size() == chunks * 77 && weights.size() == tokens.size(), "encoder chunk sizes remain unequal");
    require(std::equal(original_tokens.begin(), original_tokens.end(), tokens.begin()), "original tokens changed");
    require(std::equal(original_weights.begin(), original_weights.end(), weights.begin()), "original prompt weights changed");
    for (size_t chunk = 1; chunk < chunks; ++chunk) {
        for (size_t i = 0; i < 77; ++i) {
            int expected = padding;
            if (i == 0) expected = clip ? 49406 : 1;
            if (clip && i == 1) expected = 49407;
            require(tokens.at(chunk * 77 + i) == expected, "empty chunk special/padding tokens changed");
            require(weights.at(chunk * 77 + i) == 1.f, "empty chunk weight is not neutral");
        }
    }
    auto padded_tokens = tokens;
    auto padded_weights = weights;
    pad_encoder_chunks(tokenizer, tokens, weights, chunks * 77, 77);
    pad_encoder_chunks(tokenizer, tokens, weights, 77, 77);
    require(tokens == padded_tokens && weights == padded_weights, "padding repeated or truncated existing chunks");
}

int main() {
    try {
        PaddingTokenizer clip_l(true, 49407), clip_g(true, 0), t5(false, 0);
        for (size_t chunks : {1, 2, 4}) {
            check(clip_l, true, 49407, chunks);
            check(clip_g, true, 0, chunks);
            check(t5, false, 0, chunks);
        }
        std::cout << "PASS SD3 CLIP-L/G/T5 empty-chunk alignment, distinct special/pad tokens, weighted-prefix preservation and idempotence\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
