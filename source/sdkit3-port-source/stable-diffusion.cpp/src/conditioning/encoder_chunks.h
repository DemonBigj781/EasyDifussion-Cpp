#ifndef __SD_CONDITIONING_ENCODER_CHUNKS_H__
#define __SD_CONDITIONING_ENCODER_CHUNKS_H__

#include "tokenizers/tokenizer.h"

static inline void pad_encoder_chunks(Tokenizer& tokenizer, std::vector<int>& tokens,
                                      std::vector<float>& weights, size_t target_length, size_t chunk_len) {
    if (tokens.size() >= target_length) return;
    std::vector<int> empty_tokens;
    std::vector<float> empty_weights;
    tokenizer.pad_tokens(empty_tokens, &empty_weights, nullptr, chunk_len, chunk_len, false);
    while (tokens.size() < target_length) {
        tokens.insert(tokens.end(), empty_tokens.begin(), empty_tokens.end());
        weights.insert(weights.end(), empty_weights.begin(), empty_weights.end());
    }
}

#endif  // __SD_CONDITIONING_ENCODER_CHUNKS_H__
