#include "runtime/flex_attention.hpp"

#include <cstring>

#include "stable-diffusion.h"

size_t sd_flex_attention_mask_size(int query_count, int high_grid_size) {
    if (query_count <= 0 || high_grid_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(query_count) * high_grid_size * high_grid_size;
}

bool sd_flex_attention_select(const float* attention,
                              int head_count,
                              int query_count,
                              const sd_flex_attention_params_t* params,
                              uint8_t* mask,
                              size_t mask_size) {
    if (params == nullptr || mask == nullptr) {
        return false;
    }
    sd::flex_attention::SelectionConfig config;
    config.low_grid_size   = params->low_grid_size;
    config.high_grid_size  = params->high_grid_size;
    config.pool_grid_size  = params->pool_grid_size;
    config.threshold       = params->threshold;
    config.original_width  = params->original_width;
    config.original_height = params->original_height;
    const auto result = sd::flex_attention::select_high_resolution_tokens(attention,
                                                                           head_count,
                                                                           query_count,
                                                                           config);
    if (result.mask.empty() || result.mask.size() > mask_size) {
        return false;
    }
    std::memcpy(mask, result.mask.data(), result.mask.size());
    return true;
}
