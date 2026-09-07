#ifndef __SD_RUNTIME_FLEX_ATTENTION_HPP__
#define __SD_RUNTIME_FLEX_ATTENTION_HPP__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sd::flex_attention {

// Native inference-time port of the reusable selection stage from
// UMass-Embodied-AGI/FlexAttention (Apache-2.0). The reference method sums
// low-resolution visual attention across heads, normalizes each query map,
// thresholds it in 8-bit space, adaptive-max-pools to 9x9, then expands the
// selection to the high-resolution vision grid.
struct SelectionConfig {
    int low_grid_size      = 24;
    int high_grid_size     = 48;
    int pool_grid_size     = 9;
    float threshold        = 64.f;
    int original_width     = 0;
    int original_height    = 0;
};

struct SelectionResult {
    int query_count    = 0;
    int high_grid_size = 0;
    std::vector<uint8_t> mask;  // [query, high_y, high_x]

    size_t selected_count() const {
        return static_cast<size_t>(std::count(mask.begin(), mask.end(), uint8_t{1}));
    }
};

inline bool valid_config(const SelectionConfig& config) {
    return config.low_grid_size > 0 &&
           config.high_grid_size > 0 &&
           config.pool_grid_size > 0 &&
           std::isfinite(config.threshold);
}

inline SelectionResult select_high_resolution_tokens(const float* attention,
                                                      int head_count,
                                                      int query_count,
                                                      const SelectionConfig& config) {
    SelectionResult result;
    if (attention == nullptr || head_count <= 0 || query_count <= 0 || !valid_config(config)) {
        return result;
    }

    const int low_size  = config.low_grid_size;
    const int high_size = config.high_grid_size;
    const int pool_size = config.pool_grid_size;
    const size_t low_tokens = static_cast<size_t>(low_size) * low_size;
    result.query_count       = query_count;
    result.high_grid_size    = high_size;
    result.mask.assign(static_cast<size_t>(query_count) * high_size * high_size, 0);

    int margin_patches = 0;
    if (config.original_width > 0 && config.original_height > 0) {
        const int longest = std::max(config.original_width, config.original_height);
        margin_patches = static_cast<int>(
            static_cast<float>(std::abs(config.original_width - config.original_height) / 2) /
            static_cast<float>(longest) * low_size);
        margin_patches = std::clamp(margin_patches, 0, low_size / 2);
    }

    std::vector<float> summed(low_tokens);
    std::vector<uint8_t> thresholded(low_tokens);
    std::vector<uint8_t> pooled(static_cast<size_t>(pool_size) * pool_size);

    for (int query = 0; query < query_count; ++query) {
        std::fill(summed.begin(), summed.end(), 0.f);
        for (int head = 0; head < head_count; ++head) {
            const size_t base = (static_cast<size_t>(head) * query_count + query) * low_tokens;
            for (size_t token = 0; token < low_tokens; ++token) {
                const float value = attention[base + token];
                if (std::isfinite(value)) {
                    summed[token] += value;
                }
            }
        }

        if (margin_patches > 0) {
            if (config.original_width > config.original_height) {
                for (int y = 0; y < margin_patches; ++y) {
                    std::fill_n(summed.begin() + static_cast<size_t>(y) * low_size, low_size, 0.f);
                    std::fill_n(summed.begin() + static_cast<size_t>(low_size - 1 - y) * low_size, low_size, 0.f);
                }
            } else {
                for (int y = 0; y < low_size; ++y) {
                    for (int x = 0; x < margin_patches; ++x) {
                        summed[static_cast<size_t>(y) * low_size + x] = 0.f;
                        summed[static_cast<size_t>(y) * low_size + (low_size - 1 - x)] = 0.f;
                    }
                }
            }
        }

        const auto [min_it, max_it] = std::minmax_element(summed.begin(), summed.end());
        const float range = *max_it - *min_it;
        for (size_t token = 0; token < low_tokens; ++token) {
            const float normalized = range > 1e-12f ? (summed[token] - *min_it) / range : 0.f;
            thresholded[token] = normalized * 255.f > config.threshold ? 1 : 0;
        }

        // Match adaptive_max_pool2d bin boundaries: floor(start), ceil(end).
        for (int py = 0; py < pool_size; ++py) {
            const int y0 = (py * low_size) / pool_size;
            const int y1 = ((py + 1) * low_size + pool_size - 1) / pool_size;
            for (int px = 0; px < pool_size; ++px) {
                const int x0 = (px * low_size) / pool_size;
                const int x1 = ((px + 1) * low_size + pool_size - 1) / pool_size;
                uint8_t selected = 0;
                for (int y = y0; y < y1 && !selected; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        selected = std::max(selected,
                                            thresholded[static_cast<size_t>(y) * low_size + x]);
                    }
                }
                pooled[static_cast<size_t>(py) * pool_size + px] = selected;
            }
        }

        const size_t query_base = static_cast<size_t>(query) * high_size * high_size;
        for (int y = 0; y < high_size; ++y) {
            const int py = std::min(pool_size - 1, (y * pool_size) / high_size);
            for (int x = 0; x < high_size; ++x) {
                const int px = std::min(pool_size - 1, (x * pool_size) / high_size);
                result.mask[query_base + static_cast<size_t>(y) * high_size + x] =
                    pooled[static_cast<size_t>(py) * pool_size + px];
            }
        }
    }
    return result;
}

}  // namespace sd::flex_attention

#endif  // __SD_RUNTIME_FLEX_ATTENTION_HPP__
