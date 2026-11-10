#include "flex_attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace edcpp::api::attention::flex::cpu::definition::cpu {
namespace {

std::size_t required_mask_size(const NativeRequest& request) noexcept {
    if (request.query_count <= 0 || request.config.high_grid_size <= 0) {
        return 0;
    }
    const auto queries = static_cast<std::size_t>(request.query_count);
    const auto grid =
        static_cast<std::size_t>(request.config.high_grid_size);
    if (grid > std::numeric_limits<std::size_t>::max() / grid) return 0;
    const std::size_t tokens = grid * grid;
    if (queries > std::numeric_limits<std::size_t>::max() / tokens) return 0;
    return queries * tokens;
}

} // namespace

NativeCapabilities capabilities() noexcept {
    return {true, true, true, true, true, true};
}

NativeResult validate(const NativeRequest& request) noexcept {
    if (request.attention == nullptr || request.mask == nullptr) {
        return {false, 0, "CPU Flex Attention requires input and output buffers"};
    }
    if (request.head_count <= 0 || request.query_count <= 0 ||
        request.config.low_grid_size <= 0 ||
        request.config.high_grid_size <= 0 ||
        request.config.pool_grid_size <= 0 ||
        !std::isfinite(request.config.threshold)) {
        return {false, 0, "CPU Flex Attention configuration is invalid"};
    }
    const std::size_t required = required_mask_size(request);
    if (required == 0 || request.capacity < required) {
        return {false, 0, "CPU Flex Attention output capacity is insufficient"};
    }
    return {true, 0, nullptr};
}

NativeResult select(const NativeRequest& request) {
    const NativeResult checked = validate(request);
    if (!checked.ok) return checked;

    const int low_size = request.config.low_grid_size;
    const int high_size = request.config.high_grid_size;
    const int pool_size = request.config.pool_grid_size;
    const std::size_t low_tokens =
        static_cast<std::size_t>(low_size) * low_size;

    int margin_patches = 0;
    if (request.config.original_width > 0 &&
        request.config.original_height > 0) {
        const int longest = std::max(
            request.config.original_width,
            request.config.original_height);
        margin_patches = static_cast<int>(
            static_cast<float>(
                std::abs(
                    request.config.original_width -
                    request.config.original_height) /
                2) /
            static_cast<float>(longest) * low_size);
        margin_patches =
            std::clamp(margin_patches, 0, low_size / 2);
    }

    std::vector<float> summed(low_tokens);
    std::vector<std::uint8_t> thresholded(low_tokens);
    std::vector<std::uint8_t> pooled(
        static_cast<std::size_t>(pool_size) * pool_size);
    std::size_t selected_count = 0;

    for (int query = 0; query < request.query_count; ++query) {
        std::fill(summed.begin(), summed.end(), 0.0f);
        for (int head = 0; head < request.head_count; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(head) * request.query_count +
                 query) *
                low_tokens;
            for (std::size_t token = 0; token < low_tokens; ++token) {
                const float value = request.attention[base + token];
                if (std::isfinite(value)) summed[token] += value;
            }
        }

        if (margin_patches > 0) {
            if (request.config.original_width >
                request.config.original_height) {
                for (int y = 0; y < margin_patches; ++y) {
                    std::fill_n(
                        summed.begin() +
                            static_cast<std::size_t>(y) * low_size,
                        low_size, 0.0f);
                    std::fill_n(
                        summed.begin() +
                            static_cast<std::size_t>(
                                low_size - 1 - y) *
                                low_size,
                        low_size, 0.0f);
                }
            } else {
                for (int y = 0; y < low_size; ++y) {
                    for (int x = 0; x < margin_patches; ++x) {
                        summed[
                            static_cast<std::size_t>(y) * low_size + x] =
                            0.0f;
                        summed[
                            static_cast<std::size_t>(y) * low_size +
                            (low_size - 1 - x)] = 0.0f;
                    }
                }
            }
        }

        const auto extrema =
            std::minmax_element(summed.begin(), summed.end());
        const float range = *extrema.second - *extrema.first;
        for (std::size_t token = 0; token < low_tokens; ++token) {
            const float normalized = range > 1e-12f
                ? (summed[token] - *extrema.first) / range
                : 0.0f;
            thresholded[token] =
                normalized * 255.0f > request.config.threshold ? 1 : 0;
        }

        for (int pool_y = 0; pool_y < pool_size; ++pool_y) {
            const int y_begin = (pool_y * low_size) / pool_size;
            const int y_end =
                ((pool_y + 1) * low_size + pool_size - 1) / pool_size;
            for (int pool_x = 0; pool_x < pool_size; ++pool_x) {
                const int x_begin = (pool_x * low_size) / pool_size;
                const int x_end =
                    ((pool_x + 1) * low_size + pool_size - 1) /
                    pool_size;
                std::uint8_t selected = 0;
                for (int y = y_begin; y < y_end && !selected; ++y) {
                    for (int x = x_begin; x < x_end; ++x) {
                        selected = std::max(
                            selected,
                            thresholded[
                                static_cast<std::size_t>(y) * low_size +
                                x]);
                    }
                }
                pooled[
                    static_cast<std::size_t>(pool_y) * pool_size +
                    pool_x] = selected;
            }
        }

        const std::size_t query_base =
            static_cast<std::size_t>(query) * high_size * high_size;
        for (int y = 0; y < high_size; ++y) {
            const int pool_y =
                std::min(pool_size - 1, (y * pool_size) / high_size);
            for (int x = 0; x < high_size; ++x) {
                const int pool_x =
                    std::min(pool_size - 1, (x * pool_size) / high_size);
                const std::uint8_t selected =
                    pooled[
                        static_cast<std::size_t>(pool_y) * pool_size +
                        pool_x];
                request.mask[
                    query_base + static_cast<std::size_t>(y) * high_size +
                    x] = selected;
                selected_count += selected;
            }
        }
    }
    return {true, selected_count, nullptr};
}

} // namespace edcpp::api::attention::flex::cpu::definition::cpu
