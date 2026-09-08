#include "flex_attention.hpp"

#include <cstring>

namespace edcpp::api::attention::flex {

size_t mask_size(int query_count, int high_grid_size) {
    if (query_count <= 0 || high_grid_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(query_count) * high_grid_size * high_grid_size;
}

bool select(const float* attention,
            int head_count,
            int query_count,
            const SelectionConfig& config,
            uint8_t* mask,
            size_t capacity) {
    if (mask == nullptr) {
        return false;
    }
    const auto result = select_high_resolution_tokens(attention, head_count, query_count, config);
    if (result.mask.empty() || result.mask.size() > capacity) {
        return false;
    }
    std::memcpy(mask, result.mask.data(), result.mask.size());
    return true;
}

}  // namespace edcpp::api::attention::flex
