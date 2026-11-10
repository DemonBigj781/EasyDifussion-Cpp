#pragma once

#include <cstddef>
#include <cstdint>

namespace edcpp::api::attention::flex::cpu::definition::cpu {

struct NativeConfig {
    int low_grid_size = 24;
    int high_grid_size = 48;
    int pool_grid_size = 9;
    float threshold = 64.0f;
    int original_width = 0;
    int original_height = 0;
};

struct NativeRequest {
    const float* attention = nullptr;
    int head_count = 0;
    int query_count = 0;
    NativeConfig config{};
    std::uint8_t* mask = nullptr;
    std::size_t capacity = 0;
};

struct NativeCapabilities {
    bool select = false;
    bool margin = false;
    bool normalize = false;
    bool threshold = false;
    bool adaptive_pool = false;
    bool expand = false;
};

struct NativeResult {
    bool ok = false;
    std::size_t selected_count = 0;
    const char* message = nullptr;
};

NativeCapabilities capabilities() noexcept;
NativeResult validate(const NativeRequest& request) noexcept;
NativeResult select(const NativeRequest& request);

} // namespace edcpp::api::attention::flex::cpu::definition::cpu
