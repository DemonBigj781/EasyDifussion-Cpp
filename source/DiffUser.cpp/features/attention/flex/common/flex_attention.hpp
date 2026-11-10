#pragma once

#include "common/api.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace edcpp::api::attention::flex {

struct SelectionConfig {
    int low_grid_size = 24;
    int high_grid_size = 48;
    int pool_grid_size = 9;
    float threshold = 64.0f;
    int original_width = 0;
    int original_height = 0;
};

struct SelectionResult {
    int query_count = 0;
    int high_grid_size = 0;
    std::vector<std::uint8_t> mask{};

    std::size_t selected_count() const noexcept;
};

struct Request {
    const float* attention = nullptr;
    int head_count = 0;
    int query_count = 0;
    SelectionConfig config{};
    std::uint8_t* mask = nullptr;
    std::size_t capacity = 0;
};

struct Capabilities {
    bool select = false;
    bool margin = false;
    bool normalize = false;
    bool threshold = false;
    bool adaptive_pool = false;
    bool expand = false;
};

struct Result {
    bool ok = false;
    std::size_t selected_count = 0;
    const char* message = nullptr;
};

using ValidateFn = Result (*)(const Request&) noexcept;
using SelectFn = Result (*)(const Request&);

struct Translation {
    Backend backend = Backend::none;
    const char* name = "unknown";
    Capabilities capabilities{};
    ValidateFn validate = nullptr;
    SelectFn select = nullptr;
};

bool register_translation(const Translation* translation) noexcept;
const Translation* translation_for(Backend backend) noexcept;
Capabilities capabilities(Backend backend) noexcept;
std::size_t mask_size(int query_count, int high_grid_size) noexcept;
Result validate(Backend backend, const Request& request) noexcept;
Result select(Backend backend, const Request& request);
SelectionResult select_high_resolution_tokens(
    Backend backend, const float* attention, int head_count,
    int query_count, const SelectionConfig& config);

// Compatibility overloads retain the original CPU-facing surface while
// routing through Common.
SelectionResult select_high_resolution_tokens(
    const float* attention, int head_count, int query_count,
    const SelectionConfig& config);
bool select(
    const float* attention, int head_count, int query_count,
    const SelectionConfig& config, std::uint8_t* mask,
    std::size_t capacity);

} // namespace edcpp::api::attention::flex
