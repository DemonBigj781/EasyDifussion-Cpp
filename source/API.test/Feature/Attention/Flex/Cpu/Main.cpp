#include "features/attention/flex/common/flex_attention.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace flex = edcpp::api::attention::flex;
using edcpp::api::Backend;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) {
        std::cout << "[PASS] " << message << '\n';
        return;
    }
    std::cerr << "[FAIL] " << message << '\n';
    ++failures;
}

bool expected_bottom_right(const std::vector<std::uint8_t>& mask) {
    if (mask.size() != 16) return false;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        const std::size_t y = index / 4;
        const std::size_t x = index % 4;
        const std::uint8_t expected = y >= 2 && x >= 2 ? 1 : 0;
        if (mask[index] != expected) return false;
    }
    return true;
}

} // namespace

int main() {
    const flex::Capabilities available = flex::capabilities(Backend::cpu);
    expect(available.select && available.margin && available.normalize &&
               available.threshold && available.adaptive_pool &&
               available.expand,
           "CPU Flex translation registers complete selection capabilities");

    flex::SelectionConfig config;
    config.low_grid_size = 2;
    config.high_grid_size = 4;
    config.pool_grid_size = 2;
    config.threshold = 64.0f;

    const std::array<float, 4> attention{0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<std::uint8_t> mask(
        flex::mask_size(1, config.high_grid_size), 0);
    const flex::Request request{
        attention.data(), 1, 1, config, mask.data(), mask.size()};

    expect(flex::validate(Backend::cpu, request).ok,
           "Common validation accepts a deterministic CPU Flex request");
    const flex::Result selected = flex::select(Backend::cpu, request);
    expect(selected.ok, "Common dispatch executes CPU Flex selection");
    expect(selected.selected_count == 4 && expected_bottom_right(mask),
           "CPU Flex expands the selected low-resolution token deterministically");

    const flex::SelectionResult owned = flex::select_high_resolution_tokens(
        attention.data(), 1, 1, config);
    expect(owned.query_count == 1 && owned.high_grid_size == 4 &&
               owned.selected_count() == 4 &&
               expected_bottom_right(owned.mask),
           "Compatibility result wrapper routes through Common");

    std::fill(mask.begin(), mask.end(), 0);
    expect(flex::select(
               attention.data(), 1, 1, config, mask.data(), mask.size()) &&
               expected_bottom_right(mask),
           "Compatibility buffer wrapper routes through Common");

    flex::Request undersized = request;
    undersized.capacity = mask.size() - 1;
    expect(!flex::validate(Backend::cpu, undersized).ok,
           "Common validation rejects insufficient output capacity");
    expect(!flex::validate(Backend::cuda, request).ok,
           "Common validation rejects an unregistered Flex backend");

    if (failures != 0) {
        std::cerr << failures << " CPU Flex test(s) failed\n";
        return 1;
    }
    std::cout << "CPU Flex Attention Common route passed\n";
    return 0;
}
