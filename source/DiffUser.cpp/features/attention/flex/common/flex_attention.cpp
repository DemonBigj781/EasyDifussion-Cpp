#include "flex_attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace edcpp::api::attention::flex {
namespace {

constexpr std::size_t backend_slots =
    static_cast<std::size_t>(Backend::directml) + 1;

std::array<const Translation*, backend_slots>& registry() noexcept {
    static std::array<const Translation*, backend_slots> translations{};
    return translations;
}

Result validate_common(const Request& request) noexcept {
    if (request.attention == nullptr || request.mask == nullptr) {
        return {false, 0, "Flex Attention requires input and output buffers"};
    }
    if (request.head_count <= 0 || request.query_count <= 0 ||
        request.config.low_grid_size <= 0 ||
        request.config.high_grid_size <= 0 ||
        request.config.pool_grid_size <= 0 ||
        !std::isfinite(request.config.threshold)) {
        return {false, 0, "Flex Attention dimensions and threshold are invalid"};
    }
    const std::size_t required =
        mask_size(request.query_count, request.config.high_grid_size);
    if (required == 0 || request.capacity < required) {
        return {false, 0, "Flex Attention output capacity is insufficient"};
    }
    return {true, 0, nullptr};
}

} // namespace

std::size_t SelectionResult::selected_count() const noexcept {
    return static_cast<std::size_t>(
        std::count(mask.begin(), mask.end(), std::uint8_t{1}));
}

bool register_translation(const Translation* translation) noexcept {
    if (translation == nullptr || translation->backend == Backend::none ||
        translation->validate == nullptr || translation->select == nullptr) {
        return false;
    }
    const auto index = static_cast<std::size_t>(translation->backend);
    if (index >= backend_slots) return false;
    registry()[index] = translation;
    return true;
}

const Translation* translation_for(Backend backend) noexcept {
    if (backend == Backend::none) return nullptr;
    const auto index = static_cast<std::size_t>(backend);
    return index < backend_slots ? registry()[index] : nullptr;
}

Capabilities capabilities(Backend backend) noexcept {
    const Translation* translation = translation_for(backend);
    return translation == nullptr
        ? Capabilities{}
        : translation->capabilities;
}

std::size_t mask_size(
    int query_count, int high_grid_size) noexcept {
    if (query_count <= 0 || high_grid_size <= 0) return 0;
    const auto queries = static_cast<std::size_t>(query_count);
    const auto grid = static_cast<std::size_t>(high_grid_size);
    if (grid > std::numeric_limits<std::size_t>::max() / grid) return 0;
    const std::size_t tokens = grid * grid;
    if (queries > std::numeric_limits<std::size_t>::max() / tokens) return 0;
    return queries * tokens;
}

Result validate(Backend backend, const Request& request) noexcept {
    const Result common = validate_common(request);
    if (!common.ok) return common;
    const Translation* translation = translation_for(backend);
    if (translation == nullptr) {
        return {false, 0, "Flex Attention backend is not registered"};
    }
    return translation->validate(request);
}

Result select(Backend backend, const Request& request) {
    const Result checked = validate(backend, request);
    if (!checked.ok) return checked;
    return translation_for(backend)->select(request);
}

SelectionResult select_high_resolution_tokens(
    Backend backend, const float* attention, int head_count,
    int query_count, const SelectionConfig& config) {
    SelectionResult result;
    const std::size_t required =
        mask_size(query_count, config.high_grid_size);
    if (required == 0) return result;
    result.mask.assign(required, 0);
    Request request{
        attention, head_count, query_count, config,
        result.mask.data(), result.mask.size()};
    const Result selected = select(backend, request);
    if (!selected.ok) {
        result.mask.clear();
        return result;
    }
    result.query_count = query_count;
    result.high_grid_size = config.high_grid_size;
    return result;
}

SelectionResult select_high_resolution_tokens(
    const float* attention, int head_count, int query_count,
    const SelectionConfig& config) {
    return select_high_resolution_tokens(
        Backend::cpu, attention, head_count, query_count, config);
}

bool select(
    const float* attention, int head_count, int query_count,
    const SelectionConfig& config, std::uint8_t* mask,
    std::size_t capacity) {
    return select(
        Backend::cpu,
        {attention, head_count, query_count, config, mask, capacity}).ok;
}

} // namespace edcpp::api::attention::flex
