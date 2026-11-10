#include "features/attention/flex/common/flex_attention.hpp"
#include "features/attention/flex/cpu/definition/cpu/flex_attention.hpp"

namespace edcpp::api::attention::flex::cpu::translation::cpu {
namespace {

namespace native =
    edcpp::api::attention::flex::cpu::definition::cpu;

native::NativeRequest translate_request(const Request& request) noexcept {
    native::NativeRequest result;
    result.attention = request.attention;
    result.head_count = request.head_count;
    result.query_count = request.query_count;
    result.config = {
        request.config.low_grid_size,
        request.config.high_grid_size,
        request.config.pool_grid_size,
        request.config.threshold,
        request.config.original_width,
        request.config.original_height};
    result.mask = request.mask;
    result.capacity = request.capacity;
    return result;
}

Result translate_result(native::NativeResult result) noexcept {
    return {result.ok, result.selected_count, result.message};
}

Result validate_cpu(const Request& request) noexcept {
    return translate_result(native::validate(translate_request(request)));
}

Result select_cpu(const Request& request) {
    return translate_result(native::select(translate_request(request)));
}

Capabilities translated_capabilities() noexcept {
    const native::NativeCapabilities source = native::capabilities();
    return {
        source.select,
        source.margin,
        source.normalize,
        source.threshold,
        source.adaptive_pool,
        source.expand};
}

const Translation cpu_translation = {
    Backend::cpu,
    "cpu",
    translated_capabilities(),
    &validate_cpu,
    &select_cpu,
};

[[maybe_unused]] const bool registered =
    register_translation(&cpu_translation);

} // namespace
} // namespace edcpp::api::attention::flex::cpu::translation::cpu
