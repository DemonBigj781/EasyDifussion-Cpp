#include "detect.hpp"

#include <utility>
#include <vector>

namespace edcpp::api::detect::vulkan::translation {
namespace {

Result detect_all() {
    std::vector<Result> parts;
    parts.push_back(cpu::detect());
    parts.push_back(gpu::detect());
    return combine(Backend::vulkan, "vulkan", std::move(parts));
}

[[maybe_unused]] const bool registered =
    register_translation(Backend::vulkan, &detect_all);

} // namespace
} // namespace edcpp::api::detect::vulkan::translation
