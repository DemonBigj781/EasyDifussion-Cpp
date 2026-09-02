#include "api/vulkan_handler.hpp"
#include "features/detect/common/detect.hpp"
#include "features/detect/vulkan/translation/detect.hpp"
#include <utility>
#include <vector>

namespace easyapi {
namespace {
edcpp::api::detect::Result detect_vulkan() {
    std::vector<edcpp::api::detect::Result> parts;
    parts.push_back(edcpp::api::detect::vulkan::translation::cpu::detect());
    parts.push_back(edcpp::api::detect::vulkan::translation::gpu::detect());
    return edcpp::api::detect::combine(
        edcpp::api::Backend::vulkan, "vulkan", std::move(parts));
}
} // namespace
const char* VulkanHandler::name() const noexcept { return "vulkan"; }
bool VulkanHandler::available() const noexcept {
    try { return detect_vulkan().backend_available; } catch (...) { return false; }
}
std::vector<DeviceInfo> VulkanHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_vulkan();
    if (!detection.backend_available) return result;
    result.reserve(detection.devices.size());
    for (auto& detected : detection.devices) {
        DeviceInfo info;
        info.index = static_cast<int>(detected.index);
        info.name = std::move(detected.name);
        info.backend = detection.backend_name;
        info.architecture = std::move(detected.architecture);
        info.total_memory = static_cast<std::size_t>(detected.total_memory);
        info.free_memory = static_cast<std::size_t>(detected.free_memory);
        info.available = detected.available;
        result.push_back(std::move(info));
    }
    return result;
}
} // namespace easyapi
