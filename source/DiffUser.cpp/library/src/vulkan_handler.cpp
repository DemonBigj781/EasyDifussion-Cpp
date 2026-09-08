#include "api/vulkan_handler.hpp"
#include "features/detect/common/detect.hpp"
#include <utility>

namespace easyapi {
namespace {
edcpp::api::detect::Result detect_vulkan() {
    return edcpp::api::detect::detect(edcpp::api::Backend::vulkan);
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
