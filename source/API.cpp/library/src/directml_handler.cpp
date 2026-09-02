#include "api/directml_handler.hpp"
#include "features/detect/common/detect.hpp"
#include "features/detect/directml/translation/gpu/detect.hpp"
#include <utility>
namespace easyapi {
namespace {
edcpp::api::detect::Result detect_directml() {
    return edcpp::api::detect::normalize(
        edcpp::api::detect::directml::translation::gpu::detect());
}
} // namespace
const char* DirectmlHandler::name() const noexcept { return "directml"; }
bool DirectmlHandler::available() const noexcept {
    try { return detect_directml().backend_available; } catch (...) { return false; }
}
std::vector<DeviceInfo> DirectmlHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_directml();
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
