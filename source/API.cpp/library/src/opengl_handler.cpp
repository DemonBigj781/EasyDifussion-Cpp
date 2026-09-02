#include "api/opengl_handler.hpp"
#include "features/detect/common/detect.hpp"
#include "features/detect/opengl/translation/gpu/detect.hpp"
#include <utility>

namespace easyapi {
namespace {
edcpp::api::detect::Result detect_opengl() {
    return edcpp::api::detect::normalize(
        edcpp::api::detect::opengl::translation::gpu::detect());
}
} // namespace
const char* OpenglHandler::name() const noexcept { return "opengl"; }
bool OpenglHandler::available() const noexcept {
    try { return detect_opengl().backend_available; } catch (...) { return false; }
}
std::vector<DeviceInfo> OpenglHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_opengl();
    if (!detection.backend_available) return result;
    result.reserve(detection.devices.size());
    for (auto& detected : detection.devices) {
        DeviceInfo info;
        info.index = static_cast<int>(detected.index);
        info.name = std::move(detected.name);
        info.backend = detection.backend_name;
        info.architecture = std::move(detected.architecture);
        info.available = detected.available;
        result.push_back(std::move(info));
    }
    return result;
}
} // namespace easyapi
