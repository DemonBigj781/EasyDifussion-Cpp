#include "api/mesa_handler.hpp"
#include "features/detect/common/detect.hpp"
#include <utility>
namespace easyapi {
namespace {
edcpp::api::detect::Result detect_mesa() {
    return edcpp::api::detect::detect(edcpp::api::Backend::mesa);
}
} // namespace
const char* MesaHandler::name() const noexcept { return "mesa"; }
bool MesaHandler::available() const noexcept {
    try { return detect_mesa().backend_available; } catch (...) { return false; }
}
std::vector<DeviceInfo> MesaHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_mesa();
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
