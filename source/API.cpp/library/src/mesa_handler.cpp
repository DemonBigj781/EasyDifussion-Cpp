#include "api/mesa_handler.hpp"
#include "features/detect/common/detect.hpp"
#include "features/detect/mesa/translation/detect.hpp"
#include <utility>
#include <vector>
namespace easyapi {
namespace {
edcpp::api::detect::Result detect_mesa() {
    std::vector<edcpp::api::detect::Result> parts;
    parts.push_back(edcpp::api::detect::mesa::translation::cpu::detect());
    parts.push_back(edcpp::api::detect::mesa::translation::gpu::detect());
    return edcpp::api::detect::combine(
        edcpp::api::Backend::mesa, "mesa", std::move(parts));
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
