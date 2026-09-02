#include "api/openvino_handler.hpp"
#include "features/detect/common/detect.hpp"
#include "features/detect/openvino/translation/detect.hpp"
#include <utility>
#include <vector>

namespace easyapi {
namespace {
edcpp::api::detect::Result detect_openvino() {
    std::vector<edcpp::api::detect::Result> parts;
    parts.push_back(edcpp::api::detect::openvino::translation::cpu::detect());
    parts.push_back(edcpp::api::detect::openvino::translation::gpu::detect());
    parts.push_back(edcpp::api::detect::openvino::translation::npu::detect());
    return edcpp::api::detect::combine(
        edcpp::api::Backend::openvino, "openvino", std::move(parts));
}
} // namespace

const char* OpenvinoHandler::name() const noexcept { return "openvino"; }
bool OpenvinoHandler::available() const noexcept {
    try { return detect_openvino().backend_available; } catch (...) { return false; }
}
std::vector<DeviceInfo> OpenvinoHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_openvino();
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
