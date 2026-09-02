#include "api/cpu_handler.hpp"

#include "features/detect/common/detect.hpp"

#include <utility>

namespace easyapi {
namespace {

edcpp::api::detect::Result detect_cpu() {
    return edcpp::api::detect::detect(edcpp::api::Backend::cpu);
}

} // namespace

const char* CpuHandler::name() const noexcept {
    return "cpu";
}

bool CpuHandler::available() const noexcept {
    try {
        return detect_cpu().backend_available;
    } catch (...) {
        return false;
    }
}

std::vector<DeviceInfo> CpuHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_cpu();
    if (!detection.backend_available) {
        return result;
    }

    result.reserve(detection.devices.size());
    for (auto& detected : detection.devices) {
        DeviceInfo info;
        info.index = static_cast<int>(detected.index);
        info.name = std::move(detected.name);
        info.backend = detection.backend_name;
        info.architecture = std::move(detected.architecture);
        info.compute_major = detected.compute_major;
        info.compute_minor = detected.compute_minor;
        info.total_memory = static_cast<std::size_t>(detected.total_memory);
        info.free_memory = static_cast<std::size_t>(detected.free_memory);
        info.available = detected.available;
        result.push_back(std::move(info));
    }
    return result;
}

} // namespace easyapi
