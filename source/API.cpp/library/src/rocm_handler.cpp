#include "api/rocm_handler.hpp"

#include "features/detect/common/detect.hpp"
#include "features/detect/rocm/translation/gpu/detect.hpp"

#include <utility>

namespace easyapi {
namespace {

edcpp::api::detect::Result detect_rocm() {
    return edcpp::api::detect::normalize(
        edcpp::api::detect::rocm::translation::gpu::detect());
}

} // namespace

const char* RocmHandler::name() const noexcept {
    return "rocm";
}

bool RocmHandler::available() const noexcept {
    try {
        return detect_rocm().backend_available;
    } catch (...) {
        return false;
    }
}

std::vector<DeviceInfo> RocmHandler::devices() const {
    std::vector<DeviceInfo> result;
    auto detection = detect_rocm();
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
