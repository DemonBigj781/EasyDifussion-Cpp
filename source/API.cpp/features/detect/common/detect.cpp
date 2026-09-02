#include "detect.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace edcpp::api::detect {
namespace {
constexpr std::size_t backend_slots = static_cast<std::size_t>(Backend::directml) + 1;
std::array<TranslationFn, backend_slots>& translations() {
    static std::array<TranslationFn, backend_slots> table{};
    return table;
}
}

bool register_translation(Backend backend, TranslationFn translation) noexcept {
    const auto index = static_cast<std::size_t>(backend);
    if (backend == Backend::none || index >= backend_slots || translation == nullptr) {
        return false;
    }
    translations()[index] = translation;
    return true;
}

Result detect(Backend backend) {
    const auto index = static_cast<std::size_t>(backend);
    if (backend == Backend::none || index >= backend_slots || translations()[index] == nullptr) {
        Result result;
        result.backend = backend;
        result.backend_name = edcpp::api::backend_name(backend);
        result.diagnostic = "Common Detect has no registered translation for the requested backend";
        return normalize(std::move(result));
    }
    return normalize(translations()[index]());
}

Result normalize(Result result) {
    if (result.backend == Backend::none) {
        result.backend_available = false;
        result.devices.clear();
        if (result.diagnostic.empty()) result.diagnostic = "detect translation did not identify a backend";
        return result;
    }
    if (result.backend_name.empty()) result.backend_name = edcpp::api::backend_name(result.backend);
    for (auto& device : result.devices) {
        device.backend = result.backend;
        if (device.name.empty()) device.available = false;
        if (device.total_memory != 0 && device.free_memory > device.total_memory) device.free_memory = device.total_memory;
    }
    const bool have_available_device = std::any_of(result.devices.begin(), result.devices.end(), [](const Device& device) { return device.available; });
    result.backend_available = result.backend_available && have_available_device;
    if (!result.backend_available && result.diagnostic.empty()) result.diagnostic = "backend reported no available devices";
    return result;
}

Result combine(Backend backend, std::string backend_name, std::vector<Result> parts) {
    Result combined;
    combined.backend = backend;
    combined.backend_name = std::move(backend_name);
    for (auto& part : parts) {
        part = normalize(std::move(part));
        combined.backend_available = combined.backend_available || part.backend_available;
        if (!part.diagnostic.empty()) {
            if (!combined.diagnostic.empty()) combined.diagnostic += "; ";
            combined.diagnostic += part.diagnostic;
        }
        for (auto& device : part.devices) combined.devices.push_back(std::move(device));
    }
    return normalize(std::move(combined));
}

} // namespace edcpp::api::detect
