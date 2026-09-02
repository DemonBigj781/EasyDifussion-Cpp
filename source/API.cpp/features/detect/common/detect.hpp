#pragma once

#include "common/api.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect {

enum class DeviceClass : std::uint8_t { unknown = 0, cpu, gpu, npu };

struct Device {
    Backend backend = Backend::none;
    DeviceClass device_class = DeviceClass::unknown;
    std::uint32_t index = 0;
    std::string name;
    std::string architecture;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_memory = 0;
    std::uint64_t free_memory = 0;
    bool available = false;
};

struct Result {
    Backend backend = Backend::none;
    std::string backend_name;
    bool backend_available = false;
    std::vector<Device> devices;
    std::string diagnostic;
};

using TranslationFn = Result (*)();

// Backend translations register themselves here when linked. Common remains
// backend-neutral and owns all application-facing selection and dispatch.
bool register_translation(Backend backend, TranslationFn translation) noexcept;
Result detect(Backend backend);
Result normalize(Result result);
Result combine(Backend backend, std::string backend_name, std::vector<Result> parts);

} // namespace edcpp::api::detect
