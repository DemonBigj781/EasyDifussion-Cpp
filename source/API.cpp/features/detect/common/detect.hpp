#pragma once

#include "common/api.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace edcpp::api::detect {

enum class DeviceClass : std::uint8_t {
    unknown = 0,
    cpu,
    gpu,
    npu,
};

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

// Public Common entry point. Callers select a normalized backend identity only;
// Common owns the backend translation route.
Result detect(Backend backend);

// Applies backend-neutral invariants at the Common boundary.
Result normalize(Result result);

// Combines device-class translations belonging to one backend, then applies
// the same Common invariants to the aggregate.
Result combine(Backend backend, std::string backend_name, std::vector<Result> parts);

} // namespace edcpp::api::detect
