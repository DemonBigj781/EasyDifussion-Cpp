#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace easyapi {

struct DeviceInfo {
    int index = -1;
    std::string name;
    std::string backend;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_memory = 0;
    std::size_t free_memory = 0;
    bool available = false;
};

class BackendHandler {
public:
    virtual ~BackendHandler() = default;
    virtual const char* name() const noexcept = 0;
    virtual bool available() const noexcept = 0;
    virtual std::vector<DeviceInfo> devices() const = 0;
};

} // namespace easyapi
