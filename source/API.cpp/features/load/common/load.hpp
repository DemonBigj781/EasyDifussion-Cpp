#pragma once

#include "common/api.hpp"

#include <cstdint>
#include <string>

namespace edcpp::api::load {

struct Request {
    const void* data = nullptr;
    std::uint64_t size = 0;
    int device_index = 0;
};

struct Resource {
    Backend backend = Backend::none;
    void* native_handle = nullptr;
    std::uint64_t size = 0;
    int device_index = -1;

    bool loaded() const noexcept {
        return backend != Backend::none && native_handle != nullptr && size != 0;
    }
};

struct Result {
    bool loaded = false;
    Resource resource;
    std::string diagnostic;
};

Result normalize(Result result);

} // namespace edcpp::api::load
