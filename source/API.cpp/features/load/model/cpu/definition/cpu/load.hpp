#pragma once

#include <cstdint>
#include <string>

namespace edcpp::api::load::model::cpu::definition::cpu {

struct NativeResult {
    bool loaded = false;
    void* handle = nullptr;
    std::uint64_t size = 0;
    std::string diagnostic;
};

NativeResult load(const void* data, std::uint64_t size);

} // namespace edcpp::api::load::model::cpu::definition::cpu
