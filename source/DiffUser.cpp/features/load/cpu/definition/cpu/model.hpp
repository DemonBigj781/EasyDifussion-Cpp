#pragma once

#include <cstdint>
#include <string>

namespace edcpp::api::load::cpu::definition::cpu {

struct NativeResult {
    bool loaded = false;
    void* handle = nullptr;
    std::uint64_t size = 0;
    std::uint64_t allocation_id = 0;
    std::string diagnostic{};
};

NativeResult load_model(const void* data, std::uint64_t size, std::uint64_t max_bytes);

bool discard_model(void* handle, std::uint64_t allocation_id) noexcept;

} // namespace edcpp::api::load::cpu::definition::cpu
