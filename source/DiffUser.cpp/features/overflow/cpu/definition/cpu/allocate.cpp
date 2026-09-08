#include "allocate.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace edcpp::api::overflow::cpu::definition::cpu {
namespace {

std::uint64_t available_host_memory() noexcept {
#if defined(__linux__)
    struct sysinfo info {};
    if (::sysinfo(&info) == 0) {
        const auto unit = static_cast<std::uint64_t>(info.mem_unit);
        return (static_cast<std::uint64_t>(info.freeram) +
                static_cast<std::uint64_t>(info.bufferram)) * unit;
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX info {};
    info.dwLength = sizeof(info);
    if (GlobalMemoryStatusEx(&info) != 0) {
        return static_cast<std::uint64_t>(info.ullAvailPhys);
    }
#endif
    return 0;
}

bool has_headroom(
    std::uint64_t available, std::uint64_t size, std::uint64_t reserve) noexcept {
    return available == 0 ||
           (size <= available && reserve <= available - size);
}

} // namespace

NativeAllocation allocate(
    const void* data, std::uint64_t size, std::uint64_t host_reserve) {
    NativeAllocation result;
    if (size == 0 || size > std::numeric_limits<std::size_t>::max()) {
        result.diagnostic = "CPU overflow allocation requires a representable non-zero size";
        return result;
    }

    if (!has_headroom(available_host_memory(), size, host_reserve)) {
        result.diagnostic = "CPU overflow allocation would violate the host-memory reserve";
        return result;
    }

    void* allocation = std::malloc(static_cast<std::size_t>(size));
    if (allocation == nullptr) {
        result.diagnostic = "CPU overflow allocation failed";
        return result;
    }
    if (data != nullptr) {
        std::memcpy(allocation, data, static_cast<std::size_t>(size));
    }

    result.allocated = true;
    result.handle = allocation;
    result.size = size;
    return result;
}

} // namespace edcpp::api::overflow::cpu::definition::cpu
