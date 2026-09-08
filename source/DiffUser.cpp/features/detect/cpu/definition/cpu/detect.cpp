#include "detect.hpp"

#include <cstdint>
#include <string>
#include <utility>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace edcpp::api::detect::cpu::definition::cpu {
namespace {

const char* target_architecture() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__powerpc64__)
    return "powerpc64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    return "riscv64";
#elif defined(__s390x__)
    return "s390x";
#else
    return "unknown";
#endif
}

void read_memory(std::uint64_t& free_memory, std::uint64_t& total_memory) noexcept {
#if defined(__linux__)
    struct sysinfo memory_info {};
    if (::sysinfo(&memory_info) == 0) {
        const auto unit = static_cast<std::uint64_t>(memory_info.mem_unit);
        total_memory = static_cast<std::uint64_t>(memory_info.totalram) * unit;
        free_memory = static_cast<std::uint64_t>(memory_info.freeram) * unit;
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX memory_info {};
    memory_info.dwLength = sizeof(memory_info);
    if (GlobalMemoryStatusEx(&memory_info) != 0) {
        total_memory = static_cast<std::uint64_t>(memory_info.ullTotalPhys);
        free_memory = static_cast<std::uint64_t>(memory_info.ullAvailPhys);
    }
#elif defined(__APPLE__)
    std::uint64_t memory_size = 0;
    std::size_t memory_size_length = sizeof(memory_size);
    if (::sysctlbyname("hw.memsize", &memory_size, &memory_size_length, nullptr, 0) == 0) {
        total_memory = memory_size;
    }
#else
    (void) free_memory;
    (void) total_memory;
#endif
}

} // namespace

NativeResult detect() {
    NativeResult result;
    NativeDevice device;
    device.name = std::string("Host CPU (") + target_architecture() + ')';
    device.architecture = target_architecture();
    device.available = true;
    read_memory(device.free_memory, device.total_memory);

    result.available = true;
    result.devices.push_back(std::move(device));
    return result;
}

} // namespace edcpp::api::detect::cpu::definition::cpu
