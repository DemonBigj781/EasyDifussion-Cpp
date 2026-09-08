#include "allocate.hpp"

#ifndef EDCPP_OVERFLOW_CUDA
#define EDCPP_OVERFLOW_CUDA 0
#endif

#if EDCPP_OVERFLOW_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace edcpp::api::overflow::cuda::definition::gpu {
namespace {

std::uint64_t available_host_memory() noexcept {
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (meminfo >> key >> kib >> unit) {
        if (key == "MemAvailable:") {
            return kib * 1024ull;
        }
    }

    struct sysinfo info {};
    if (::sysinfo(&info) == 0) {
        return static_cast<std::uint64_t>(info.freeram) *
               static_cast<std::uint64_t>(info.mem_unit);
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

#if EDCPP_OVERFLOW_CUDA
class DeviceRestore {
public:
    DeviceRestore() noexcept {
        valid_ = cudaGetDevice(&device_) == cudaSuccess;
        if (!valid_) {
            cudaGetLastError();
        }
    }

    ~DeviceRestore() {
        if (valid_ && device_ >= 0) {
            if (cudaSetDevice(device_) != cudaSuccess) {
                cudaGetLastError();
            }
        }
    }

private:
    int device_ = -1;
    bool valid_ = false;
};

bool set_device(int device_index, std::string& diagnostic) {
    const cudaError_t rc = cudaSetDevice(device_index);
    if (rc == cudaSuccess) {
        return true;
    }
    diagnostic = cudaGetErrorString(rc);
    cudaGetLastError();
    return false;
}

void record_cuda_failure(
    NativeAllocation& result, cudaError_t error, const char* operation) {
    result.out_of_memory = error == cudaErrorMemoryAllocation;
    result.diagnostic = std::string(operation) + ": " + cudaGetErrorString(error);
    cudaGetLastError();
}
#endif

bool valid_size(std::uint64_t size) noexcept {
    return size != 0 && size <= std::numeric_limits<std::size_t>::max();
}

} // namespace

NativeCapabilities detect_capabilities(int active_device_index) {
    NativeCapabilities result;
    if (active_device_index < 0) {
        result.diagnostic = "CUDA overflow requires a valid active device index";
        return result;
    }

#if EDCPP_OVERFLOW_CUDA
    DeviceRestore restore;
    int count = 0;
    const cudaError_t count_rc = cudaGetDeviceCount(&count);
    if (count_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(count_rc);
        cudaGetLastError();
        return result;
    }
    if (active_device_index >= count) {
        result.diagnostic = "CUDA overflow active device index is out of range";
        return result;
    }

    cudaDeviceProp active {};
    const cudaError_t property_rc =
        cudaGetDeviceProperties(&active, active_device_index);
    if (property_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(property_rc);
        cudaGetLastError();
        return result;
    }

    result.available = true;
    result.compute_major = active.major;
    result.compute_minor = active.minor;
    // Full explicit Managed Memory oversubscription requires concurrent access.
    // This is available on Linux for both Volta (V100) and Ampere (RTX 3060).
    result.managed_memory =
        active.managedMemory != 0 && active.concurrentManagedAccess != 0;
    result.mapped_host_memory = active.canMapHostMemory != 0;

    for (int index = 0; index < count; ++index) {
        if (index == active_device_index) {
            continue;
        }
        if (!set_device(index, result.diagnostic)) {
            continue;
        }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        result.secondary_devices.push_back(
            {index, static_cast<std::uint64_t>(free_bytes),
             static_cast<std::uint64_t>(total_bytes)});
    }
#else
    (void) active_device_index;
    result.diagnostic = "CUDA overflow was not enabled for this compiler target";
#endif
    return result;
}

NativeAllocation allocate_device(
    const void* data, std::uint64_t size, int device_index) {
    NativeAllocation result;
    if (!valid_size(size) || device_index < 0) {
        result.diagnostic = "CUDA VRAM allocation requires a valid device and non-zero size";
        return result;
    }

#if EDCPP_OVERFLOW_CUDA
    DeviceRestore restore;
    if (!set_device(device_index, result.diagnostic)) {
        return result;
    }

    void* allocation = nullptr;
    const cudaError_t allocation_rc =
        cudaMalloc(&allocation, static_cast<std::size_t>(size));
    if (allocation_rc != cudaSuccess) {
        record_cuda_failure(result, allocation_rc, "cudaMalloc");
        return result;
    }
    if (data != nullptr) {
        const cudaError_t copy_rc = cudaMemcpy(
            allocation, data, static_cast<std::size_t>(size),
            cudaMemcpyHostToDevice);
        if (copy_rc != cudaSuccess) {
            record_cuda_failure(result, copy_rc, "cudaMemcpy to VRAM");
            cudaFree(allocation);
            return result;
        }
    }

    result.allocated = true;
    result.memory_kind = NativeMemoryKind::device;
    result.device_handle = allocation;
    result.size = size;
    result.owner_device_index = device_index;
    result.access_device_index = device_index;
#else
    (void) data;
    (void) device_index;
    result.diagnostic = "CUDA overflow was not enabled for this compiler target";
#endif
    return result;
}

NativeAllocation allocate_managed(
    const void* data, std::uint64_t size, int device_index,
    std::uint64_t host_reserve) {
    NativeAllocation result;
    if (!valid_size(size) || device_index < 0) {
        result.diagnostic = "CUDA managed allocation requires a valid device and non-zero size";
        return result;
    }
    if (!has_headroom(available_host_memory(), size, host_reserve)) {
        result.diagnostic = "CUDA managed allocation would violate the host-memory reserve";
        result.out_of_memory = true;
        return result;
    }

#if EDCPP_OVERFLOW_CUDA
    DeviceRestore restore;
    if (!set_device(device_index, result.diagnostic)) {
        return result;
    }

    void* allocation = nullptr;
    const cudaError_t allocation_rc = cudaMallocManaged(
        &allocation, static_cast<std::size_t>(size), cudaMemAttachGlobal);
    if (allocation_rc != cudaSuccess) {
        record_cuda_failure(result, allocation_rc, "cudaMallocManaged");
        return result;
    }

    // Mirror the Windows sysmem-fallback intent: keep the backing pages in
    // system RAM while installing a GPU-accessible mapping when supported.
    const cudaError_t preferred_rc = cudaMemAdvise(
        allocation, static_cast<std::size_t>(size),
        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (preferred_rc != cudaSuccess) {
        cudaGetLastError();
    }
    const cudaError_t accessed_rc = cudaMemAdvise(
        allocation, static_cast<std::size_t>(size),
        cudaMemAdviseSetAccessedBy, device_index);
    if (accessed_rc != cudaSuccess) {
        cudaGetLastError();
    }
    if (data != nullptr) {
        std::memcpy(allocation, data, static_cast<std::size_t>(size));
    }

    result.allocated = true;
    result.memory_kind = NativeMemoryKind::managed;
    result.host_handle = allocation;
    result.device_handle = allocation;
    result.size = size;
    result.access_device_index = device_index;
#else
    (void) data;
    (void) device_index;
    (void) host_reserve;
    result.diagnostic = "CUDA overflow was not enabled for this compiler target";
#endif
    return result;
}

NativeAllocation allocate_mapped_host(
    const void* data, std::uint64_t size, int device_index,
    std::uint64_t host_reserve) {
    NativeAllocation result;
    if (!valid_size(size) || device_index < 0) {
        result.diagnostic = "CUDA mapped-host allocation requires a valid device and non-zero size";
        return result;
    }
    if (!has_headroom(available_host_memory(), size, host_reserve)) {
        result.diagnostic = "CUDA mapped-host allocation would violate the host-memory reserve";
        result.out_of_memory = true;
        return result;
    }

#if EDCPP_OVERFLOW_CUDA
    DeviceRestore restore;
    if (!set_device(device_index, result.diagnostic)) {
        return result;
    }

    void* host_allocation = nullptr;
    const cudaError_t allocation_rc = cudaHostAlloc(
        &host_allocation, static_cast<std::size_t>(size),
        cudaHostAllocMapped | cudaHostAllocPortable);
    if (allocation_rc != cudaSuccess) {
        record_cuda_failure(result, allocation_rc, "cudaHostAlloc(mapped)");
        return result;
    }

    void* device_pointer = nullptr;
    const cudaError_t pointer_rc =
        cudaHostGetDevicePointer(&device_pointer, host_allocation, 0);
    if (pointer_rc != cudaSuccess) {
        record_cuda_failure(result, pointer_rc, "cudaHostGetDevicePointer");
        cudaFreeHost(host_allocation);
        return result;
    }
    if (data != nullptr) {
        std::memcpy(host_allocation, data, static_cast<std::size_t>(size));
    }

    result.allocated = true;
    result.memory_kind = NativeMemoryKind::mapped_host;
    result.host_handle = host_allocation;
    result.device_handle = device_pointer;
    result.size = size;
    result.access_device_index = device_index;
#else
    (void) data;
    (void) device_index;
    (void) host_reserve;
    result.diagnostic = "CUDA overflow was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::overflow::cuda::definition::gpu
