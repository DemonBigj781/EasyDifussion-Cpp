#include "api/overflow.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kib = 1024;
constexpr std::size_t mib = 1024 * kib;
constexpr std::size_t gib = 1024 * mib;

__global__ void touch_pages(unsigned char* data, std::size_t size) {
    const std::size_t page_size = 4096;
    const std::size_t page_count = (size + page_size - 1) / page_size;
    std::size_t page = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    while (page < page_count) {
        data[page * page_size] =
            static_cast<unsigned char>((page * 29u + 17u) & 0xffu);
        page += stride;
    }
}

class VramReservations {
public:
    explicit VramReservations(int device_index) : device_index_(device_index) {}

    ~VramReservations() {
        if (cudaSetDevice(device_index_) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        for (void* allocation : allocations_) {
            if (cudaFree(allocation) != cudaSuccess) {
                cudaGetLastError();
            }
        }
    }

    bool fill_until(std::size_t safety_margin, std::size_t failed_request) {
        if (cudaSetDevice(device_index_) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        constexpr std::size_t chunk = 256 * mib;
        for (;;) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            if (free_bytes < failed_request) {
                return free_bytes >= safety_margin;
            }
            if (free_bytes <= safety_margin) {
                return false;
            }

            const std::size_t allocation_size =
                std::min(chunk, free_bytes - safety_margin);
            if (allocation_size == 0) {
                return false;
            }
            void* allocation = nullptr;
            const cudaError_t rc = cudaMalloc(&allocation, allocation_size);
            if (rc != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            allocations_.push_back(allocation);
        }
    }

private:
    int device_index_ = 0;
    std::vector<void*> allocations_;
};

bool verify_gpu_access(easyapi::OverflowResource& resource) {
    if (!resource.gpu_accessible() || resource.device_handle == nullptr ||
        resource.access_device_index < 0) {
        return false;
    }
    if (cudaSetDevice(resource.access_device_index) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const std::size_t page_count = (resource.size + 4095) / 4096;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>((page_count + 255) / 256, 4096));
    touch_pages<<<blocks, 256>>>(
        static_cast<unsigned char*>(resource.device_handle), resource.size);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    if (resource.host_handle != nullptr) {
        const auto* host = static_cast<const unsigned char*>(resource.host_handle);
        for (std::size_t page = 0; page < page_count; ++page) {
            const unsigned char expected =
                static_cast<unsigned char>((page * 29u + 17u) & 0xffu);
            if (host[page * 4096] != expected) {
                return false;
            }
        }
        return true;
    }

    std::vector<unsigned char> sample(1);
    return cudaMemcpy(
               sample.data(), resource.device_handle, sample.size(),
               cudaMemcpyDeviceToHost) == cudaSuccess &&
           sample.front() == 17u;
}

bool run_regular_test(easyapi::CudaOverflowHandler& handler) {
    const auto empty = handler.allocate({});
    if (empty.success || empty.diagnostic.empty()) {
        std::cerr << "CUDA Overflow accepted an empty request\n";
        return false;
    }

    std::vector<unsigned char> payload(4 * mib);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<unsigned char>((index * 13u + 5u) & 0xffu);
    }

    easyapi::OverflowRequest vram_request;
    vram_request.data = payload.data();
    vram_request.size = payload.size();
    auto vram = handler.allocate(vram_request);
    if (!vram.success || vram.overflowed ||
        vram.resource.storage != easyapi::OverflowStorage::cuda_device ||
        vram.resource.tier != easyapi::OverflowTier::active_vram) {
        std::cerr << "CUDA Overflow primary allocation failed: "
                  << vram.diagnostic << '\n';
        return false;
    }
    std::vector<unsigned char> round_trip(payload.size());
    if (cudaMemcpy(
            round_trip.data(), vram.resource.device_handle, round_trip.size(),
            cudaMemcpyDeviceToHost) != cudaSuccess || round_trip != payload) {
        std::cerr << "CUDA Overflow primary allocation corrupted its payload\n";
        handler.release(vram.resource);
        return false;
    }
    if (!handler.release(vram.resource).success) {
        std::cerr << "CUDA Overflow primary release failed\n";
        return false;
    }

    easyapi::OverflowRequest forced_request;
    forced_request.size = 16 * mib;
    forced_request.force_overflow = true;
    forced_request.host_reserve = 0;
    auto fallback = handler.allocate(forced_request);
    if (!fallback.success || !fallback.overflowed ||
        !verify_gpu_access(fallback.resource)) {
        std::cerr << "CUDA Overflow forced fallback failed: "
                  << fallback.diagnostic << '\n';
        if (fallback.success) {
            handler.release(fallback.resource);
        }
        return false;
    }

    std::cout << "forced fallback storage="
              << easyapi::overflow_storage_name(fallback.resource.storage)
              << " tier=" << easyapi::overflow_tier_name(fallback.resource.tier)
              << '\n';
    if (!handler.release(fallback.resource).success) {
        std::cerr << "CUDA Overflow forced fallback release failed\n";
        return false;
    }
    return true;
}

bool run_oom_stress(easyapi::CudaOverflowHandler& handler) {
    constexpr std::size_t failed_request = 2 * gib;
    constexpr std::size_t gpu_safety_margin = 1536 * mib;
    constexpr std::size_t host_safety_margin = 8 * gib;

    VramReservations reservations(0);
    if (!reservations.fill_until(gpu_safety_margin, failed_request)) {
        std::cerr << "Could not create a bounded CUDA OOM condition while preserving "
                     "the GPU safety margin\n";
        return false;
    }

    easyapi::OverflowRequest request;
    request.size = failed_request;
    request.device_index = 0;
    request.host_reserve = host_safety_margin;
    auto fallback = handler.allocate(request);
    if (!fallback.success || !fallback.overflowed ||
        !fallback.primary_out_of_memory ||
        fallback.resource.tier == easyapi::OverflowTier::active_vram) {
        std::cerr << "Real CUDA OOM did not enter Overflow: "
                  << fallback.diagnostic << '\n';
        if (fallback.success) {
            handler.release(fallback.resource);
        }
        return false;
    }
    if (!verify_gpu_access(fallback.resource)) {
        std::cerr << "Overflow memory was not GPU-accessible after real CUDA OOM\n";
        handler.release(fallback.resource);
        return false;
    }

    std::cout << "real cudaMalloc OOM -> "
              << easyapi::overflow_storage_name(fallback.resource.storage)
              << " (" << (fallback.resource.size / mib) << " MiB), GPU verified\n";
    const auto released = handler.release(fallback.resource);
    if (!released.success) {
        std::cerr << "CUDA Overflow stress release failed: "
                  << released.diagnostic << '\n';
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    easyapi::CudaOverflowHandler handler;
    if (std::string_view(handler.name()) != "cuda") {
        std::cerr << "CUDA Overflow handler returned the wrong name\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties {};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        std::cerr << "CUDA device 0 is unavailable\n";
        return EXIT_FAILURE;
    }
    std::cout << properties.name << " sm_" << properties.major << properties.minor
              << " managed=" << properties.managedMemory
              << " concurrent-managed=" << properties.concurrentManagedAccess
              << " mapped-host=" << properties.canMapHostMemory << '\n';

    if (!run_regular_test(handler)) {
        return EXIT_FAILURE;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--stress" &&
        !run_oom_stress(handler)) {
        return EXIT_FAILURE;
    }

    std::cout << "CUDA Overflow tests passed\n";
    return EXIT_SUCCESS;
}
