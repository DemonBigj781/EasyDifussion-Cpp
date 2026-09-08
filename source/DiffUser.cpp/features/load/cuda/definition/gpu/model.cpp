#include "features/load/cuda/definition/gpu/model.hpp"

#ifndef EDCPP_LOAD_CUDA
#define EDCPP_LOAD_CUDA 0
#endif
#if EDCPP_LOAD_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstddef>

namespace edcpp::api::load::cuda::definition::gpu {

NativeResult load_model(const void* data, std::uint64_t size, int device_index) {
    NativeResult result;
    if (data == nullptr || size == 0) {
        result.diagnostic = "CUDA model load requires a non-empty source";
        return result;
    }
    if (device_index < 0) {
        result.diagnostic = "CUDA model load requires a valid device index";
        return result;
    }
#if EDCPP_LOAD_CUDA
    int previous_device = -1;
    const bool restore = cudaGetDevice(&previous_device) == cudaSuccess;
    if (!restore) cudaGetLastError();
    const cudaError_t set_rc = cudaSetDevice(device_index);
    if (set_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(set_rc);
        cudaGetLastError();
        return result;
    }
    void* allocation = nullptr;
    const cudaError_t alloc_rc = cudaMalloc(&allocation, static_cast<std::size_t>(size));
    if (alloc_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(alloc_rc);
        cudaGetLastError();
    } else {
        const cudaError_t copy_rc = cudaMemcpy(allocation, data, static_cast<std::size_t>(size),
                                               cudaMemcpyHostToDevice);
        if (copy_rc != cudaSuccess) {
            result.diagnostic = cudaGetErrorString(copy_rc);
            cudaGetLastError();
            cudaFree(allocation);
        } else {
            result = {true, allocation, size, device_index, {}};
        }
    }
    if (restore && previous_device >= 0 && previous_device != device_index) {
        if (cudaSetDevice(previous_device) != cudaSuccess) cudaGetLastError();
    }
#else
    result.diagnostic = "CUDA model loading was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::load::cuda::definition::gpu
