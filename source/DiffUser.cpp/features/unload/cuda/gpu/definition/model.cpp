#include "features/unload/cuda/gpu/definition/model.hpp"

#ifndef EDCPP_UNLOAD_CUDA
#define EDCPP_UNLOAD_CUDA 0
#endif
#if EDCPP_UNLOAD_CUDA
#include <cuda_runtime_api.h>
#endif

namespace edcpp::api::unload::cuda::gpu::definition {

NativeResult unload_model(void* handle, int device_index) noexcept {
    if (handle == nullptr) return {false, "CUDA model unload received an empty handle"};
    if (device_index < 0) return {false, "CUDA model unload requires a valid device index"};
#if EDCPP_UNLOAD_CUDA
    int previous_device = -1;
    const bool restore = cudaGetDevice(&previous_device) == cudaSuccess;
    if (!restore) cudaGetLastError();
    const cudaError_t set_rc = cudaSetDevice(device_index);
    if (set_rc != cudaSuccess) {
        const char* message = cudaGetErrorString(set_rc);
        cudaGetLastError();
        return {false, message};
    }
    NativeResult result;
    const cudaError_t free_rc = cudaFree(handle);
    if (free_rc == cudaSuccess) result.unloaded = true;
    else {
        result.diagnostic = cudaGetErrorString(free_rc);
        cudaGetLastError();
    }
    if (restore && previous_device >= 0 && previous_device != device_index) {
        if (cudaSetDevice(previous_device) != cudaSuccess) cudaGetLastError();
    }
    return result;
#else
    return {false, "CUDA model unloading was not enabled for this compiler target"};
#endif
}

} // namespace edcpp::api::unload::cuda::gpu::definition
