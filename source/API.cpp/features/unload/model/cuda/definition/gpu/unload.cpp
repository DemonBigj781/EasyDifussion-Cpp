#include "unload.hpp"

#ifndef EDCPP_UNLOAD_CUDA
#define EDCPP_UNLOAD_CUDA 0
#endif

#if EDCPP_UNLOAD_CUDA
#include <cuda_runtime_api.h>
#endif

namespace edcpp::api::unload::model::cuda::definition::gpu {

NativeResult unload(void* handle, int device_index) noexcept {
    NativeResult result;
    if (handle == nullptr) {
        result.diagnostic = "CUDA model unload received an empty handle";
        return result;
    }
    if (device_index < 0) {
        result.diagnostic = "CUDA model unload requires a valid device index";
        return result;
    }

#if EDCPP_UNLOAD_CUDA
    int previous_device = -1;
    const bool have_previous_device = cudaGetDevice(&previous_device) == cudaSuccess;
    if (!have_previous_device) {
        cudaGetLastError();
    }

    const cudaError_t set_rc = cudaSetDevice(device_index);
    if (set_rc != cudaSuccess) {
        result.diagnostic = cudaGetErrorString(set_rc);
        cudaGetLastError();
        return result;
    }

    const cudaError_t free_rc = cudaFree(handle);
    if (free_rc == cudaSuccess) {
        result.unloaded = true;
    } else {
        result.diagnostic = cudaGetErrorString(free_rc);
        cudaGetLastError();
    }

    if (have_previous_device && previous_device >= 0 && previous_device != device_index) {
        if (cudaSetDevice(previous_device) != cudaSuccess) {
            cudaGetLastError();
        }
    }
#else
    (void) device_index;
    result.diagnostic = "CUDA model unloading was not enabled for this compiler target";
#endif

    return result;
}

} // namespace edcpp::api::unload::model::cuda::definition::gpu
