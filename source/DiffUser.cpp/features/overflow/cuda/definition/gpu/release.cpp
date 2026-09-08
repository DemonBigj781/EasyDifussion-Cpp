#include "release.hpp"

#ifndef EDCPP_OVERFLOW_CUDA
#define EDCPP_OVERFLOW_CUDA 0
#endif

#if EDCPP_OVERFLOW_CUDA
#include <cuda_runtime_api.h>
#endif

namespace edcpp::api::overflow::cuda::definition::gpu {

NativeRelease release(
    NativeMemoryKind memory_kind, void* host_handle, void* device_handle,
    int owner_device_index) noexcept {
    NativeRelease result;

#if EDCPP_OVERFLOW_CUDA
    cudaError_t rc = cudaErrorInvalidValue;
    switch (memory_kind) {
        case NativeMemoryKind::device: {
            if (device_handle == nullptr || owner_device_index < 0) {
                result.diagnostic = "CUDA VRAM release received an invalid resource";
                return result;
            }
            int previous_device = -1;
            const bool restore = cudaGetDevice(&previous_device) == cudaSuccess;
            if (!restore) {
                cudaGetLastError();
            }
            rc = cudaSetDevice(owner_device_index);
            if (rc == cudaSuccess) {
                rc = cudaFree(device_handle);
            }
            if (restore && previous_device >= 0 && previous_device != owner_device_index) {
                if (cudaSetDevice(previous_device) != cudaSuccess) {
                    cudaGetLastError();
                }
            }
            break;
        }
        case NativeMemoryKind::managed:
            if (host_handle == nullptr) {
                result.diagnostic = "CUDA managed release received an empty handle";
                return result;
            }
            rc = cudaFree(host_handle);
            break;
        case NativeMemoryKind::mapped_host:
            if (host_handle == nullptr) {
                result.diagnostic = "CUDA mapped-host release received an empty handle";
                return result;
            }
            rc = cudaFreeHost(host_handle);
            break;
        case NativeMemoryKind::none:
            result.diagnostic = "CUDA overflow release received no allocation kind";
            return result;
    }

    if (rc == cudaSuccess) {
        result.released = true;
    } else {
        result.diagnostic = cudaGetErrorString(rc);
        cudaGetLastError();
    }
#else
    (void) memory_kind;
    (void) host_handle;
    (void) device_handle;
    (void) owner_device_index;
    result.diagnostic = "CUDA overflow was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::overflow::cuda::definition::gpu
