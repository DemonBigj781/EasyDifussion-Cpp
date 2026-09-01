#include "api/cuda_handler.hpp"

#include <cuda_runtime_api.h>

#include <utility>

namespace easyapi {

const char* CudaHandler::name() const noexcept {
    return "cuda";
}

bool CudaHandler::available() const noexcept {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::vector<DeviceInfo> CudaHandler::devices() const {
    std::vector<DeviceInfo> result;

    int count = 0;
    const cudaError_t count_rc = cudaGetDeviceCount(&count);
    if (count_rc != cudaSuccess || count <= 0) {
        cudaGetLastError();
        return result;
    }

    result.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, index) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }

        DeviceInfo info;
        info.index = index;
        info.name = props.name;
        info.backend = "cuda";
        info.compute_major = props.major;
        info.compute_minor = props.minor;
        info.total_memory = props.totalGlobalMem;
        info.available = true;

        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(index) == cudaSuccess) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                info.free_memory = free_bytes;
                info.total_memory = total_bytes;
            } else {
                cudaGetLastError();
            }
        } else {
            cudaGetLastError();
        }
        if (have_previous && previous >= 0 && previous != index) {
            cudaSetDevice(previous);
        }

        result.push_back(std::move(info));
    }

    return result;
}

} // namespace easyapi
