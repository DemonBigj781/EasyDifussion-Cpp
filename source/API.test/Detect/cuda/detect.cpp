#include "api/cuda_handler.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

int main() {
    easyapi::CudaHandler handler;
    if (std::string_view(handler.name()) != "cuda") {
        std::cerr << "CUDA handler returned the wrong backend name\n";
        return EXIT_FAILURE;
    }
    if (!handler.available()) {
        std::cerr << "CUDA backend is unavailable\n";
        return EXIT_FAILURE;
    }

    const auto devices = handler.devices();
    if (devices.empty()) {
        std::cerr << "CUDA backend reported available without a device\n";
        return EXIT_FAILURE;
    }

    for (const auto& device : devices) {
        const std::string expected_architecture =
            "sm_" + std::to_string(device.compute_major) +
            std::to_string(device.compute_minor);
        if (!device.available || device.name.empty() || device.backend != "cuda" ||
            device.compute_major <= 0 || device.architecture != expected_architecture) {
            std::cerr << "CUDA detector returned an invalid device record\n";
            return EXIT_FAILURE;
        }
        if (device.total_memory != 0 && device.free_memory > device.total_memory) {
            std::cerr << "CUDA detector reported free memory above total memory\n";
            return EXIT_FAILURE;
        }

        std::cout << "cuda[" << device.index << "] " << device.name
                  << " arch=" << device.architecture
                  << " compute=" << device.compute_major << '.' << device.compute_minor
                  << " memory=" << (device.free_memory / (1024 * 1024)) << '/'
                  << (device.total_memory / (1024 * 1024)) << " MiB\n";
    }

    return EXIT_SUCCESS;
}
