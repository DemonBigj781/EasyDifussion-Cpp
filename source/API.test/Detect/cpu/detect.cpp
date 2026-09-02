#include "api/cpu_handler.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

int main() {
    easyapi::CpuHandler handler;
    if (std::string_view(handler.name()) != "cpu") {
        std::cerr << "CPU handler returned the wrong backend name\n";
        return EXIT_FAILURE;
    }
    if (!handler.available()) {
        std::cerr << "CPU backend is unavailable\n";
        return EXIT_FAILURE;
    }

    const auto devices = handler.devices();
    if (devices.empty()) {
        std::cerr << "CPU backend reported available without a device\n";
        return EXIT_FAILURE;
    }

    for (const auto& device : devices) {
        if (!device.available || device.name.empty() || device.backend != "cpu" ||
            device.architecture.empty()) {
            std::cerr << "CPU detector returned an invalid device record\n";
            return EXIT_FAILURE;
        }
        if (device.total_memory != 0 && device.free_memory > device.total_memory) {
            std::cerr << "CPU detector reported free memory above total memory\n";
            return EXIT_FAILURE;
        }

        std::cout << "cpu[" << device.index << "] " << device.name
                  << " arch=" << device.architecture
                  << " memory=" << (device.free_memory / (1024 * 1024)) << '/'
                  << (device.total_memory / (1024 * 1024)) << " MiB\n";
    }

    return EXIT_SUCCESS;
}
