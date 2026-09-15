#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "stable-diffusion.h"

int main() {
    const size_t count = sd_get_backend_device_count();
    if (count == 0) {
        std::cerr << "no backend devices were registered" << std::endl;
        return 1;
    }

    std::set<std::string> selectors;
    bool found_cpu = false;
    for (size_t i = 0; i < count; ++i) {
        sd_backend_device_info_t info{};
        if (!sd_get_backend_device_info(i, &info)) {
            std::cerr << "could not query backend device " << i << std::endl;
            return 1;
        }
        if (info.index != i || info.backend == nullptr || info.backend[0] == '\0' ||
            info.name == nullptr || info.name[0] == '\0' || info.description == nullptr ||
            info.vendor == nullptr || info.device_id == nullptr) {
            std::cerr << "incomplete metadata for backend device " << i << std::endl;
            return 1;
        }
        if (!selectors.insert(info.name).second) {
            std::cerr << "duplicate backend selector: " << info.name << std::endl;
            return 1;
        }
        if (info.memory_total > 0 && info.memory_free > info.memory_total) {
            std::cerr << "invalid memory accounting for " << info.name << std::endl;
            return 1;
        }
        found_cpu = found_cpu || info.type == SD_BACKEND_DEVICE_TYPE_CPU;
    }
    if (!found_cpu) {
        std::cerr << "CPU fallback backend is missing" << std::endl;
        return 1;
    }

    if (sd_get_backend_device_info(count, nullptr)) {
        std::cerr << "out-of-range device query unexpectedly succeeded" << std::endl;
        return 1;
    }

    const size_t list_size = sd_list_devices(nullptr, 0);
    std::vector<char> device_list(list_size + 1);
    sd_list_devices(device_list.data(), device_list.size());
    for (const std::string& selector : selectors) {
        if (std::strstr(device_list.data(), selector.c_str()) == nullptr) {
            std::cerr << "selector missing from legacy device list: " << selector << std::endl;
            return 1;
        }
    }

    std::cout << "validated " << count << " backend device(s)" << std::endl;
    return 0;
}
