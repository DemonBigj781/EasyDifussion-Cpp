#include <iostream>
#include <string>

#include "core/ggml_extend_backend.h"
#include "stable-diffusion.h"

int main() {
    SDBackendManager manager;
    std::string error;
    if (!manager.init("diffusion=CPU", nullptr, nullptr, &error)) {
        std::cerr << "failed to initialize CPU backend: " << error << std::endl;
        return 1;
    }

    const std::string single_device = sd_format_generation_failure(manager.runtime_backend(SDBackendModule::DIFFUSION),
                                                                    "diffusion",
                                                                    "CPU",
                                                                    "compute_buffer_allocation",
                                                                    "unet",
                                                                    123456);
    if (single_device.find("cause=compute_buffer_allocation") == std::string::npos ||
        single_device.find("module=diffusion") == std::string::npos ||
        single_device.find("requested_bytes=123456") == std::string::npos ||
        single_device.find("selector=CPU") == std::string::npos ||
        single_device.find("assignment=CPU") == std::string::npos) {
        std::cerr << "single-device failure formatting is incomplete: " << single_device << std::endl;
        return 1;
    }

    const std::string split_device = sd_format_generation_failure(nullptr,
                                                                  "diffusion",
                                                                  "Vulkan0&Vulkan1&Vulkan2&Vulkan3",
                                                                  "scheduled_graph_allocation",
                                                                  "unet",
                                                                  0);
    if (split_device.find("assignment=Vulkan0&Vulkan1&Vulkan2&Vulkan3") == std::string::npos ||
        split_device.find("Vulkan1 is invalid") != std::string::npos ||
        split_device.find("failed_member=") != std::string::npos) {
        std::cerr << "split-device failure formatting changed the assignment semantics: " << split_device << std::endl;
        return 1;
    }

    const std::string identified_member = sd_format_generation_failure(nullptr,
                                                                        "diffusion",
                                                                        "Vulkan0&Vulkan1&Vulkan2&Vulkan3",
                                                                        "device_memory_allocation",
                                                                        "unet",
                                                                        4096,
                                                                        "Vulkan2");
    if (identified_member.find("failed_member=Vulkan2") == std::string::npos) {
        std::cerr << "explicit split failure member was not retained: " << identified_member << std::endl;
        return 1;
    }

    sd_set_last_error(single_device);
    if (std::string(sd_get_last_error()) != single_device) {
        std::cerr << "last-error API did not return the recorded failure" << std::endl;
        return 1;
    }
    sd_clear_last_error();
    if (std::string(sd_get_last_error()) != "") {
        std::cerr << "last-error API did not clear the failure" << std::endl;
        return 1;
    }

    std::cout << "validated generation failure formatting" << std::endl;
    return 0;
}
