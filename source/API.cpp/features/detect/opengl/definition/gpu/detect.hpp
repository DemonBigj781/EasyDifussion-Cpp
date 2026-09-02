#pragma once
#include <string>
#include <vector>
namespace edcpp::api::detect::opengl::definition::gpu {
struct NativeDevice {
    int index = 0;
    std::string name;
    std::string architecture;
    bool available = false;
};
struct NativeResult {
    bool available = false;
    std::vector<NativeDevice> devices;
    std::string diagnostic;
};
NativeResult detect();
} // namespace edcpp::api::detect::opengl::definition::gpu
