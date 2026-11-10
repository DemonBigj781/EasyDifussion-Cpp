#include "features/load/common/devices/load.hpp"
#include "features/unload/common/devices/unload.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace edcpp::api;
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    try {
        const std::array<unsigned char, 6> bytes{0, 1, 2, 127, 254, 255};
        load::Request request{bytes.data(), bytes.size(), 0, load::ResourceType::model};
        require(!load::register_translation(Backend::none, load::ResourceType::model, nullptr), "invalid registration");
        require(!unload::register_translation(Backend::cpu, load::ResourceType::count, nullptr), "invalid unload registration");
        for (auto backend : {Backend::none, Backend::cuda, Backend::oneapi,
                             static_cast<Backend>(255)}) {
            auto result = load::load_resource(backend, request);
            require(!result.loaded && !result.diagnostic.empty(), "unregistered backend accepted");
        }
        for (int kind = 1; kind <= static_cast<int>(load::ResourceType::count); ++kind) {
            auto typed = request;
            typed.resource_type = static_cast<load::ResourceType>(kind);
            auto result = load::load_resource(Backend::cpu, typed);
            require(!result.loaded && !result.resource.loaded(), "typed request borrowed model route");
        }
        for (int cycle = 0; cycle < 32; ++cycle) {
            auto loaded = load::load_resource(Backend::cpu, request);
            require(loaded.loaded && loaded.resource.loaded(), "CPU load failed");
            require(loaded.resource.native_handle != bytes.data(), "bytes were borrowed instead of copied");
            require(loaded.resource.backend == Backend::cpu && loaded.resource.device_index == 0, "wrong owner");
            require(loaded.resource.size == bytes.size(), "wrong allocation size");
            require(std::memcmp(loaded.resource.native_handle, bytes.data(), bytes.size()) == 0, "bytes changed");
            loaded.resource.resource_type = load::ResourceType::mask;
            auto rejected = unload::unload_resource(loaded.resource);
            require(!rejected.unloaded && loaded.resource.loaded(), "typed unload lost ownership");
            loaded.resource.resource_type = load::ResourceType::model;
            require(unload::unload_resource(loaded.resource).unloaded, "CPU unload failed");
            require(!loaded.resource.loaded() && loaded.resource.native_handle == nullptr, "resource not cleared");
            auto twice = unload::unload_resource(loaded.resource);
            require(!twice.unloaded && !twice.diagnostic.empty(), "double unload accepted");
        }
        auto empty = load::load_resource(Backend::cpu, {});
        require(!empty.loaded && !empty.resource.loaded() && !empty.diagnostic.empty(), "empty bytes accepted");
        std::cout << "CPU typed byte lifecycle: 32 cycles and unsupported-route checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
