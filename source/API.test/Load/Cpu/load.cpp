#include "api/model_lifecycle.hpp"
#include "model_fixture.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    easyapi::CpuModelLifecycleHandler handler;
    if (std::string_view(handler.name()) != "cpu") {
        std::cerr << "CPU lifecycle returned the wrong backend name\n";
        return EXIT_FAILURE;
    }

    const auto empty = handler.load(nullptr, 0);
    if (empty.success || empty.diagnostic.empty()) {
        std::cerr << "CPU load accepted an empty model payload\n";
        return EXIT_FAILURE;
    }

    api_test::ModelFixture fixture;
    try {
        fixture = api_test::read_model_fixture(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    auto loaded = handler.load(fixture.bytes.data(), fixture.bytes.size());
    if (!loaded.success || !loaded.resource.loaded() ||
        loaded.resource.storage != easyapi::ModelStorage::cpu ||
        loaded.resource.size != fixture.bytes.size() ||
        loaded.resource.device_index != 0) {
        std::cerr << "CPU load failed: " << loaded.diagnostic << '\n';
        return EXIT_FAILURE;
    }
    if (std::memcmp(
            loaded.resource.native_handle,
            fixture.bytes.data(),
            fixture.bytes.size()) != 0) {
        std::cerr << "CPU load did not preserve the model payload\n";
        handler.unload(loaded.resource);
        return EXIT_FAILURE;
    }

    std::cout << "CPU loaded and verified " << loaded.resource.size
              << " bytes from " << fixture.path << '\n';
    const auto cleanup = handler.unload(loaded.resource);
    if (!cleanup.success) {
        std::cerr << "CPU cleanup failed: " << cleanup.diagnostic << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
