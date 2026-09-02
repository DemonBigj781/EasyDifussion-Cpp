#include "api/model_lifecycle.hpp"
#include "model_fixture.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    easyapi::CpuModelLifecycleHandler handler;
    api_test::ModelFixture fixture;
    try {
        fixture = api_test::read_model_fixture(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    auto loaded = handler.load(fixture.bytes.data(), fixture.bytes.size());
    if (!loaded.success) {
        std::cerr << "CPU setup load failed: " << loaded.diagnostic << '\n';
        return EXIT_FAILURE;
    }

    const void* owned_handle = loaded.resource.native_handle;
    loaded.resource.storage = easyapi::ModelStorage::cuda;
    const auto wrong_backend = handler.unload(loaded.resource);
    if (wrong_backend.success || wrong_backend.diagnostic.empty() ||
        loaded.resource.native_handle != owned_handle) {
        std::cerr << "CPU unload did not reject a backend ownership mismatch\n";
        return EXIT_FAILURE;
    }
    loaded.resource.storage = easyapi::ModelStorage::cpu;

    const auto unloaded = handler.unload(loaded.resource);
    if (!unloaded.success || loaded.resource.loaded() ||
        loaded.resource.storage != easyapi::ModelStorage::none ||
        loaded.resource.native_handle != nullptr ||
        loaded.resource.size != 0 ||
        loaded.resource.device_index != -1) {
        std::cerr << "CPU unload failed: " << unloaded.diagnostic << '\n';
        return EXIT_FAILURE;
    }

    const auto second_unload = handler.unload(loaded.resource);
    if (second_unload.success || second_unload.diagnostic.empty()) {
        std::cerr << "CPU unload accepted an already released resource\n";
        return EXIT_FAILURE;
    }

    std::cout << "CPU released " << fixture.bytes.size()
              << " model bytes and rejected mismatch/double-unload\n";
    return EXIT_SUCCESS;
}
