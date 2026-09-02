#include "api/model_lifecycle.hpp"
#include "model_fixture.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    easyapi::CudaModelLifecycleHandler handler;
    if (std::string_view(handler.name()) != "cuda") {
        std::cerr << "CUDA lifecycle returned the wrong backend name\n";
        return EXIT_FAILURE;
    }

    const auto empty = handler.load(nullptr, 0);
    if (empty.success || empty.diagnostic.empty()) {
        std::cerr << "CUDA load accepted an empty model payload\n";
        return EXIT_FAILURE;
    }

    api_test::ModelFixture fixture;
    try {
        fixture = api_test::read_model_fixture(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::vector<unsigned char> round_trip(fixture.bytes.size());

    auto loaded = handler.load(fixture.bytes.data(), fixture.bytes.size(), 0);
    if (!loaded.success || !loaded.resource.loaded() ||
        loaded.resource.storage != easyapi::ModelStorage::cuda ||
        loaded.resource.size != fixture.bytes.size() ||
        loaded.resource.device_index != 0) {
        std::cerr << "CUDA load failed: " << loaded.diagnostic << '\n';
        return EXIT_FAILURE;
    }

    const cudaError_t copy_rc = cudaMemcpy(
        round_trip.data(), loaded.resource.native_handle, round_trip.size(),
        cudaMemcpyDeviceToHost);
    if (copy_rc != cudaSuccess || round_trip != fixture.bytes) {
        std::cerr << "CUDA load did not preserve the model payload: "
                  << cudaGetErrorString(copy_rc) << '\n';
        handler.unload(loaded.resource);
        return EXIT_FAILURE;
    }

    std::cout << "CUDA loaded and verified " << loaded.resource.size
              << " bytes from " << fixture.path << '\n';
    const auto cleanup = handler.unload(loaded.resource);
    if (!cleanup.success) {
        std::cerr << "CUDA cleanup failed: " << cleanup.diagnostic << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
