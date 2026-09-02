#include "api/overflow.hpp"
#include "features/overflow/common/overflow.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool validate_cross_backend_plan() {
    using namespace edcpp::api;
    using namespace edcpp::api::overflow;

    PlanInput input;
    input.active_backend = Backend::cuda;
    input.active_device_index = 0;
    input.secondary_devices = {
        {Backend::rocm, 7, 2ull * 1024ull * 1024ull * 1024ull},
        {Backend::cuda, 3, 4ull * 1024ull * 1024ull * 1024ull},
    };
    input.managed_memory_available = true;
    input.mapped_host_available = true;

    const auto candidates = plan(input, 1024ull * 1024ull * 1024ull);
    return candidates.size() == 5 &&
           candidates[0].backend == Backend::cuda &&
           candidates[0].tier == Tier::active_vram &&
           candidates[1].backend == Backend::cuda &&
           candidates[1].device_index == 3 &&
           candidates[2].backend == Backend::rocm &&
           candidates[2].device_index == 7 &&
           candidates[3].kind == CandidateKind::managed &&
           candidates[4].kind == CandidateKind::mapped_host;
}

} // namespace

int main() {
    if (!validate_cross_backend_plan()) {
        std::cerr << "Overflow plan did not preserve cross-backend fallback order\n";
        return EXIT_FAILURE;
    }

    easyapi::CpuOverflowHandler handler;
    if (std::string_view(handler.name()) != "cpu") {
        std::cerr << "CPU Overflow handler returned the wrong name\n";
        return EXIT_FAILURE;
    }

    const auto empty = handler.allocate({});
    if (empty.success || empty.diagnostic.empty()) {
        std::cerr << "CPU Overflow accepted an empty request\n";
        return EXIT_FAILURE;
    }

    std::vector<unsigned char> source(4 * 1024 * 1024);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<unsigned char>((index * 37u + 11u) & 0xffu);
    }

    easyapi::OverflowRequest request;
    request.data = source.data();
    request.size = source.size();
    request.host_reserve = 0;
    auto allocated = handler.allocate(request);
    if (!allocated.success || !allocated.overflowed ||
        !allocated.resource.loaded() || allocated.resource.gpu_accessible() ||
        allocated.resource.tier != easyapi::OverflowTier::system_ram ||
        allocated.resource.storage != easyapi::OverflowStorage::cpu_heap ||
        allocated.resource.size != source.size()) {
        std::cerr << "CPU Overflow allocation failed: "
                  << allocated.diagnostic << '\n';
        return EXIT_FAILURE;
    }

    const auto* copied =
        static_cast<const unsigned char*>(allocated.resource.host_handle);
    if (!std::equal(source.begin(), source.end(), copied)) {
        std::cerr << "CPU Overflow did not preserve the payload\n";
        handler.release(allocated.resource);
        return EXIT_FAILURE;
    }

    const auto original_storage = allocated.resource.storage;
    allocated.resource.storage = easyapi::OverflowStorage::cuda_managed;
    const auto mismatch = handler.release(allocated.resource);
    if (mismatch.success || mismatch.diagnostic.empty()) {
        std::cerr << "CPU Overflow release accepted foreign ownership\n";
        return EXIT_FAILURE;
    }
    allocated.resource.storage = original_storage;

    const auto released = handler.release(allocated.resource);
    if (!released.success || allocated.resource.loaded()) {
        std::cerr << "CPU Overflow release failed: " << released.diagnostic << '\n';
        return EXIT_FAILURE;
    }
    const auto duplicate = handler.release(allocated.resource);
    if (duplicate.success || duplicate.diagnostic.empty()) {
        std::cerr << "CPU Overflow accepted a duplicate release\n";
        return EXIT_FAILURE;
    }

    std::cout << "CPU Overflow allocation/release and mixed-backend plan passed\n";
    return EXIT_SUCCESS;
}
