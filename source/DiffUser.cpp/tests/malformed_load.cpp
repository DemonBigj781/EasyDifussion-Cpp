#include "features/load/common/devices/load.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace edcpp::api;
static int mode = 0, allocations = 0, rollbacks = 0;
static load::Result produce(const load::Request& request) {
    load::Result result;
    result.loaded = true;
    auto* pointer = std::malloc(request.size);
    if (!pointer) throw std::bad_alloc();
    ++allocations;
    std::memcpy(pointer, request.data, request.size);
    result.resource = {Backend::mesa, pointer, request.size, request.device_index, request.resource_type};
    switch (mode) {
        case 0: result.loaded = false; break;
        case 1: result.resource.backend = Backend::cuda; break;
        case 2: result.resource.resource_type = load::ResourceType::mask; break;
        case 3: result.resource.size += 1; break;
        case 4: result.resource.device_index += 1; break;
        case 5: result.resource.size = 0; break;
    }
    return result;
}
static void rollback(load::Resource& resource) noexcept {
    std::free(resource.native_handle);
    resource = {};
    ++rollbacks;
}
int main() {
    const unsigned char bytes[]{1, 2, 3};
    load::Request request{bytes, sizeof(bytes), 0, load::ResourceType::model};
    if (load::register_model_translation(Backend::mesa, &produce, nullptr)) return 1;
    if (!load::register_model_translation(Backend::mesa, &produce, &rollback)) return 2;
    auto over_budget = request;
    over_budget.max_bytes = sizeof(bytes) - 1;
    if (load::load_resource(Backend::mesa, over_budget).loaded || allocations != 0) return 5;
    for (mode = 0; mode < 6; ++mode) {
        auto result = load::load_resource(Backend::mesa, request);
        if (result.loaded || result.resource.native_handle || result.diagnostic.empty()) return 3;
        if (rollbacks != allocations) return 4;
    }
    std::cout << "Malformed load: six metadata/failure cases rolled back exactly once\n";
    return 0;
}
