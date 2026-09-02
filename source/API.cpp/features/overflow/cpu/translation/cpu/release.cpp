#include "release.hpp"

#include "features/overflow/cpu/definition/cpu/release.hpp"

#include <utility>

namespace edcpp::api::overflow::cpu::translation::cpu {

ReleaseResult release(const Resource& resource) noexcept {
    ReleaseResult result;
    if (resource.backend != Backend::cpu ||
        resource.memory_kind != MemoryKind::cpu_heap) {
        result.diagnostic = "CPU overflow release received a resource owned by another allocator";
        return result;
    }

    auto native = definition::cpu::release(resource.host_handle);
    result.released = native.released;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

} // namespace edcpp::api::overflow::cpu::translation::cpu
