#include "allocate.hpp"

#include "features/overflow/cpu/definition/cpu/allocate.hpp"

#include <utility>

namespace edcpp::api::overflow::cpu::translation::cpu {

Result allocate(const Request& request) {
    auto native = definition::cpu::allocate(
        request.data, request.size, request.host_reserve);

    Result result;
    result.allocated = native.allocated;
    result.overflowed = native.allocated;
    result.resource.backend = Backend::cpu;
    result.resource.tier = Tier::system_ram;
    result.resource.memory_kind = MemoryKind::cpu_heap;
    result.resource.host_handle = native.handle;
    result.resource.size = native.size;
    result.diagnostic = std::move(native.diagnostic);
    return result;
}

} // namespace edcpp::api::overflow::cpu::translation::cpu
