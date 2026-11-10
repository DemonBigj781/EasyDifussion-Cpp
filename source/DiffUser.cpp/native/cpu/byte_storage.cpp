#include "native/cpu/byte_storage.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#ifdef EDCPP_USE_GGML_CPU_RESOURCE
#include "ggml-backend.h"
#endif

namespace edcpp::native::cpu {
namespace {
struct Storage {
    void* data = nullptr;
#ifdef EDCPP_USE_GGML_CPU_RESOURCE
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() { ggml_backend_buffer_free(buffer); }
#else
    ~Storage() { std::free(data); }
#endif
    std::uint64_t identity = 0;
};
struct Allocations {
    std::mutex mutex;
    std::unordered_map<void*, std::unique_ptr<Storage>> owned;
    std::uint64_t next_identity = 1;
};
Allocations& allocations() {
    static Allocations state;
    return state;
}
}
Allocation copy_bytes(const void* source, std::size_t size) {
    auto storage = std::make_unique<Storage>();
#ifdef EDCPP_USE_GGML_CPU_RESOURCE
    storage->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (!storage->buffer) throw std::bad_alloc();
    storage->data = ggml_backend_buffer_get_base(storage->buffer);
    if (ggml_backend_buffer_get_size(storage->buffer) < size) throw std::bad_alloc();
#else
    storage->data = std::malloc(size);
#endif
    if (!storage->data) throw std::bad_alloc();
    std::memcpy(storage->data, source, size);
    auto& state = allocations();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.next_identity == std::numeric_limits<std::uint64_t>::max()) throw std::bad_alloc();
    storage->identity = state.next_identity++;
    Allocation result{storage->data, storage->identity};
    // If map insertion throws, local Storage still frees its allocation.
    if (!state.owned.emplace(result.data, std::move(storage)).second) throw std::bad_alloc();
    return result;
}
bool release_bytes(void* data, std::uint64_t identity) noexcept {
    try {
        auto& state = allocations();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto found = state.owned.find(data);
        if (!data || identity == 0 || found == state.owned.end() ||
            found->second->identity != identity) return false;
        state.owned.erase(found);
        return true;
    } catch (...) { return false; }
}
std::size_t live_allocations() noexcept {
    try {
        auto& state = allocations();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.owned.size();
    } catch (...) { return std::numeric_limits<std::size_t>::max(); }
}
const char* allocator_name() noexcept {
#ifdef EDCPP_USE_GGML_CPU_RESOURCE
    return "ggml-cpu";
#else
    return "system-cpu";
#endif
}
}
