#include "api/cpu_byte_buffer.hpp"
#include "features/load/common/devices/load.hpp"
#include "features/unload/common/devices/unload.hpp"
#include <exception>
#include "native/cpu/byte_storage.hpp"
#include <stdexcept>
#include <utility>

namespace easyapi {
struct CpuByteBuffer::State {
    edcpp::api::load::Resource resource{};
    ~State() noexcept {
        if (!resource.loaded()) return;
        try {
            if (!edcpp::api::unload::unload_resource(resource).unloaded) std::terminate();
        } catch (...) { std::terminate(); }
    }
};
CpuByteBuffer::CpuByteBuffer() noexcept = default;
CpuByteBuffer::~CpuByteBuffer() noexcept = default;
CpuByteBuffer::CpuByteBuffer(CpuByteBuffer&& other) noexcept = default;
CpuByteBuffer& CpuByteBuffer::operator=(CpuByteBuffer&& other) noexcept {
    if (this != &other) state_ = std::move(other.state_);
    return *this;
}
CpuByteBuffer CpuByteBuffer::copy_from(const void* data, std::size_t size, std::size_t max_bytes) {
    // Allocate the owner before the backend resource, so owner allocation failure
    // cannot orphan an already allocated byte buffer.
    auto state = std::make_unique<State>();
    edcpp::api::load::Request request;
    request.data = data;
    request.size = size;
    request.max_bytes = max_bytes;
    auto result = edcpp::api::load::load_resource(edcpp::api::Backend::cpu, request);
    if (!result.loaded) throw std::invalid_argument(result.diagnostic);
    state->resource = result.resource;
    CpuByteBuffer owned;
    owned.state_ = std::move(state);
    return owned;
}
const std::byte* CpuByteBuffer::data() const noexcept {
    return state_ ? static_cast<const std::byte*>(state_->resource.native_handle) : nullptr;
}
std::size_t CpuByteBuffer::size() const noexcept {
    return state_ ? static_cast<std::size_t>(state_->resource.size) : 0;
}
bool CpuByteBuffer::empty() const noexcept { return !state_; }
void CpuByteBuffer::reset() noexcept { state_.reset(); }
const char* CpuByteBuffer::storage_backend() noexcept { return edcpp::native::cpu::allocator_name(); }
} // namespace easyapi
