#include "api/cpu_byte_buffer.hpp"
#include "features/unload/common/devices/unload.hpp"
#include <exception>
#include <stdexcept>
#include <utility>

namespace easyapi {

CpuByteBuffer::~CpuByteBuffer() noexcept { reset(); }

CpuByteBuffer::CpuByteBuffer(CpuByteBuffer&& other) noexcept
    : resource_(std::exchange(other.resource_, {})) {}

CpuByteBuffer& CpuByteBuffer::operator=(CpuByteBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        resource_ = std::exchange(other.resource_, {});
    }
    return *this;
}

CpuByteBuffer CpuByteBuffer::copy_from(const void* data, std::size_t size) {
    edcpp::api::load::Request request;
    request.data = data;
    request.size = size;
    auto result = edcpp::api::load::load_resource(edcpp::api::Backend::cpu, request);
    if (!result.loaded) throw std::invalid_argument(result.diagnostic);
    CpuByteBuffer owned;
    owned.resource_ = result.resource;
    return owned;
}

const std::byte* CpuByteBuffer::data() const noexcept {
    return static_cast<const std::byte*>(resource_.native_handle);
}
std::size_t CpuByteBuffer::size() const noexcept {
    return static_cast<std::size_t>(resource_.size);
}
bool CpuByteBuffer::empty() const noexcept { return !resource_.loaded(); }

void CpuByteBuffer::reset() noexcept {
    if (empty()) return;
    // The registered CPU route is immutable and must release its own resource.
    // An invariant failure must not silently look like a successful release.
    try {
        if (!edcpp::api::unload::unload_resource(resource_).unloaded) std::terminate();
    } catch (...) {
        std::terminate();
    }
}

} // namespace easyapi
