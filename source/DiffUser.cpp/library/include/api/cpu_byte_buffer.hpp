#pragma once

#include "features/load/common/devices/load.hpp"
#include <cstddef>

namespace easyapi {

// Exclusive ownership of synthetic/model bytes, not a parsed inference model.
// Independent instances may be used on different threads. Mutating one instance
// concurrently or retaining data() across reset/move/destruction is not allowed.
class CpuByteBuffer final {
public:
    CpuByteBuffer() noexcept = default;
    ~CpuByteBuffer() noexcept;
    CpuByteBuffer(const CpuByteBuffer&) = delete;
    CpuByteBuffer& operator=(const CpuByteBuffer&) = delete;
    CpuByteBuffer(CpuByteBuffer&& other) noexcept;
    CpuByteBuffer& operator=(CpuByteBuffer&& other) noexcept;

    static CpuByteBuffer copy_from(const void* data, std::size_t size);
    const std::byte* data() const noexcept;
    std::size_t size() const noexcept;
    bool empty() const noexcept;
    void reset() noexcept;

private:
    edcpp::api::load::Resource resource_{};
};

} // namespace easyapi
