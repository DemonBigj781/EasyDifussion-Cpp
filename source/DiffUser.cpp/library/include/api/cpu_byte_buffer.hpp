#pragma once

#include <cstddef>
#include <memory>

namespace easyapi {

// Exclusive byte storage, not a parsed inference model. No raw backend handle
// types are exposed by this header. Borrowed data expires on move/reset/destruction.
// Independent owners may be used by different threads, not the same owner.
class CpuByteBuffer final {
public:
    CpuByteBuffer() noexcept;
    ~CpuByteBuffer() noexcept;
    CpuByteBuffer(const CpuByteBuffer&) = delete;
    CpuByteBuffer& operator=(const CpuByteBuffer&) = delete;
    CpuByteBuffer(CpuByteBuffer&& other) noexcept;
    CpuByteBuffer& operator=(CpuByteBuffer&& other) noexcept;

    static CpuByteBuffer copy_from(const void* data, std::size_t size,
                                   std::size_t max_bytes = 256u * 1024 * 1024);
    const std::byte* data() const noexcept;
    std::size_t size() const noexcept;
    bool empty() const noexcept;
    void reset() noexcept;
    static const char* storage_backend() noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace easyapi
