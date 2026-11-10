#pragma once
#include <cstddef>
#include <cstdint>

namespace edcpp::native::cpu {
// Internal allocator utility shared by this backend's load/unload definitions.
struct Allocation { void* data = nullptr; std::uint64_t identity = 0; };
Allocation copy_bytes(const void* source, std::size_t size);
bool release_bytes(void* data, std::uint64_t identity) noexcept;
const char* allocator_name() noexcept;
std::size_t live_allocations() noexcept;
}
