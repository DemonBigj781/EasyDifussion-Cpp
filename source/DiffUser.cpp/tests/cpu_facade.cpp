#include "api/cpu_byte_buffer.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

int main() {
    try {
#ifdef EXPECT_GGML_CPU
        if (std::strcmp(easyapi::CpuByteBuffer::storage_backend(), "ggml-cpu") != 0) return 5;
#endif
        const std::array<unsigned char, 4> bytes{1, 2, 3, 4};
        for (int cycle = 0; cycle < 16; ++cycle) {
            auto owner = easyapi::CpuByteBuffer::copy_from(bytes.data(), bytes.size(), bytes.size());
            auto moved = std::move(owner);
            if (!owner.empty() || moved.size() != bytes.size() ||
                std::memcmp(moved.data(), bytes.data(), bytes.size())) return 1;
        }
        for (auto budget : {std::size_t{0}, bytes.size() - 1}) {
            bool rejected = false;
            try { (void)easyapi::CpuByteBuffer::copy_from(bytes.data(), bytes.size(), budget); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) return 2;
        }
        bool rejected = false;
        try {
            (void)easyapi::CpuByteBuffer::copy_from(bytes.data(),
                std::numeric_limits<std::size_t>::max(),
                std::numeric_limits<std::size_t>::max());
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 3;
        std::cout << "Allocator: " << easyapi::CpuByteBuffer::storage_backend() << "\n";
        std::cout << "Public facade: static archive retention and bounded allocation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 4;
    }
}
