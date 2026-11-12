#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include "core/rng_philox.hpp"

static size_t allocation_count = 0;

void* operator new(size_t size) {
    ++allocation_count;
    if (void* ptr = std::malloc(size ? size : 1)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, size_t) noexcept { std::free(ptr); }

// Scalar reference preserves the previous round's four lanes before replacing
// them. Keep the floating-point conversion order identical to the legacy RNG.
static float reference_value(uint64_t seed, uint32_t offset, uint32_t index) {
    std::array<uint32_t, 4> counter = {offset, 0, index, 0};
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        const uint64_t product0 = uint64_t(counter[0]) * 0xD2511F53ULL;
        const uint64_t product1 = uint64_t(counter[2]) * 0xCD9E8D57ULL;
        counter = {uint32_t(product1 >> 32) ^ counter[1] ^ key0,
                   uint32_t(product1 & 0xffffffffULL),
                   uint32_t(product0 >> 32) ^ counter[3] ^ key1,
                   uint32_t(product0 & 0xffffffffULL)};
        key0 += 0x9E3779B9U;
        key1 += 0xBB67AE85U;
    }
    const float inv = 2.3283064e-10f;
    const float inv_2pi = inv * 6.2831855f;
    float u = float(counter[0]) * inv + inv / 2;
    float v = float(counter[1]) * inv_2pi + inv_2pi / 2;
    float s = sqrt(-2.0f * log(u));
    return s * sin(v);
}

static bool check_call(PhiloxRNG& rng, uint64_t seed, uint32_t offset, uint32_t count) {
    auto values = rng.randn(count);
    if (values.size() != count) return false;
    for (uint32_t i = 0; i < count; ++i) {
        const float expected = reference_value(seed, offset, i);
        if (std::memcmp(&values[i], &expected, sizeof(float)) != 0) {
            std::fprintf(stderr, "Philox mismatch: seed=%llu offset=%u index=%u\n",
                         static_cast<unsigned long long>(seed), offset, i);
            return false;
        }
    }
    return true;
}

int main() {
    const uint64_t seeds[] = {0, 1, 42, 0xffffffffULL, 0x100000000ULL,
                              0xffffffffffffffffULL, 0x0123456789abcdefULL};
    const uint32_t counts[] = {0, 1, 2, 3, 7, 16, 65, 257, 4096};
    size_t samples = 0;
    for (uint64_t seed : seeds) {
        PhiloxRNG rng(seed);
        uint32_t offset = 0;
        for (uint32_t count : counts) {
            if (!check_call(rng, seed, offset++, count)) return 1;
            samples += count;
        }
        rng.manual_seed(seed);
        if (!check_call(rng, seed, 0, 257)) return 1;
        rng.manual_seed(seed ^ 0xffffffffffffffffULL);
        if (!check_call(rng, seed ^ 0xffffffffffffffffULL, 0, 65)) return 1;
    }
    PhiloxRNG rng(42);
    const size_t before = allocation_count;
    const auto values = rng.randn(4096);
    const size_t allocations = allocation_count - before;
    // Four counter lanes and one output allocation, with scalar shared keys.
    if (values.size() != 4096 || allocations > 5) {
        std::fprintf(stderr, "Too many Philox allocations: %zu\n", allocations);
        return 1;
    }
    std::printf("Philox: %zu stream samples bit-identical, reseeding and empty calls pass; %zu allocations for 4096 outputs\n",
                samples, allocations);
    return 0;
}
