#include "core/native_streaming_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool result, const char* description) {
    if (!result) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

int main() {
    constexpr size_t MiB = 1024ull * 1024;
    using namespace sd_streaming;
    require(validate_activation(true, 1, true) == nullptr, "one accelerator + CPU weights permits streaming");
    require(validate_activation(false, 1, true) != nullptr, "no CPU backing must not pretend to stream");
    require(validate_activation(true, 1, false) != nullptr, "CPU compute is not accelerator streaming");
    require(validate_activation(true, 2, true) != nullptr, "multi-device path needs separate validation");
    require(validate_activation(true, 0, true) != nullptr, "unavailable runtime device cannot stream");
    require(reported_streaming_headroom(2048 * MiB, 4096 * MiB) == 1536 * MiB,
            "no explicit VRAM option still produces planner headroom");
    require(reported_streaming_headroom(2048 * MiB, 4096 * MiB, 1024 * MiB) == 1024 * MiB,
            "an explicit library cap remains an upper bound");
    require(reported_streaming_headroom(640 * MiB, 4096 * MiB, 1024 * MiB) == 128 * MiB,
            "a free-memory drop beats an earlier cap");
    require(reported_streaming_headroom(512 * MiB, 4096 * MiB) == 0,
            "reserved memory cannot become an unlimited planning budget");
    require(reported_streaming_headroom(2048 * MiB, 0) == 0 &&
            reported_streaming_headroom(4096 * MiB, 2048 * MiB) == 0,
            "absent or impossible device telemetry is not used");
    require(fits_budget(10, {4, 4, 2}) && !fits_budget(10, {4, 4, 3}),
            "estimated segment includes weights, workspace and retained output");
    require(!fits_budget(std::numeric_limits<size_t>::max(), {std::numeric_limits<size_t>::max(), 1}),
            "estimate overflow must not make an impossible segment fit");
    std::puts("PASS: streaming activation requires host backing and a bounded single-accelerator plan");
}
