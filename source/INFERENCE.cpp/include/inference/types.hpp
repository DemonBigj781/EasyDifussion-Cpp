#pragma once

#include "features/load/common/devices/load.hpp"

#include <cstdint>
#include <string>

namespace edcpp::inference {

enum class MediaKind : std::uint8_t { image, video };
enum class Sampler : std::uint8_t { euler, euler_a, heun, dpm2, dpmpp_2m, lcm };
enum class Scheduler : std::uint8_t { discrete, karras, exponential, ays, gits };

struct GenerationRequest {
    MediaKind media = MediaKind::image;
    std::string prompt{};
    std::string negative_prompt{};
    const edcpp::api::load::Resource* model = nullptr;
    std::uint32_t width = 512;
    std::uint32_t height = 512;
    std::uint32_t steps = 20;
    std::uint32_t frames = 1;
    std::uint32_t batch = 1;
    float guidance = 7.0f;
    float strength = 1.0f;
    std::int64_t seed = -1;
    Sampler sampler = Sampler::euler_a;
    Scheduler scheduler = Scheduler::discrete;
};

struct ValidationResult {
    bool ok = false;
    std::string message{};
};

} // namespace edcpp::inference
