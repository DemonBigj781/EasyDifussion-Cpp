#pragma once

#include <cstdint>
#include <string>

namespace edcpp::inference::vae {

enum class Operation : std::uint8_t { encode, decode };

struct Shape {
    std::uint32_t batch = 1;
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
};

struct Tiling {
    bool enabled = false;
    std::uint32_t tile_width = 0;
    std::uint32_t tile_height = 0;
    std::uint32_t overlap = 0;
};

struct Config {
    std::uint32_t image_channels = 3;
    std::uint32_t latent_channels = 4;
    std::uint32_t spatial_downscale = 8;
    float latent_scale = 1.0f;
    float latent_shift = 0.0f;
    Tiling tiling{};
};

struct Plan {
    Operation operation = Operation::encode;
    Shape input{};
    Shape output{};
    Config config{};
};

struct Result {
    bool ok = false;
    std::string message{};
    Plan plan{};
};

Result make_encode_plan(const Shape& image, const Config& config = {});
Result make_decode_plan(const Shape& latent, const Config& config = {});

} // namespace edcpp::inference::vae
