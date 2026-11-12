#include <cmath>
#include <iostream>
#include <stdexcept>

#include "core/ggml_extend.hpp"
#include "runtime/upscale_alpha.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void check_image(int width, int height, int scale, int batch, int channels, bool tiled) {
    sd::Tensor<float> input({width,height,channels,batch});
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    input[((n * channels + c) * height + y) * width + x] = c == 3 ? float(x + y + n) / (width + height + batch) : (c + n + 1) / 8.f;
                }
            }
        }
    }
    auto model = [&](const sd::Tensor<float>& rgb) {
        require(rgb.dim() == 4 && rgb.shape()[2] == 3 && rgb.shape()[3] == batch, "alpha or incorrect batch sent to RGB model");
        auto shape = rgb.shape();
        shape[0] *= scale;
        shape[1] *= scale;
        return sd::ops::interpolate(rgb, shape, sd::ops::InterpolateMode::Nearest);
    };
    auto output = sd::upscale_rgb_preserving_alpha(input, [&](const sd::Tensor<float>& rgb) {
        return tiled ? process_tiles_2d(rgb, width * scale, height * scale, scale, 4, 4, 0.25f, false, false, model, true) : model(rgb);
    });
    const int ow = width * scale, oh = height * scale;
    require(output.shape() == std::vector<int64_t>({ow,oh,channels,batch}), "upscale shape/channels changed");
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < oh; ++y) {
                for (int x = 0; x < ow; ++x) {
                    const float sx = std::max(0.f, std::min(float(width - 1), (x + 0.5f) / scale - 0.5f));
                    const float sy = std::max(0.f, std::min(float(height - 1), (y + 0.5f) / scale - 0.5f));
                    const float expected = c == 3 ? (sx + sy + n) / (width + height + batch) : (c + n + 1) / 8.f;
                    require(std::abs(output[((n * channels + c) * oh + y) * ow + x] - expected) < 0.00001f,
                            "RGB changed or alpha was not independently bilinear-resized");
                }
            }
        }
    }
}
int main() {
    try {
        int cases = 0;
        for (auto size : {std::pair<int,int>{3,2},{9,7}}) {
            for (int scale : {1,2,4}) {
                for (int batch : {1,2}) {
                    for (int channels : {3,4}) {
                        for (bool tiled : {false,true}) {
                            check_image(size.first, size.second, scale, batch, channels, tiled);
                            ++cases;
                        }
                    }
                }
            }
        }
        int calls = 0;
        auto callback = [&](const sd::Tensor<float>& input) { ++calls; return input; };
        for (auto shape : {std::vector<int64_t>{2,2,1,1}, {2,2,2,1}, {2,2,4}, {2,2,3,1,1}}) {
            require(sd::upscale_rgb_preserving_alpha(sd::Tensor<float>(shape), callback).empty(), "invalid shape accepted");
        }
        require(sd::upscale_rgb_preserving_alpha(sd::Tensor<float>(), callback).empty() && calls == 0,
                "invalid input reached RGB computation");
        require(sd::upscale_rgb_preserving_alpha(sd::Tensor<float>({2,2,4,1}),
                    [](const sd::Tensor<float>&) { return sd::Tensor<float>(); }).empty(),
                "failed/cancelled RGB compute produced alpha-only result");
        std::cout << "PASS " << cases << " RGB/RGBA batch/scale/tiled cases, bilinear alpha, invalid input and empty compute propagation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
