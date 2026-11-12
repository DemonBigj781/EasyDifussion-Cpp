#include <cmath>
#include <iostream>
#include <stdexcept>

#include "core/util.h"

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void check_constant(int64_t width, int64_t height, int target_width, int target_height) {
    sd::Tensor<float> image({width, height, 3, 1});
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                image.index(x, y, c, 0) = 0.5f;
            }
        }
    }
    auto result = clip_preprocess(image, target_width, target_height);
    require(result.shape() == std::vector<int64_t>({target_width, target_height, 3, 1}), "wrong crop dimensions");
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float stddev[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    for (int64_t y = 0; y < target_height; ++y) {
        for (int64_t x = 0; x < target_width; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                require(std::abs(result.index(x, y, c, 0) - (0.5f - mean[c]) / stddev[c]) < 0.00001f,
                        "crop or CLIP normalization corrupted");
            }
        }
    }
}

static void check_unchanged_center_crop() {
    sd::Tensor<float> image({2, 6, 3, 1});
    for (int64_t y = 0; y < 6; ++y) {
        for (int64_t x = 0; x < 2; ++x) {
            image.index(x, y, 0, 0) = -0.5f;
            image.index(x, y, 1, 0) = static_cast<float>(y) / 5.0f;
            image.index(x, y, 2, 0) = 1.5f;
        }
    }
    auto result = clip_preprocess(image, 2, 2);
    for (int64_t y = 0; y < 2; ++y) {
        for (int64_t x = 0; x < 2; ++x) {
            require(std::abs(result.index(x, y, 0, 0) - (0.0f - 0.48145466f) / 0.26862954f) < 0.00001f,
                    "lower clamp changed");
            require(std::abs(result.index(x, y, 1, 0) - (static_cast<float>(y + 2) / 5.0f - 0.4578275f) / 0.26130258f) < 0.00001f,
                    "ordinary center crop shifted");
            require(std::abs(result.index(x, y, 2, 0) - (1.0f - 0.40821073f) / 0.27577711f) < 0.00001f,
                    "upper clamp changed");
        }
    }
}

int main() {
    try {
        check_constant(23, 23, 224, 224);
        check_constant(23, 47, 224, 224);
        check_constant(47, 23, 224, 224);
        check_constant(131, 131, 336, 336);
        check_constant(730, 730, 736, 736);
        check_constant(23, 31, 224, 336);
        check_constant(512, 256, 224, 224);
        check_constant(8, 8, 8, 8);
        check_unchanged_center_crop();
        std::cout << "PASS rounded resize bounds, portrait/landscape/rectangular/identity cases, center crop and normalization\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
