#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "core/ggml_extend.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_axis_plans() {
    for (int dimension = 1; dimension <= 64; ++dimension) {
        for (int size = 1; size <= dimension + 4; ++size) {
            for (float overlap : {0.f, 0.125f, 0.25f, 0.5f}) {
                for (bool circular : {false, true}) {
                    const auto plan = sd_tiling_plan_axis(dimension, size, overlap, circular);
                    require(!plan.empty() && plan.size() <= static_cast<size_t>(dimension), "unbounded axis plan");
                    std::vector<int> coverage(dimension, 0);
                    int previous = -1;
                    const int extent = std::min(size, dimension);
                    for (const auto& tile : plan) {
                        require(tile.offset > previous && tile.offset < dimension, "invalid/non-increasing offset");
                        require(tile.overlap_before >= 0 && tile.overlap_after >= 0 &&
                                    tile.overlap_before < extent && tile.overlap_after < extent,
                                "negative or excessive overlap");
                        require(circular || tile.offset + extent <= dimension, "non-circular plan wrapped");
                        previous = tile.offset;
                        for (int i = 0; i < extent; ++i) ++coverage[(tile.offset + i) % dimension];
                    }
                    for (int count : coverage) require(count >= 1 && count <= 2, "uncovered or triple-covered axis pixel");
                }
            }
        }
    }
}

static float value_at(int x, int y, int plane) {
    return (x * 3 - y * 2 + plane * 7) / 31.f;
}

static void check_round_trip(int width, int height, int tile_x, int tile_y, float overlap,
                             bool circular_x, bool circular_y, int scale, bool decode, bool video) {
    const int scale_in = decode ? 1 : scale;
    const int scale_out = decode ? scale : 1;
    const int input_width = width * scale_in;
    const int input_height = height * scale_in;
    const int output_width = width * scale_out;
    const int output_height = height * scale_out;
    std::vector<int64_t> shape = video ? std::vector<int64_t>{input_width, input_height, 3, 2, 2}
                                      : std::vector<int64_t>{input_width, input_height, 3, 2};
    sd::Tensor<float> input(shape);
    const int planes = video ? 12 : 6;
    for (int plane = 0; plane < planes; ++plane) {
        for (int y = 0; y < input_height; ++y) {
            for (int x = 0; x < input_width; ++x) {
                input[plane * input_width * input_height + y * input_width + x] = value_at(x / scale_in, y / scale_in, plane);
            }
        }
    }
    int calls = 0;
    auto output = process_tiles_2d(input, output_width, output_height, scale, tile_x, tile_y, overlap,
                                  circular_x, circular_y, [&](const sd::Tensor<float>& tile) {
        ++calls;
        require(tile.shape()[0] == std::min(tile_x, width) * scale_in &&
                    tile.shape()[1] == std::min(tile_y, height) * scale_in,
                "tile sizes changed from latent/small-dimension units");
        auto result_shape = tile.shape();
        result_shape[0] = tile.shape()[0] / scale_in * scale_out;
        result_shape[1] = tile.shape()[1] / scale_in * scale_out;
        sd::Tensor<float> result(result_shape);
        const int tw = static_cast<int>(tile.shape()[0]);
        const int th = static_cast<int>(tile.shape()[1]);
        const int rw = static_cast<int>(result.shape()[0]);
        const int rh = static_cast<int>(result.shape()[1]);
        for (int p = 0; p < planes; ++p) {
            for (int y = 0; y < rh; ++y) {
                for (int x = 0; x < rw; ++x) {
                    result[p * rw * rh + y * rw + x] = tile[p * tw * th + (y / scale_out * scale_in) * tw + x / scale_out * scale_in];
                }
            }
        }
        return result;
    }, true);
    auto expected_shape = shape;
    expected_shape[0] = output_width;
    expected_shape[1] = output_height;
    require(output.shape() == expected_shape, "image/video shape changed");
    require(calls == static_cast<int>(sd_tiling_plan_axis(width, tile_x, overlap, circular_x).size() *
                                     sd_tiling_plan_axis(height, tile_y, overlap, circular_y).size()),
            "processing did not follow bounded plan");
    for (int p = 0; p < planes; ++p) {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                const float actual = output[p * output_width * output_height + y * output_width + x];
                require(std::isfinite(actual) && std::abs(actual - value_at(x / scale_out, y / scale_out, p)) < 0.0001f,
                        "tile blend introduced holes, seams, scaling errors or plane corruption");
            }
        }
    }
}

int main() {
    try {
        check_axis_plans();
        int cases = 0;
        int shape_direction_cases[2][2] = {};
        for (auto size : {std::pair<int,int>{8,8}, {13,9}, {17,5}, {1,7}, {7,1}}) {
            for (auto tile : {std::pair<int,int>{1,1}, {4,4}, {6,7}, {7,6}, {32,32}}) {
                for (float overlap : {0.f, 0.25f, 0.5f}) {
                    for (int wrap = 0; wrap < 4; ++wrap) {
                        for (int scale : {1,2,8}) {
                            for (bool decode : {false,true}) {
                                for (bool video : {false,true}) {
                                    check_round_trip(size.first, size.second, tile.first, tile.second, overlap,
                                                     wrap & 1, wrap & 2, scale, decode, video);
                                    ++shape_direction_cases[video][decode];
                                    ++cases;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (bool video : {false,true}) {
            for (bool decode : {false,true}) {
                require(shape_direction_cases[video][decode] == 900,
                        "image/video and encode/decode coverage is not independent");
            }
        }
        sd::Tensor<float> input({8,8,3,2,2});
        int calls = 0;
        auto failed = process_tiles_2d(input, 8, 8, 1, 4, 4, 0.25f, true, true,
                                      [&](const sd::Tensor<float>& tile) { return ++calls == 2 ? sd::Tensor<float>() : tile; }, true);
        require(failed.empty() && calls == 2, "tile failure/cancellation did not stop processing");
        std::cout << "PASS bounded axis coverage and " << cases << " image/video tiling cases, encode/decode scales, circular regression and early failure\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
