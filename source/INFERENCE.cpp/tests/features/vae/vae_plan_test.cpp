#include "features/vae/common/vae.hpp"

#include <cassert>

int main() {
    namespace vae = edcpp::inference::vae;

    const auto encoded = vae::make_encode_plan({2, 3, 512, 768});
    assert(encoded.ok);
    assert(encoded.plan.output.batch == 2);
    assert(encoded.plan.output.channels == 4);
    assert(encoded.plan.output.height == 64);
    assert(encoded.plan.output.width == 96);

    const auto decoded = vae::make_decode_plan(encoded.plan.output);
    assert(decoded.ok);
    assert(decoded.plan.output.channels == 3);
    assert(decoded.plan.output.height == 512);
    assert(decoded.plan.output.width == 768);

    assert(!vae::make_encode_plan({1, 3, 513, 512}).ok);
    assert(!vae::make_decode_plan({1, 3, 64, 64}).ok);

    vae::Config tiled;
    tiled.tiling = {true, 256, 256, 32};
    assert(vae::make_encode_plan({1, 3, 512, 512}, tiled).ok);
    tiled.tiling.overlap = 256;
    assert(!vae::make_encode_plan({1, 3, 512, 512}, tiled).ok);
}
