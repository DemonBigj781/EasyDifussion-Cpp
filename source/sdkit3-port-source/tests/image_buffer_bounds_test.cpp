#include "image_utils.h"
#include "image_generation_resources.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        for (uint8_t value : {0, 127, 255}) {
            auto mask = createSolidImage(5, 3, 1, value);
            auto* original = mask.data;
            for (float sigma : {0.0f, 0.01f, 1.0f, 4.0f, 64.0f}) {
                require(blurMaskImage(mask, sigma), "solid mask blur failed");
                require(mask.data == original && mask.width == 5 && mask.height == 3, "blur replaced image ownership");
                for (size_t i = 0; i < 15; ++i) require(mask.data[i] == value, "solid mask changed");
            }
            freeImage(mask);
        }
        auto impulse = createSolidImage(3, 3, 1, 0);
        impulse.data[4] = 255;
        require(blurMaskImage(impulse, 0) && impulse.data[4] == 255 && impulse.data[0] == 0,
                "zero blur changed pixels");
        require(!blurMaskImage(impulse, -1) && !blurMaskImage(impulse, 65) &&
                !blurMaskImage(impulse, std::numeric_limits<float>::infinity()), "bad blur accepted");
        require(impulse.data[4] == 255 && impulse.data[0] == 0, "failed blur mutated mask");
        require(blurMaskImage(impulse, 1), "impulse blur failed");
        const uint8_t expected[] = {15, 25, 15, 25, 41, 25, 15, 25, 15};
        require(std::memcmp(impulse.data, expected, 9) == 0, "Gaussian impulse kernel changed");
        freeImage(impulse);
        auto edge = createSolidImage(8, 1, 1, 0);
        for (size_t i = 0; i < 4; ++i) edge.data[i] = 255;
        require(blurMaskImage(edge, 1), "edge blur failed");
        require(edge.data[0] == 255 && edge.data[7] == 0 && edge.data[3] == 178 && edge.data[4] == 77,
                "edge clamp or Gaussian transition changed");
        freeImage(edge);
        auto color = createSolidImage(2, 2, 3, 255);
        require(!blurMaskImage(color, 1), "multichannel mask accepted");
        freeImage(color);

        for (uint32_t channels : {1u, 2u, 3u, 4u}) {
            sd_image_t image = createSolidImage(2, 2, channels, 127);
            require(image.data != nullptr, "fixture allocation failed");
            auto* original = image.data;
            require(!resizeImage(image, std::numeric_limits<int>::max(), 2), "overflow target accepted");
            require(!resizeImage(image, 2, -1), "negative target accepted");
            require(image.data == original && image.width == 2 && image.height == 2,
                    "failed resize changed ownership or dimensions");
            require(resizeImage(image, 2, 2) && image.data == original, "no-op resize replaced allocation");
            require(resizeImage(image, 9, 9, true) && image.width == 8 && image.height == 8,
                    "existing clamp-to-eight behavior changed");
            for (size_t i = 0; i < 64 * channels; ++i) require(image.data[i] == 127, "solid resize corrupted pixels");
            const auto encoded = imageToBase64(image);
            require(!encoded.empty(), "PNG encoding failed");
            auto decoded = base64ToImage(encoded, channels);
            require(decoded.data && decoded.width == 8 && decoded.height == 8 && decoded.channel == channels,
                    "PNG round trip changed shape");
            require(std::memcmp(decoded.data, image.data, 64 * channels) == 0, "PNG round trip changed pixels");
            auto automatic = base64ToImage(encoded, 0);
            require(automatic.data && automatic.channel == channels, "automatic decoder channels changed");
            require(std::memcmp(automatic.data, image.data, 64 * channels) == 0,
                    "automatic channel round trip changed pixels");
            freeImage(automatic);
            freeImage(decoded);
            freeImage(image);
            require(!image.data && image.width == 0 && image.height == 0, "release failed");
        }
        require(!createSolidImage(-1, 8, 1, 255).data &&
                !createSolidImage(65536, 65536, 1, 255).data &&
                !createSolidImage(8, 8, 5, 255).data, "invalid solid mask allocation accepted");
        auto mask = createSolidImage(8, 8, 1, 255);
        require(mask.data != nullptr, "default mask allocation failed");
        for (size_t i = 0; i < 64; ++i) require(mask.data[i] == 255, "default mask not white");
        freeImage(mask);
        sd_img_gen_params_t request{};
        auto source = createSolidImage(8, 8, 3, 127);
        const auto source_png = imageToBase64(source);
        freeImage(source);
        bool decoded_init = false, rejected_mask = false;
        try {
            sdkit::ImageInputResources resources(request);
            request.init_image = base64ToImage(source_png);
            decoded_init = request.init_image.data != nullptr;
            request.mask_image = base64ToImage("invalid-mask", 1);
            rejected_mask = !request.mask_image.data;
            if (rejected_mask) throw std::runtime_error("malformed mask");
        } catch (const std::runtime_error&) {}
        require(decoded_init && rejected_mask, "real malformed-mask fixture did not execute");
        require(!request.init_image.data && !request.mask_image.data, "real malformed mask retained init image");
        // Raw pixels fit int, but stb's ring or total workspace would overflow.
        auto scratch_source = createSolidImage(2, 2, 3, 127);
        auto* scratch_original = scratch_source.data;
        require(!resizeImage(scratch_source, 100000000, 1), "overflowing ring workspace accepted");
        require(scratch_source.data == scratch_original && scratch_source.width == 2,
                "workspace rejection changed ownership");
        freeImage(scratch_source);
        auto total_source = createSolidImage(2, 2, 1, 127);
        require(!resizeImage(total_source, 40000000, 1), "overflowing workspace sum accepted");
        freeImage(total_source);
        for (int channels : {1, 2, 3, 4}) {
            auto mixed = createSolidImage(8, 4, channels, 127);
            for (const auto dimensions : {std::pair<int, int>{16, 2}, {1, 1}, {3, 7}, {1, 4}}) {
                require(resizeImage(mixed, dimensions.first, dimensions.second), "ordinary mixed-scale resize rejected");
                for (size_t i = 0; i < static_cast<size_t>(mixed.width) * mixed.height * mixed.channel; ++i)
                    require(mixed.data[i] == 127, "mixed-scale resize changed solid pixels");
            }
            freeImage(mixed);
        }
        uint8_t pixel = 0;
        for (auto invalid : {sd_image_t{std::numeric_limits<uint32_t>::max(), 1, 3, &pixel},
                             sd_image_t{65536, 65536, 4, &pixel}, sd_image_t{1, 1, 5, &pixel},
                             sd_image_t{1, 1, 0, &pixel}}) {
            require(!resizeImage(invalid, 8, 8), "invalid source descriptor accepted");
            require(imageToBase64(invalid).empty(), "invalid PNG descriptor accepted");
            require(invalid.data == &pixel, "invalid source was freed");
        }
        sd_image_t extreme_downsample{1, 1000000000, 1, &pixel};
        require(!resizeImage(extreme_downsample, 1, 1), "unsafe filter-width cast accepted");
        require(extreme_downsample.data == &pixel, "rejected downsample released source");
        // The pixel bytes fit int, but PNG's extra filter byte per row does not.
        sd_image_t png_overflow{1, static_cast<uint32_t>(std::numeric_limits<int>::max()), 1, &pixel};
        require(imageToBase64(png_overflow).empty(), "PNG row-filter arithmetic overflow accepted");
        require(!base64ToImage("AA==", 5).data && !base64ToImage("AA==", -1).data,
                "invalid decoder channel policy accepted");
        std::cout << "Image buffer bounds and real resize/PNG ownership round trips passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
