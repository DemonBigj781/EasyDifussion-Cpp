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
