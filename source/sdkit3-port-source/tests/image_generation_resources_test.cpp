#include "image_generation_resources.h"
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <type_traits>

static std::set<uint8_t*> live;
static int freed = 0;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static sd_image_t allocation() {
    auto* data = static_cast<uint8_t*>(std::malloc(1));
    require(data != nullptr, "fixture allocation failed");
    live.insert(data);
    return {1, 1, 1, data};
}
// Instrument only the real guards' cleanup boundary; no model/GPU execution.
void freeImage(sd_image_t& image) {
    if (image.data) {
        if (live.erase(image.data) != 1) std::abort();
        std::free(image.data);
        ++freed;
    }
    image = {};
}
extern "C" void free_sd_images(sd_image_t* images, int count) {
    for (int i = 0; images && i < count; ++i) freeImage(images[i]);
    std::free(images);
}
int main() {
    static_assert(!std::is_copy_constructible<sdkit::ImageInputResources>::value);
    static_assert(!std::is_copy_constructible<sdkit::ImageResultResources>::value);
    try {
        for (int fail_after = 1; fail_after <= 8; ++fail_after) {
            sd_img_gen_params_t params{};
            const int before = freed;
            try {
                sdkit::ImageInputResources resources(params);
                sd_image_t* slots[]{&params.init_image, &params.mask_image, &params.control_image,
                    &params.control_net_lllite_image, &params.ip_adapter.image,
                    &params.latent_interposer_source_image};
                int added = 0;
                for (auto* slot : slots) {
                    *slot = allocation();
                    if (++added == fail_after) throw std::runtime_error("later input failed");
                }
                for (int i = 0; i < 2; ++i) {
                    resources.references.push_back(allocation());
                    params.ref_images = resources.references.data();
                    params.ref_images_count = static_cast<int>(resources.references.size());
                    if (++added == fail_after) throw std::runtime_error("later reference failed");
                }
            } catch (const std::runtime_error&) {}
            require(live.empty() && freed - before == fail_after, "partial inputs leaked or were double-freed");
            require(!params.init_image.data && !params.ref_images && params.ref_images_count == 0,
                    "borrowed descriptors survived cleanup");
        }
        {
            std::vector<sd_image_t> references;
            try {
                sdkit::ImageListGuard list_guard(references);
                references.push_back(allocation());
                auto current = allocation();
                sdkit::ImageBufferGuard current_guard(current);
                throw std::runtime_error("reference append failed");
            } catch (const std::runtime_error&) {}
            require(live.empty(), "partial reference list/current image leaked");
        }
        {
            std::vector<sd_image_t> delivered;
            {
                std::vector<sd_image_t> prepared;
                sdkit::ImageListGuard list_guard(prepared);
                auto current = allocation();
                sdkit::ImageBufferGuard current_guard(current);
                prepared.push_back(current);
                current = {};
                list_guard.dismiss();
                delivered = std::move(prepared);
            }
            require(live.size() == 1, "successful reference transfer freed borrowed data");
            sdkit::ImageListGuard delivered_guard(delivered);
        }
        require(live.empty(), "transferred reference was not released");
        for (int failure_stage : {0, 1}) {
            const int before = freed;
            try {
                sdkit::ImageResultResources output;
                output.images = static_cast<sd_image_t*>(std::calloc(2, sizeof(sd_image_t)));
                require(output.images != nullptr, "result descriptor allocation failed");
                output.count = 2;
                output.images[0] = allocation();
                if (failure_stage == 0) throw std::runtime_error("native call threw after publishing output");
                output.images[1] = allocation();
                throw std::runtime_error("output encoding failed");
            } catch (const std::runtime_error&) {}
            require(live.empty() && freed - before == failure_stage + 1, "output failure leaked allocations");
        }
        int active = 1, callbacks = 1, cleanup_count = 0;
        {
            sdkit::ImageGenerationStateGuard state([&] { active = callbacks = 0; ++cleanup_count; });
            sd_img_gen_params_t params{};
            sdkit::ImageInputResources input(params);
            params.init_image = allocation();
            input.reset();
            state.reset();
            require(live.empty() && active == 0 && callbacks == 0, "fallback began before cleanup");
            active = callbacks = 2; // A recursive/new request publishes new state.
        }
        require(active == 2 && callbacks == 2 && cleanup_count == 1, "old scope cleared retry state");
        try {
            sdkit::ImageGenerationStateGuard state([&] { active = callbacks = 0; });
            throw std::runtime_error("generation threw");
        } catch (const std::runtime_error&) {}
        require(active == 0 && callbacks == 0, "exception left active generation state");
        std::cout << "Input/result ownership and fallback state cleanup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
