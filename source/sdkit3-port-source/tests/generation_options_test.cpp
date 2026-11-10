#include "generation_options.h"
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        auto absent = crow::json::load("{}");
        sd_img_gen_params_t image;
        sd_img_gen_params_init(&image);
        sd_vid_gen_params_t video;
        sd_vid_gen_params_init(&video);
        require(image.sample_params.guidance.distilled_guidance == 3.5f, "runtime default changed; review compatibility");
        apply_distilled_guidance(image.sample_params, parse_distilled_guidance(absent));
        apply_distilled_guidance(video.sample_params, parse_distilled_guidance(absent));
        require(image.sample_params.guidance.distilled_guidance == 3.5f &&
                video.sample_params.guidance.distilled_guidance == 3.5f, "absent field overwrote runtime default");
        for (float value : {0.0f, 1.25f, -1.0f, 7.0f}) {
            crow::json::wvalue request;
            request["distilled_cfg_scale"] = value;
            const auto parsed = crow::json::load(request.dump());
            require(validate_generation_options(parsed, false).empty(), "finite guidance rejected");
            apply_distilled_guidance(image.sample_params, parse_distilled_guidance(parsed));
            apply_distilled_guidance(video.sample_params, parse_distilled_guidance(parsed));
            require(image.sample_params.guidance.distilled_guidance == value &&
                    video.sample_params.guidance.distilled_guidance == value, "explicit guidance lost");
        }
        for (const auto* json : {"{\"distilled_cfg_scale\":null}", "{\"distilled_cfg_scale\":\"3.5\"}",
                                "{\"distilled_cfg_scale\":true}", "{\"distilled_cfg_scale\":1e100}"}) {
            require(!validate_generation_options(crow::json::load(json), false).empty(), "bad guidance accepted");
        }
        for (const auto* json : {"{}", "{\"mask_blur\":0}", "{\"mask_blur\":4}", "{\"mask_blur\":0.25}", "{\"mask_blur\":64}"}) {
            require(validate_generation_options(crow::json::load(json), true).empty(), "valid mask blur rejected");
        }
        for (const auto* json : {"{\"mask_blur\":null}", "{\"mask_blur\":\"4\"}", "{\"mask_blur\":true}",
                                "{\"mask_blur\":-1}", "{\"mask_blur\":65}", "{\"mask_blur\":1e100}"}) {
            require(!validate_generation_options(crow::json::load(json), true).empty(), "bad mask blur accepted");
            require(validate_generation_options(crow::json::load(json), false).empty(), "unused mask field changed txt2img/video compatibility");
        }
        std::cout << "Guidance forwarding, native defaults, and mask request contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
