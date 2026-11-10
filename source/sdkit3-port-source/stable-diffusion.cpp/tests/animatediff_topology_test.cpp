#include <iostream>
#include <memory>

#include "model/diffusion/unet.hpp"

static bool validate_frame_batch(SDVersion version, int frames, bool per_frame_timesteps) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({128 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) {
        return false;
    }
    auto config = UNetConfig::detect_from_weights({}, "", version);
    config.enable_animatediff = true;
    UnetModelBlock unet(config);
    unet.init(ctx.get());
    GGMLRunnerContext runner;
    runner.ggml_ctx = ctx.get();
    auto x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 16, 16, 4, frames);
    auto timesteps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, per_frame_timesteps ? frames : 1);
    auto context = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, config.context_dim, 77, 1);
    auto y = sd_version_is_sdxl(version)
                 ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, config.adm_in_channels, 1)
                 : nullptr;
    auto result = unet.forward(&runner, x, timesteps, context, nullptr, y, frames);
    return ggml_are_same_shape(result, x);
}

int main() {
    const auto sdxl = UNetConfig::animatediff_motion_config(VERSION_SDXL, false);
    if (sdxl.down_channels != std::vector<int64_t>({320, 640, 1280}) ||
        sdxl.up_channels != std::vector<int64_t>({1280, 640, 320}) ||
        sdxl.num_down_motion_per_block != 2 ||
        sdxl.num_up_motion_per_block != 3 ||
        sdxl.enable_mid_block) {
        std::cerr << "SDXL AnimateDiff topology does not match mm_sdxl_v10_beta" << std::endl;
        return 1;
    }

    const auto sdxl_with_mid = UNetConfig::animatediff_motion_config(VERSION_SDXL, true);
    if (!sdxl_with_mid.enable_mid_block) {
        std::cerr << "explicit mid-block configuration was lost" << std::endl;
        return 1;
    }

    const auto sd15 = UNetConfig::animatediff_motion_config(VERSION_SD1, false);
    if (sd15.down_channels != std::vector<int64_t>({320, 640, 1280, 1280}) ||
        sd15.up_channels != std::vector<int64_t>({1280, 1280, 640, 320}) ||
        sd15.num_down_motion_per_block != 2 ||
        sd15.num_up_motion_per_block != 3) {
        std::cerr << "SD 1.5 AnimateDiff topology changed unexpectedly" << std::endl;
        return 1;
    }

    String2TensorStorage lightning_weights;
    lightning_weights["model.diffusion_model.motion_module.down_blocks.0.motion_modules.0.temporal_transformer.proj_in.weight"] =
        TensorStorage("model.diffusion_model.motion_module.down_blocks.0.motion_modules.0.temporal_transformer.proj_in.weight",
                      GGML_TYPE_F16,
                      nullptr,
                      0,
                      0);
    lightning_weights["model.diffusion_model.motion_module.down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attn1.to_q.weight"] =
        TensorStorage("model.diffusion_model.motion_module.down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attn1.to_q.weight",
                      GGML_TYPE_F16,
                      nullptr,
                      0,
                      0);
    const auto lightning = UNetConfig::detect_from_weights(lightning_weights,
                                                            "model.diffusion_model",
                                                            VERSION_SD1);
    if (!lightning.enable_animatediff || !lightning.animatediff_uses_diffusers_names) {
        std::cerr << "AnimateDiff-Lightning Diffusers tensor layout was not detected" << std::endl;
        return 1;
    }

    for (auto version : {VERSION_SDXL, VERSION_SD1}) {
        for (int frames : {1, 2, 16}) {
            for (bool per_frame_timesteps : {false, true}) {
                std::cout << "checking version " << version << ", frames " << frames
                          << ", per-frame timesteps " << per_frame_timesteps << std::endl;
                if (!validate_frame_batch(version, frames, per_frame_timesteps)) {
                    std::cerr << "AnimateDiff graph did not preserve the frame batch" << std::endl;
                    return 1;
                }
            }
        }
    }
    std::cout << "validated AnimateDiff SDXL, SD 1.5, and Lightning layouts and frame batches" << std::endl;
    return 0;
}
