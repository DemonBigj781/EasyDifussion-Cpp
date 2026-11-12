// Compile the actual preparation function without making its private types public API.
#include "stable-diffusion.cpp"
#include <cstdio>
#include <stdexcept>

static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct FaultConditioner : Conditioner {
    int calls = 0, fail_at = 0, done = 0;
    SDCondition get_learned_condition(int, const ConditionerParams&) override {
        if (++calls == fail_at) return {};
        SDCondition out;
        out.c_crossattn = sd::Tensor<float>::ones({4,4});
        out.c_vector = sd::Tensor<float>::ones({4});
        return out;
    }
    void get_param_tensors(std::map<std::string, ggml_tensor*>&) override {}
    void set_flash_attention_enabled(bool) override {}
    void runner_done() override { ++done; }
};
int main() {
    try {
        StableDiffusionGGML app;
        app.version = VERSION_PID;
        app.n_threads = 1;
        sd_ctx_t ctx{&app};
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.width = params.height = 64;
        params.sample_params.sample_method = EULER_SAMPLE_METHOD;
        float sigmas[] = {1.f, 0.f};
        params.sample_params.custom_sigmas = sigmas;
        params.sample_params.custom_sigmas_count = 2;
        GenerationRequest request(&ctx, &params);
        SamplePlan plan(&ctx, &params, request);
        ImageGenerationLatents latents;
        latents.concat_latent = sd::Tensor<float>::ones({2,2,1,1});
        latents.img_uncond_concat_latent = latents.concat_latent;
        RefImageParams refs;
        refs.pass_to_dit = true;
        auto conditioner = std::make_shared<FaultConditioner>();
        app.cond_stage_model = conditioner;
        for (int mode = 0; mode < 4; ++mode) {
            request.use_uncond = mode == 1;
            request.use_img_uncond = mode == 2;
            request.use_high_noise_uncond = mode == 3;
            for (int failure : {0, 1, 2}) {
                conditioner->calls = conditioner->done = 0;
                conditioner->fail_at = failure;
                auto result = prepare_image_generation_embeds(&ctx, &params, &request, &plan, &latents, refs);
                const bool should_fail = failure == 1 || (failure == 2 && mode != 0);
                require(result.has_value() != should_fail, "conditioning failure was masked by concat or ignored");
                require(conditioner->done == 1, "conditioner cleanup did not run exactly once");
            }
        }
        request.use_uncond = true; request.use_img_uncond = false; request.use_high_noise_uncond = false;
        for (auto version : {VERSION_IDEOGRAM4, VERSION_MINIT2I}) {
            app.version = version;
            conditioner->calls = conditioner->done = 0;
            conditioner->fail_at = 2;
            auto result = prepare_image_generation_embeds(&ctx, &params, &request, &plan, &latents, refs);
            require(result.has_value() && conditioner->calls == 1 && conditioner->done == 1, "special unconditional path changed");
            if (version == VERSION_IDEOGRAM4)
                require(result->uncond.c_crossattn.empty() && result->uncond.c_vector[0] == 1.f, "Ideogram vector-only condition changed");
            else {
                require(!result->uncond.c_crossattn.empty(), "MiniT2I shared positive condition missing");
                for (int64_t i = 0; i < result->uncond.c_vector.numel(); ++i)
                    require(result->uncond.c_vector[i] == 0.f, "MiniT2I unconditional mask not zero");
            }
        }
        std::puts("PASS actual image preparation: primary/negative/image-unconditional faults, concat isolation, cleanup and Ideogram/MiniT2I");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
