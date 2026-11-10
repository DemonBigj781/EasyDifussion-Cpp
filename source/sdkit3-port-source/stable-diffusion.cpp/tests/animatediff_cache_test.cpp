#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "runtime/sample-cache.h"

static void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "AnimateDiff cache regression: %s\n", message);
        std::exit(1);
    }
}

int main() {
    std::string logs;
    sd_set_log_callback([](sd_log_level_t, const char* message, void* data) {
        static_cast<std::string*>(data)->append(message);
    }, &logs);
    CompVisDenoiser denoiser;
    for (int i = 0; i < TIMESTEPS; ++i) {
        denoiser.sigmas[i] = .01f + i * .01f;
        denoiser.log_sigmas[i] = std::log(denoiser.sigmas[i]);
    }
    const std::vector<float> sigmas = {10.f, 8.f, 6.f, 4.f, 2.f, 1.f, 0.f};
    int positive = 1, negative = 2;
    for (auto mode : {SD_CACHE_EASYCACHE, SD_CACHE_TEACACHE}) {
        sd_cache_params_t params{};
        params.mode = mode;
        params.reuse_threshold = INFINITY;
        params.start_percent = 0.f;
        params.end_percent = 1.f;
        for (auto version : {VERSION_SD1, VERSION_SDXL}) {
            auto cache = sd_sample::init_sample_cache_runtime(version, &params, &denoiser, sigmas, "test", true);
            require(cache.easycache_enabled() == (mode == SD_CACHE_EASYCACHE), "EasyCache selection is exclusive");
            require(cache.teacache_enabled() == (mode == SD_CACHE_TEACACHE), "TeaCache selection is exclusive");
            require(cache.easycache_enabled() ? cache.easycache.config.reuse_threshold == .2f
                                              : cache.teacache.config.reuse_threshold == .2f, "UNet threshold defaults to 0.2");
            if (cache.teacache_enabled()) require(cache.teacache.config.coefficients.empty(), "No DiT polynomial is used for AnimateDiff");
            int hits = 0;
            for (int step = 1; step <= 6; ++step) {
                sd_sample::SampleStepCacheDispatcher dispatcher(cache, step, step == 6 ? 0.f : sigmas[step - 1]);
                // Four-frame UNet batch, with a small change between denoising evaluations.
                sd::Tensor<float> input({2, 2, 4, 4});
                std::fill(input.values().begin(), input.values().end(), 10.f + step * .001f);
                for (auto condition : {&positive, &negative}) {
                    sd::Tensor<float> output;
                    const bool reused = dispatcher.before_condition(condition, input, &output);
                    const float residual = condition == &positive ? 2.f : 5.f;
                    if (step == 1 || step == 6) require(!reused, "First and terminal evaluations are exact");
                    if (reused) {
                        ++hits;
                        require(output.shape() == input.shape(), "Reuse preserves the full video tensor shape");
                        for (size_t i = 0; i < input.values().size(); ++i)
                            require(std::abs(output.values()[i] - input.values()[i] - residual) < 1e-4f,
                                    "Positive and negative residuals must not leak into each other");
                    } else {
                        output = input;
                        for (auto& value : output.values()) value += residual;
                        dispatcher.after_condition(condition, input, output);
                    }
                }
            }
            require(hits > 0, "Selected cache actually reuses evaluations");
            sd_sample::log_sample_cache_summary(cache, 6);
            auto fresh = sd_sample::init_sample_cache_runtime(version, &params, &denoiser, sigmas, "test", true);
            sd::Tensor<float> output, input({2}, {10.f, 10.f});
            sd_sample::SampleStepCacheDispatcher fresh_step(fresh, 3, 6.f);
            require(!fresh_step.before_condition(&positive, input, &output), "New generations cannot reuse old residuals");
            auto still = sd_sample::init_sample_cache_runtime(version, &params, &denoiser, sigmas, "test", false);
            require(still.mode == sd_sample::SampleCacheMode::NONE, "Still-image UNet eligibility is unchanged");
        }
        auto unsupported = sd_sample::init_sample_cache_runtime(VERSION_SD2, &params, &denoiser, sigmas, "test", true);
        require(unsupported.mode == sd_sample::SampleCacheMode::NONE, "SD2 is not enabled by the AnimateDiff cache context");
        auto dit = sd_sample::init_sample_cache_runtime(VERSION_WAN2, &params, &denoiser, sigmas, "Wan 1.3B", false);
        require(dit.mode != sd_sample::SampleCacheMode::NONE, "Existing DiT cache support remains enabled");
        params.reuse_threshold = 0.f;
        auto exact = sd_sample::init_sample_cache_runtime(VERSION_SD1, &params, &denoiser, sigmas, "test", true);
        for (int step = 1; step <= 5; ++step) {
            sd_sample::SampleStepCacheDispatcher dispatcher(exact, step, sigmas[step - 1]);
            sd::Tensor<float> output, input({2}, {10.f + step * .001f, 20.f + step * .001f});
            require(!dispatcher.before_condition(&positive, input, &output), "Zero threshold must never reuse");
            output = input;
            for (auto& value : output.values()) value += 2.f;
            dispatcher.after_condition(&positive, input, output);
        }
        params.start_percent = .8f;
        params.end_percent = .2f;
        auto invalid = sd_sample::init_sample_cache_runtime(VERSION_SD1, &params, &denoiser, sigmas, "test", true);
        require(invalid.mode == sd_sample::SampleCacheMode::NONE, "Invalid cache ranges are rejected");
        params.mode = SD_CACHE_DISABLED;
        auto disabled = sd_sample::init_sample_cache_runtime(VERSION_SD1, &params, &denoiser, sigmas, "test", true);
        require(disabled.mode == sd_sample::SampleCacheMode::NONE, "Disabled means no cache");
    }
    require(logs.find("EasyCache enabled") != std::string::npos && logs.find("EasyCache skipped") != std::string::npos,
            "EasyCache activation and reuse are logged");
    require(logs.find("TeaCache enabled") != std::string::npos && logs.find("TeaCache reused") != std::string::npos,
            "TeaCache activation and reuse are logged");
    sd_set_log_callback(nullptr, nullptr);
    std::puts("AnimateDiff independent EasyCache/TeaCache runtime: OK");
}
