#include <cstdio>
#include <cstdlib>
#include "runtime/sampling-optimizations.hpp"
#include "runtime/guidance.h"

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "sampling optimization: %s\n", message); std::exit(1); }
}
int main() {
    auto disabled = SamplingOptimizations::parse(nullptr);
    require(disabled && disabled->tome_ratio == 0 && !disabled->skip_negative(0, 2), "default path is unchanged");
    auto enabled = SamplingOptimizations::parse("tome_ratio=0.5,negative_guidance_min_sigma=1");
    require(enabled && enabled->tome_ratio == .5f, "valid parameters");
    require(enabled->cache_compatible(SD_CACHE_DISABLED), "NGMS accepts uncached sampling");
    require(!enabled->cache_compatible(SD_CACHE_EASYCACHE) && !enabled->cache_compatible(SD_CACHE_TEACACHE), "NGMS rejects caches with incompatible condition history");
    require(disabled->cache_compatible(SD_CACHE_EASYCACHE), "zero NGMS leaves caches available");
    require(!enabled->skip_negative(1.f, 2), "threshold boundary is strict");
    require(!enabled->skip_negative(.5f, 1), "first evaluation retains CFG");
    require(enabled->skip_negative(.5f, 2), "eligible alternate evaluation skips negative");
    require(!enabled->skip_negative(.5f, -2), "sampler correction evaluations retain CFG");
    require(!enabled->skip_negative(INFINITY, 2), "nonfinite sigma cannot enable skipping");
    require(!enabled->skip_negative(-1.f, 2), "negative sigma cannot enable skipping");
    for (auto value : {"tome_ratio=-1", "tome_ratio=0.51", "tome_ratio=nan", "negative_guidance_min_sigma=inf", "negative_guidance_min_sigma=-1"})
        require(!SamplingOptimizations::parse(value), "invalid settings must fail");
    sd::Tensor<float> positive({2}, {2, 4}), negative({2}, {1, 2});
    sd::guidance::GuidanceInput input;
    input.pred_cond = &positive;
    sd::guidance::ClassifierFreeGuidance cfg(7, 1);
    require(cfg.forward(input, {}).pred.values() == positive.values(), "skipped negative means conditional prediction, never stale CFG");
    input.pred_uncond = &negative;
    require(cfg.forward(input, {}).pred.values() == std::vector<float>({8, 16}), "ordinary CFG remains unchanged");
    std::puts("Sampling optimization validation and NGMS policy: PASS");
}
