#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "runtime/denoiser.hpp"

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "sampler/scheduler regression failed: %s\n", message);
        std::exit(1);
    }
}

static void require_close(float actual, float expected, float tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr,
                     "sampler/scheduler regression failed: %s (actual=%g expected=%g)\n",
                     message,
                     actual,
                     expected);
        std::exit(1);
    }
}

static denoise_cb_t constant_denoiser(float value) {
    return [value](const sd::Tensor<float>&, float, int) {
        sd::guidance::GuiderOutput output;
        output.pred = sd::Tensor<float>({1}, {value});
        return output;
    };
}

static denoise_cb_t proportional_denoiser(float scale) {
    return [scale](const sd::Tensor<float>& x, float, int) {
        sd::guidance::GuiderOutput output;
        output.pred = scale * x;
        return output;
    };
}

static std::shared_ptr<RNG> seeded_rng(int64_t seed) {
    auto rng = std::make_shared<STDDefaultRNG>();
    rng->manual_seed(seed);
    return rng;
}

int main() {
    require(str_to_sample_method("dpm++3m_sde") == DPMPP3M_SDE_SAMPLE_METHOD,
            "DPM++ 3M public name round-trips");
    require(str_to_sample_method("unipc") == UNIPC_SAMPLE_METHOD,
            "UniPC public name round-trips");
    require(str_to_sample_method("deis") == DEIS_SAMPLE_METHOD,
            "DEIS public name round-trips");
    require(str_to_scheduler("ddim_uniform") == DDIM_UNIFORM_SCHEDULER,
            "DDIM Uniform public name round-trips");
    require(str_to_scheduler("linear_quadratic") == LINEAR_QUADRATIC_SCHEDULER,
            "Linear Quadratic public name round-trips");
    require(std::strcmp(sd_sample_method_name(DPMPP3M_SDE_SAMPLE_METHOD), "dpm++3m_sde") == 0,
            "DPM++ 3M enum emits its public name");
    require(std::strcmp(sd_scheduler_name(LINEAR_QUADRATIC_SCHEDULER), "linear_quadratic") == 0,
            "Linear Quadratic enum emits its public name");

    DDIMUniformScheduler ddim_uniform;
    const auto ddim_sigmas = ddim_uniform.get_sigmas(
        20, 0.f, 999.f, [](float timestep) { return timestep; });
    require(ddim_sigmas.size() == 21, "DDIM Uniform emits twenty steps plus zero");
    require_close(ddim_sigmas.front(), 951.f, 0.f, "DDIM Uniform starts at the highest stride index");
    require_close(ddim_sigmas[1], 901.f, 0.f, "DDIM Uniform uses a fixed training-table stride");
    require_close(ddim_sigmas[19], 1.f, 0.f, "DDIM Uniform retains training index one");
    require_close(ddim_sigmas.back(), 0.f, 0.f, "DDIM Uniform terminates at zero");

    LinearQuadraticScheduler linear_quadratic;
    const auto lq_sigmas = linear_quadratic.get_sigmas(
        4, 0.f, 10.f, [](float timestep) { return timestep; });
    const std::vector<float> expected_lq = {10.f, 9.875f, 9.75f, 7.25f, 0.f};
    require(lq_sigmas.size() == expected_lq.size(), "Linear Quadratic emits n plus one sigmas");
    for (size_t i = 0; i < expected_lq.size(); ++i) {
        require_close(lq_sigmas[i], expected_lq[i], 1e-5f, "Linear Quadratic reference curve");
    }

    const sd::Tensor<float> initial({1}, {5.f});
    const std::vector<float> positive_sigmas = {5.f, 4.f, 3.f, 2.f, 1.f};

    const auto dpm_deterministic = sample_dpmpp_3m_sde(
        constant_denoiser(2.f), initial, positive_sigmas, seeded_rng(7), 0.f);
    require_close(dpm_deterministic.values()[0], 2.6f, 1e-4f, "DPM++ 3M follows the constant-x0 solution");

    const std::vector<float> flow_sigmas = {0.9f, 0.7f, 0.5f, 0.3f, 0.1f};
    const auto dpm_flow = sample_dpmpp_3m_sde(
        constant_denoiser(2.f), initial, flow_sigmas, seeded_rng(7), 0.f, true);
    require_close(dpm_flow.values()[0],
                  7.f / 3.f,
                  2e-4f,
                  "DPM++ 3M applies rectified-flow alpha and half-log-SNR");

    const auto dpm_stochastic_a = sample_dpmpp_3m_sde(
        proportional_denoiser(0.25f), initial, positive_sigmas, seeded_rng(1234), 1.f);
    const auto dpm_stochastic_b = sample_dpmpp_3m_sde(
        proportional_denoiser(0.25f), initial, positive_sigmas, seeded_rng(1234), 1.f);
    const auto dpm_stochastic_c = sample_dpmpp_3m_sde(
        proportional_denoiser(0.25f), initial, positive_sigmas, seeded_rng(5678), 1.f);
    require_close(dpm_stochastic_a.values()[0],
                  dpm_stochastic_b.values()[0],
                  0.f,
                  "DPM++ 3M Brownian noise is reproducible for a fixed seed");
    require(std::abs(dpm_stochastic_a.values()[0] - dpm_stochastic_c.values()[0]) > 1e-6f,
            "DPM++ 3M Brownian noise changes with the seed");

    const auto unipc = sample_unipc(constant_denoiser(2.f), initial, positive_sigmas);
    require_close(unipc.values()[0], 2.6f, 2e-4f, "UniPC warmup/corrector preserves the constant-x0 solution");

    const auto deis = sample_deis(constant_denoiser(2.f), initial, positive_sigmas);
    require_close(deis.values()[0], 2.6f, 2e-4f, "third-order DEIS preserves the constant-x0 solution");

    const auto unipc_finite = sample_unipc(proportional_denoiser(0.25f), initial, positive_sigmas);
    const auto deis_finite  = sample_deis(proportional_denoiser(0.25f), initial, positive_sigmas);
    require(std::isfinite(unipc_finite.values()[0]), "UniPC third-order path remains finite");
    require(std::isfinite(deis_finite.values()[0]), "DEIS third-order path remains finite");

    return 0;
}
