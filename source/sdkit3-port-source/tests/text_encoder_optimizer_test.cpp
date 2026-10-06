#include <cmath>
#include <array>
#include <cstring>
#include <iostream>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

static float parameter_rate(const ggml_tensor* tensor, void*) {
    return std::strcmp(tensor->name, "text_encoder") == 0 ? 0.1f : 1.0f;
}

static std::array<float, 3> exercise(bool sample) {
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    auto sched = ggml_backend_sched_new(&backend, nullptr, 1, 2048, false, true);
    auto weights = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
    auto input = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 1, 1);
    auto unet = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 1);
    auto text = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 1);
    ggml_set_name(unet, "unet");
    ggml_set_name(text, "text_encoder");
    ggml_set_param(unet);
    ggml_set_param(text);
    auto buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
    float one = 1.0f, zero = 0.0f;
    ggml_backend_tensor_set(input, &one, 0, sizeof(one));
    ggml_backend_tensor_set(unet, &zero, 0, sizeof(zero));
    ggml_backend_tensor_set(text, &zero, 0, sizeof(zero));
    auto compute = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto output = ggml_add(compute, ggml_mul(compute, input, unet), ggml_mul(compute, input, text));
    auto params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    params.ctx_compute = compute;
    params.inputs = input;
    params.outputs = output;
    params.get_param_lr_scale = parameter_rate;
    auto rates = ggml_opt_get_default_optimizer_params(nullptr);
    rates.adamw.alpha = 5e-5f;
    params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    params.get_opt_pars_ud = &rates;
    auto optimizer = ggml_opt_init(params);
    ggml_backend_tensor_set(ggml_opt_labels(optimizer), &one, 0, sizeof(one));
    bool good = true;
    float previous_u = 0, previous_t = 0;
    for (float factor : {0.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.0f}) {
        rates.adamw.alpha = 5e-5f * factor;
        ggml_opt_alloc(optimizer, true);
        ggml_opt_eval(optimizer, nullptr);
        float u, t;
        ggml_backend_tensor_get(unet, &u, 0, sizeof(u));
        ggml_backend_tensor_get(text, &t, 0, sizeof(t));
        good = good && std::isfinite(u) && std::isfinite(t);
        if (factor == 0) good = good && u == previous_u && t == previous_t;
        else good = good && u > previous_u && t > previous_t && std::abs(t / u - .1f) < 1e-5f;
        previous_u = u;
        previous_t = t;
        // Sampling must execute the current forward graph without an optimizer
        // update, then allow the next training step to use the same AdamW state.
        if (sample) {
            ggml_opt_alloc(optimizer, false);
            ggml_opt_eval(optimizer, nullptr);
            float sampled, after_u, after_t;
            ggml_backend_tensor_get(ggml_opt_outputs(optimizer), &sampled, 0, sizeof(sampled));
            ggml_backend_tensor_get(unet, &after_u, 0, sizeof(after_u));
            ggml_backend_tensor_get(text, &after_t, 0, sizeof(after_t));
            good = good && after_u == u && after_t == t && std::abs(sampled - u - t) < 1e-7f;
        }
    }
    ggml_opt_free(optimizer);
    ggml_backend_sched_free(sched);
    ggml_free(compute);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights);
    ggml_backend_free(backend);
    return {good ? 1.0f : 0.0f, previous_u, previous_t};
}

int main() {
    const auto baseline = exercise(false);
    const auto sampled = exercise(true);
    const bool good = baseline[0] == 1.0f && baseline == sampled;
    std::cout << (good ? "PASS" : "FAIL") << ": separate AdamW rates; sampling leaves weights and subsequent updates unchanged\n";
    return good ? 0 : 1;
}
