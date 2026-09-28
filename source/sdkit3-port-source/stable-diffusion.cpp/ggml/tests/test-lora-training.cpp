#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_opt_optimizer_params lora_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);
    ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.adamw.alpha = 0.03f;
    params.adamw.beta1 = 0.9f;
    params.adamw.beta2 = 0.999f;
    params.adamw.eps = 1e-8f;
    params.adamw.wd = 0.0f;
    return params;
}

static bool run_lora_training(ggml_backend_t backend) {
    constexpr int64_t n_in = 4;
    constexpr int64_t n_out = 3;
    constexpr int64_t rank = 2;
    constexpr int64_t ndata = 48;
    constexpr int64_t epochs = 120;

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, n_in, n_out, ndata, ndata);
    if (dataset == nullptr) {
        std::fprintf(stderr, "failed to allocate training dataset\n");
        return false;
    }

    float * inputs_data = ggml_get_data_f32(ggml_opt_dataset_data(dataset));
    float * labels_data = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));
    const float base_w[n_in * n_out] = {
        0.4f, -0.2f, 0.1f,
        0.0f,  0.3f, 0.2f,
       -0.1f,  0.1f, 0.5f,
        0.2f,  0.0f, 0.1f,
    };
    const float a_target[n_in * rank] = {
        0.5f, -0.2f,
        0.1f,  0.4f,
       -0.3f,  0.6f,
        0.2f,  0.1f,
    };
    const float b_target[rank * n_out] = {
        0.2f, -0.1f, 0.3f,
        0.1f,  0.25f, -0.2f,
    };

    std::mt19937 rng(20260927);
    std::uniform_real_distribution<float> sample(-1.0f, 1.0f);
    for (int64_t n = 0; n < ndata; ++n) {
        float x[n_in];
        for (int64_t i = 0; i < n_in; ++i) {
            x[i] = sample(rng);
            inputs_data[n * n_in + i] = x[i];
        }
        float hidden[rank] = {};
        for (int64_t r = 0; r < rank; ++r) {
            for (int64_t i = 0; i < n_in; ++i) hidden[r] += a_target[r * n_in + i] * x[i];
        }
        for (int64_t o = 0; o < n_out; ++o) {
            float y = 0.0f;
            for (int64_t i = 0; i < n_in; ++i) y += base_w[o * n_in + i] * x[i];
            for (int64_t r = 0; r < rank; ++r) y += b_target[o * rank + r] * hidden[r];
            labels_data[n * n_out + o] = y;
        }
    }
    double initial_mse = 0.0;
    for (int64_t n = 0; n < ndata; ++n) {
        for (int64_t o = 0; o < n_out; ++o) {
            float base_prediction = 0.0f;
            for (int64_t i = 0; i < n_in; ++i) {
                base_prediction += base_w[o * n_in + i] * inputs_data[n * n_in + i];
            }
            const double error = base_prediction - labels_data[n * n_out + o];
            initial_mse += error * error;
        }
    }
    initial_mse /= ndata * n_out;

    ggml_context * ctx_static = nullptr;
    ggml_context * ctx_compute = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_buffer_t static_buffer = nullptr;
    bool ok = false;

    ggml_init_params static_params = {
        /*.mem_size =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx_static = ggml_init(static_params);
    ggml_init_params compute_params = {
        /*.mem_size =*/ GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx_compute = ggml_init(compute_params);
    if (ctx_static == nullptr || ctx_compute == nullptr) {
        std::fprintf(stderr, "failed to allocate GGML contexts\n");
        goto cleanup;
    }

    {
        const int64_t input_ne[] = {n_in, ndata};
        const int64_t weight_ne[] = {n_in, n_out};
        const int64_t down_ne[] = {n_in, rank};
        const int64_t up_ne[] = {rank, n_out};
        ggml_tensor * inputs = ggml_new_tensor(ctx_static, GGML_TYPE_F32, 2, input_ne);
        ggml_tensor * base = ggml_new_tensor(ctx_static, GGML_TYPE_F32, 2, weight_ne);
        ggml_tensor * down = ggml_new_tensor(ctx_static, GGML_TYPE_F32, 2, down_ne);
        ggml_tensor * up = ggml_new_tensor(ctx_static, GGML_TYPE_F32, 2, up_ne);
        if (!inputs || !base || !down || !up) {
            std::fprintf(stderr, "failed to create LoRA graph tensors\n");
            goto cleanup;
        }
        ggml_set_name(inputs, "inputs");
        ggml_set_name(base, "frozen_base_weight");
        ggml_set_name(down, "lora_down");
        ggml_set_name(up, "lora_up");
        ggml_set_param(down);
        ggml_set_param(up);

        ggml_tensor * base_out = ggml_mul_mat(ctx_compute, base, inputs);
        ggml_tensor * hidden = ggml_mul_mat(ctx_compute, down, inputs);
        ggml_tensor * adapter_out = ggml_mul_mat(ctx_compute, up, hidden);
        ggml_tensor * outputs = ggml_add(ctx_compute, base_out, adapter_out);
        if (!base_out || !hidden || !adapter_out || !outputs) {
            std::fprintf(stderr, "failed to construct LoRA forward graph\n");
            goto cleanup;
        }
        ggml_set_name(outputs, "frozen_base_plus_lora");

        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
        cpu_backend = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
        ggml_backend_t backends[] = {backend, cpu_backend};
        sched = cpu_backend ? ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true) : nullptr;
        static_buffer = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
        if (!sched || !static_buffer) {
            std::fprintf(stderr, "failed to allocate backend tensors or scheduler\n");
            goto cleanup;
        }

        std::vector<float> down_initial(n_in * rank);
        for (float & value : down_initial) value = sample(rng) * 0.4f;
        const std::vector<float> zero_up(rank * n_out, 0.0f); // standard zero-output LoRA initialization
        ggml_backend_tensor_set(base, base_w, 0, sizeof(base_w));
        ggml_backend_tensor_set(down, down_initial.data(), 0, down_initial.size() * sizeof(float));
        ggml_backend_tensor_set(up, zero_up.data(), 0, zero_up.size() * sizeof(float));

        ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
                     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
                     GGML_OPT_OPTIMIZER_TYPE_ADAMW, lora_optimizer_params,
                     epochs, ndata, 0.0f, true);

        std::vector<float> down_after(n_in * rank);
        std::vector<float> up_after(rank * n_out);
        std::vector<float> base_after(n_in * n_out);
        ggml_backend_tensor_get(down, down_after.data(), 0, down_after.size() * sizeof(float));
        ggml_backend_tensor_get(up, up_after.data(), 0, up_after.size() * sizeof(float));
        ggml_backend_tensor_get(base, base_after.data(), 0, base_after.size() * sizeof(float));
        float down_delta = 0.0f;
        float up_delta = 0.0f;
        for (size_t i = 0; i < down_after.size(); ++i) down_delta += std::fabs(down_after[i] - down_initial[i]);
        for (size_t i = 0; i < up_after.size(); ++i) up_delta += std::fabs(up_after[i] - zero_up[i]);
        bool base_unchanged = true;
        for (size_t i = 0; i < base_after.size(); ++i) base_unchanged &= base_after[i] == base_w[i];
        double final_mse = 0.0;
        for (int64_t n = 0; n < ndata; ++n) {
            float hidden[rank] = {};
            for (int64_t r = 0; r < rank; ++r) {
                for (int64_t i = 0; i < n_in; ++i) {
                    hidden[r] += down_after[i + r * n_in] * inputs_data[n * n_in + i];
                }
            }
            for (int64_t o = 0; o < n_out; ++o) {
                float prediction = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) {
                    prediction += base_after[i + o * n_in] * inputs_data[n * n_in + i];
                }
                for (int64_t r = 0; r < rank; ++r) {
                    prediction += up_after[r + o * rank] * hidden[r];
                }
                const double error = prediction - labels_data[n * n_out + o];
                final_mse += error * error;
            }
        }
        final_mse /= ndata * n_out;
        const bool adapters_updated = down_delta > 1e-5f && up_delta > 1e-5f;
        ok = adapters_updated && base_unchanged && final_mse < initial_mse * 0.2;
        std::printf("LoRA update backend=%s mse=%.7g->%.7g down_delta=%.7g up_delta=%.7g frozen_base=%s\n",
                    ggml_backend_name(backend), initial_mse, final_mse, down_delta, up_delta,
                    base_unchanged ? "unchanged" : "CHANGED");
    }

cleanup:
    ggml_backend_buffer_free(static_buffer);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
    ggml_opt_dataset_free(dataset);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <backend-name, e.g. SYCL1 or CPU>\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    const std::string requested = argv[1];
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (requested == ggml_backend_dev_name(dev)) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                std::fprintf(stderr, "failed to initialize backend %s\n", requested.c_str());
                return 1;
            }
            const bool ok = run_lora_training(backend);
            ggml_backend_free(backend);
            return ok ? 0 : 1;
        }
    }
    std::fprintf(stderr, "backend not found: %s\n", requested.c_str());
    return 2;
}
