#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static bool test_activation(ggml_backend_t backend, const std::string& kind) {
    const int n = 32 * 77;
    auto data_context = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
    auto x = ggml_new_tensor_2d(data_context, GGML_TYPE_F32, 32, 77);
    ggml_set_param(x);
    auto buffer = ggml_backend_alloc_ctx_tensors(data_context, backend);
    std::vector<float> values(n), expected(n), actual(n);
    for (int i = 0; i < n; ++i) {
        const double value = 12.0 * (double(i) / (n - 1) - .5);
        values[i] = float(value);
        const double k = kind == "gelu_quick" ? 1.702 : 1.0;
        if (kind == "gelu") {
            const double a = .044715, c = .7978845608028654;
            const double t = std::tanh(c * value * (1 + a * value * value));
            expected[i] = float((.5 * (1 + t) + .5 * value * (1 - t * t) * c * (1 + 3 * a * value * value)) / n);
        } else {
            const double s = 1 / (1 + std::exp(-k * value));
            expected[i] = float(s * (1 + k * value * (1 - s)) / n);
        }
    }
    ggml_backend_tensor_set(x, values.data(), 0, n * sizeof(float));
    auto compute = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto y = kind == "gelu" ? ggml_gelu(compute, x) : kind == "gelu_quick" ? ggml_gelu_quick(compute, x) : ggml_silu(compute, x);
    auto cpu = ggml_backend_is_cpu(backend) ? nullptr : ggml_backend_cpu_init();
    if (cpu) ggml_backend_cpu_set_n_threads(cpu, 1);
    ggml_backend_t backends[] = {backend, cpu};
    auto host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    ggml_backend_buffer_type_t types[] = {ggml_backend_get_default_buffer_type(backend),
        host ? host : (cpu ? ggml_backend_get_default_buffer_type(cpu) : nullptr)};
    auto scheduler = ggml_backend_sched_new(backends, types, cpu ? 2 : 1, 2048, false, false);
    if (cpu) ggml_backend_sched_set_allow_cpu_fallback(scheduler, false);
    auto params = ggml_opt_default_params(scheduler, GGML_OPT_LOSS_TYPE_MEAN);
    params.ctx_compute = compute; params.inputs = x; params.outputs = y;
    params.opt_period = 2; // First evaluation accumulates gradients, without updating x.
    auto opt = ggml_opt_init(params);
    ggml_opt_alloc(opt, true); ggml_opt_eval(opt, nullptr);
    auto gradient = ggml_opt_grad_acc(opt, x);
    require(gradient != nullptr, "missing gradient accumulator");
    ggml_backend_tensor_get(gradient, actual.data(), 0, n * sizeof(float));
    double max_error = 0, max_abs = 0;
    size_t nonfinite = 0;
    for (int i = 0; i < n; ++i) {
        // Loss is divided by the accumulation period.
        if (!std::isfinite(actual[i])) ++nonfinite;
        max_abs = std::max(max_abs, std::abs(double(actual[i])));
        max_error = std::max(max_error, std::abs(double(actual[i]) - expected[i] / 2));
    }
    const bool pass = nonfinite == 0 && max_error < 1e-6;
    std::cout << "ACTIVATION backend=" << ggml_backend_name(backend) << " kind=" << kind
              << " max_gradient=" << max_abs << " max_error=" << max_error << " nonfinite=" << nonfinite
              << " result=" << (pass ? "PASS" : "FAIL") << std::endl;
    ggml_opt_free(opt); ggml_backend_sched_free(scheduler); ggml_free(compute);
    ggml_backend_buffer_free(buffer); ggml_free(data_context);
    if (cpu) ggml_backend_free(cpu);
    return pass;
}

int main(int argc, char** argv) {
    try {
        ggml_backend_load_all();
        bool pass = true;
        require(argc <= 2, "Usage: backward-math-test [GPU backend name]");
        std::vector<std::string> devices{"CPU"};
        if (argc == 2 && std::string(argv[1]) != "CPU") devices.emplace_back(argv[1]);
        for (const std::string& name : devices) {
            ggml_backend_t backend = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto device = ggml_backend_dev_get(i);
                if (name == ggml_backend_dev_name(device)) backend = ggml_backend_dev_init(device, nullptr);
            }
            require(backend != nullptr, "required backend unavailable");
            if (name == "CPU") ggml_backend_cpu_set_n_threads(backend, 1);
            for (const std::string kind : {"gelu", "gelu_quick", "silu"}) pass = test_activation(backend, kind) && pass;
            ggml_backend_free(backend);
        }
        return pass ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}
