#include <cstring>
#include <iostream>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

int main(int argc, char** argv) {
    const bool abort_compute = argc == 2 && std::strcmp(argv[1], "abort") == 0;
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    auto scheduler = ggml_backend_sched_new(&backend, nullptr, 1, 2048, false, false);
    auto persistent = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
    auto input = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, 1, 1);
    auto weight = ggml_new_tensor_1d(persistent, GGML_TYPE_F32, 1);
    ggml_set_param(weight);
    auto buffer = ggml_backend_alloc_ctx_tensors(persistent, backend);
    const float one = 1, zero = 0;
    ggml_backend_tensor_set(input, &one, 0, sizeof(one));
    ggml_backend_tensor_set(weight, &zero, 0, sizeof(zero));
    auto compute = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto output = ggml_mul(compute, input, weight);
    auto params = ggml_opt_default_params(scheduler, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    params.ctx_compute = compute; params.inputs = input; params.outputs = output;
    auto optimizer = ggml_opt_init(params);
    ggml_backend_tensor_set(ggml_opt_labels(optimizer), &one, 0, sizeof(one));
    if (abort_compute) {
        ggml_backend_cpu_set_abort_callback(backend, [](void*) { return true; }, nullptr);
    }
    ggml_opt_alloc(optimizer, true);
    ggml_opt_eval(optimizer, nullptr);
    std::cout << (abort_compute ? "UNEXPECTED_SUCCESS_AFTER_ABORT" : "SUCCESS") << std::endl;
    ggml_opt_free(optimizer); ggml_backend_sched_free(scheduler); ggml_free(compute);
    ggml_backend_buffer_free(buffer); ggml_free(persistent); ggml_backend_free(backend);
    return 0;
}
