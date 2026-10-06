#include "runtime.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { K = 64, N = 11, M = 13 };

static int check_values(const char *name, const float *actual,
                        const double *expected, size_t count, double tolerance) {
    double maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        double error = fabs(actual[i] - expected[i]);
        if (!isfinite(actual[i]) || error > tolerance * (1 + fabs(expected[i]))) {
            fprintf(stderr, "%s mismatch at %zu: %.9g versus %.9g\n",
                    name, i, (double)actual[i], expected[i]);
            return 1;
        }
        if (error > maximum) maximum = error;
    }
    printf("GGML_CHECK %s values=%zu max_error=%.9g PASS\n", name, count, maximum);
    return 0;
}

static void reference_matmul(const float *a, const float *b, double *out) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0;
            for (int k = 0; k < K; ++k) sum += (double)a[n * K + k] * b[m * K + k];
            out[m * N + n] = sum;
        }
    }
}

static int graph_case(int threads, int quantized) {
    int failed = 1;
    float a[K * N], b[K * M], bias[N], out[N * M];
    float a_reference[K * N], b_reference[K * M];
    double expected[N * M], activation[N * M], softmax[N * M];
    void *qa = NULL, *qb = NULL;
    ggml_backend_buffer_t buffer = NULL;
    ggml_backend_t backend = ggml_backend_cpu_init();
    struct ggml_context *ctx = NULL;
    if (!backend || !ggml_backend_is_cpu(backend)) goto done;
    ggml_backend_cpu_set_n_threads(backend, threads);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    ctx = ggml_init(params);
    if (!ctx) goto done;
    for (int i = 0; i < K * N; ++i) a[i] = (float)((i * 17 + 3) % 41 - 20) / 32;
    for (int i = 0; i < K * M; ++i) b[i] = (float)((i * 11 + 7) % 37 - 18) / 32;
    for (int i = 0; i < N; ++i) bias[i] = (float)(i - 5) / 16;

    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, quantized ? GGML_TYPE_Q4_0 : GGML_TYPE_F32, K, N);
    struct ggml_tensor *tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor *tc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    struct ggml_tensor *product = ggml_mul_mat(ctx, ta, tb);
    struct ggml_tensor *sum = ggml_add(ctx, product, tc);
    struct ggml_tensor *normalized = ggml_rms_norm(ctx, sum, 1e-5f);
    struct ggml_tensor *activated = ggml_silu(ctx, normalized);
    struct ggml_tensor *probabilities = ggml_soft_max(ctx, activated);
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, probabilities);
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) goto done;

    if (quantized) {
        const enum ggml_type dot_type = ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot_type;
        const size_t a_row = ggml_row_size(GGML_TYPE_Q4_0, K);
        const size_t b_row = ggml_row_size(dot_type, K);
        qa = malloc(a_row * N);
        qb = malloc(b_row * M);
        if (!qa || !qb) goto done;
        if (ggml_quantize_chunk(GGML_TYPE_Q4_0, a, qa, 0, N, K, NULL) != a_row * N ||
            ggml_quantize_chunk(dot_type, b, qb, 0, M, K, NULL) != b_row * M) goto done;
        for (int n = 0; n < N; ++n) {
            ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float((char *)qa + n * a_row, a_reference + n * K, K);
        }
        for (int m = 0; m < M; ++m) {
            ggml_get_type_traits(dot_type)->to_float((char *)qb + m * b_row, b_reference + m * K, K);
        }
        reference_matmul(a_reference, b_reference, expected);
        ggml_backend_tensor_set(ta, qa, 0, a_row * N);
    } else {
        reference_matmul(a, b, expected);
        ggml_backend_tensor_set(ta, a, 0, sizeof(a));
    }
    ggml_backend_tensor_set(tb, b, 0, sizeof(b));
    ggml_backend_tensor_set(tc, bias, 0, sizeof(bias));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) goto done;
    printf("GGML_CASE backend=%s threads=%d type=%s\n", ggml_backend_name(backend), threads, quantized ? "Q4_0" : "F32");
    ggml_backend_tensor_get(product, out, 0, sizeof(out));
    if (check_values("matmul", out, expected, N * M, 3e-5)) goto done;

    for (int m = 0; m < M; ++m) {
        double squares = 0, maximum = -INFINITY, total = 0;
        for (int n = 0; n < N; ++n) {
            expected[m * N + n] += bias[n];
            squares += expected[m * N + n] * expected[m * N + n];
        }
        double scale = 1 / sqrt(squares / N + 1e-5);
        for (int n = 0; n < N; ++n) {
            double value = expected[m * N + n] * scale;
            activation[m * N + n] = value / (1 + exp(-value));
            if (activation[m * N + n] > maximum) maximum = activation[m * N + n];
        }
        for (int n = 0; n < N; ++n) {
            softmax[m * N + n] = exp(activation[m * N + n] - maximum);
            total += softmax[m * N + n];
        }
        for (int n = 0; n < N; ++n) softmax[m * N + n] /= total;
    }
    ggml_backend_tensor_get(activated, out, 0, sizeof(out));
    if (check_values("bias-rmsnorm-silu", out, activation, N * M, 4e-5)) goto done;
    ggml_backend_tensor_get(probabilities, out, 0, sizeof(out));
    if (check_values("softmax", out, softmax, N * M, 4e-5)) goto done;
    failed = 0;
done:
    free(qa);
    free(qb);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    if (backend) ggml_backend_free(backend);
    if (failed) fprintf(stderr, "GGML_CASE threads=%d quantized=%d FAIL\n", threads, quantized);
    return failed;
}

int cosmo_ggml_selftest(void) {
    if (graph_case(1, 0) || graph_case(4, 0) || graph_case(1, 1) || graph_case(4, 1)) return 1;
    puts("GGML_SELFTEST PASS");
    return 0;
}
