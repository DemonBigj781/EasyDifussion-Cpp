#include "runtime.h"
#include "cosmo-webgpu.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K = 64, N = 11, M = 13 };

struct execution_counts {
    uint64_t graphs, submissions, dispatches, matmuls, readbacks;
};

static struct execution_counts snapshot(void) {
    return (struct execution_counts) {
        cosmo_webgpu_graph_count(), cosmo_webgpu_submission_count(),
        cosmo_webgpu_dispatch_count(), cosmo_webgpu_matmul_dispatch_count(),
        cosmo_webgpu_readback_count(),
    };
}

static int check_values(const char *type, unsigned iteration, const char *stage,
                        const float *actual, const double *expected, double tolerance) {
    double maximum = 0;
    for (size_t i = 0; i < N * M; ++i) {
        double error = fabs((double)actual[i] - expected[i]);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            error > tolerance * (1 + fabs(expected[i]))) {
            fprintf(stderr, "WEBGPU_CHECK type=%s iteration=%u stage=%s index=%zu "
                    "actual=%.9g expected=%.9g FAIL\n",
                    type, iteration, stage, i, (double)actual[i], expected[i]);
            return 1;
        }
        if (error > maximum) maximum = error;
    }
    printf("WEBGPU_CHECK type=%s iteration=%u stage=%s values=%d max_error=%.9g PASS\n",
           type, iteration, stage, N * M, maximum);
    return 0;
}

static void read_result(const struct ggml_tensor *tensor, float *out) {
    for (size_t i = 0; i < N * M; ++i) out[i] = NAN;
    ggml_backend_tensor_get(tensor, out, 0, sizeof(float) * N * M);
}

static int graph_case(ggml_backend_t backend, int quantized, unsigned iteration) {
    int failed = 1;
    const char *type = quantized ? "Q4_0" : "F32";
    float a[K * N], b[K * M], bias[N], decoded_a[K * N], out[N * M];
    double product_reference[N * M], activation[N * M], probabilities_reference[N * M];
    void *qa = NULL;
    ggml_backend_buffer_t buffer = NULL;
    struct ggml_context *ctx = NULL;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    ctx = ggml_init(params);
    if (!ctx) goto done;

    /* These dyadic inputs and their Q4_0 reconstruction are exactly f16
       representable. The preserved WebGPU tiled kernel stages even F32 inputs
       through f16; this test therefore uses a strict independent scalar oracle
       without claiming full-precision equivalence for arbitrary F32 inputs. */
    for (int i = 0; i < K * N; ++i) a[i] = (float)((i * 17 + 3 + (int)iteration * 5) % 41 - 20) / 32;
    for (int i = 0; i < K * M; ++i) b[i] = (float)((i * 11 + 7 + (int)iteration * 3) % 37 - 18) / 32;
    for (int i = 0; i < N; ++i) bias[i] = (float)(i - 5 + (int)iteration) / 16;

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

    /* A known unsupported backward operator must stay unadvertised. It is
       deliberately never submitted to another backend or emulated on CPU. */
    if (!quantized && iteration == 0) {
        struct ggml_tensor *unsupported = ggml_silu_back(ctx, sum, sum);
        if (ggml_backend_dev_supports_op(device, unsupported)) {
            fputs("WEBGPU_UNSUPPORTED op=SILU_BACK unexpectedly advertised\n", stderr);
            goto done;
        }
        puts("WEBGPU_UNSUPPORTED op=SILU_BACK supported=0");
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        struct ggml_tensor *node = ggml_graph_node(graph, i);
        if (!ggml_backend_dev_supports_op(device, node)) {
            fprintf(stderr, "WebGPU does not support required graph op %s\n", ggml_op_name(node->op));
            goto done;
        }
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || ggml_backend_buffer_is_host(buffer) ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) != device) {
        fputs("WebGPU probe requires tensors owned by its non-host buffer type\n", stderr);
        goto done;
    }
    if (quantized) {
        size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, K);
        qa = malloc(row_bytes * N);
        if (!qa || ggml_quantize_chunk(GGML_TYPE_Q4_0, a, qa, 0, N, K, NULL) != row_bytes * N) goto done;
        for (int n = 0; n < N; ++n) {
            ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float((char *)qa + n * row_bytes, decoded_a + n * K, K);
        }
        ggml_backend_tensor_set(ta, qa, 0, row_bytes * N);
    } else {
        memcpy(decoded_a, a, sizeof(a));
        ggml_backend_tensor_set(ta, a, 0, sizeof(a));
    }
    ggml_backend_tensor_set(tb, b, 0, sizeof(b));
    ggml_backend_tensor_set(tc, bias, 0, sizeof(bias));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double value = 0;
            /* WebGPU consumes the original F32 RHS. The CPU quantized-dot
               implementation's additional Q8 RHS rounding does not apply. */
            for (int k = 0; k < K; ++k) value += (double)decoded_a[n * K + k] * b[m * K + k];
            product_reference[m * N + n] = value;
        }
    }

    /* Snapshot only after device creation, buffer allocation and uploads.
       No GGML scheduler or CPU backend is created by this graph probe. */
    struct execution_counts before = snapshot();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) goto done;
    read_result(product, out);
    if (check_values(type, iteration, "matmul", out, product_reference, 3e-5)) goto done;
    for (int m = 0; m < M; ++m) {
        double squares = 0, maximum = -INFINITY, total = 0;
        for (int n = 0; n < N; ++n) {
            double value = product_reference[m * N + n] + bias[n];
            squares += value * value;
        }
        double scale = 1 / sqrt(squares / N + 1e-5);
        for (int n = 0; n < N; ++n) {
            double value = (product_reference[m * N + n] + bias[n]) * scale;
            activation[m * N + n] = value / (1 + exp(-value));
            if (activation[m * N + n] > maximum) maximum = activation[m * N + n];
        }
        for (int n = 0; n < N; ++n) {
            probabilities_reference[m * N + n] = exp(activation[m * N + n] - maximum);
            total += probabilities_reference[m * N + n];
        }
        for (int n = 0; n < N; ++n) probabilities_reference[m * N + n] /= total;
    }
    read_result(activated, out);
    if (check_values(type, iteration, "bias-rmsnorm-silu", out, activation, 4e-5)) goto done;
    read_result(probabilities, out);
    if (check_values(type, iteration, "softmax", out, probabilities_reference, 4e-5)) goto done;
    struct execution_counts after = snapshot();
    if (after.graphs <= before.graphs || after.submissions <= before.submissions ||
        after.dispatches <= before.dispatches || after.matmuls <= before.matmuls ||
        after.readbacks < before.readbacks + 3 ||
        cosmo_webgpu_native_loader_open_count() != 0) {
        fputs("WebGPU graph lacks actual embedded compute/submission/readback evidence\n", stderr);
        goto done;
    }
    printf("WEBGPU_GGML_EXECUTION type=%s iteration=%u provider=embedded software=1 "
           "graphs=%" PRIu64 " submissions=%" PRIu64 " dispatches=%" PRIu64
           " matmuls=%" PRIu64 " readbacks=%" PRIu64 " native_loader_opens=0 cpu_fallback=0\n",
           type, iteration, after.graphs - before.graphs, after.submissions - before.submissions,
           after.dispatches - before.dispatches, after.matmuls - before.matmuls,
           after.readbacks - before.readbacks);
    failed = 0;
done:
    free(qa);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    if (failed) fprintf(stderr, "WEBGPU_GGML_CASE type=%s iteration=%u FAIL\n", type, iteration);
    return failed;
}

int cosmo_webgpu_selftest(void) {
    int failed = 1;
    ggml_backend_t backend = NULL;
    if (cosmo_webgpu_initialize() != 0) {
        fputs("Embedded WebGPU provider initialization failed\n", stderr);
        goto done;
    }
    backend = ggml_backend_webgpu_init();
    if (!backend) {
        fputs("GGML WebGPU backend initialization failed\n", stderr);
        goto done;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), GGML_WEBGPU_NAME) ||
        strcmp(cosmo_webgpu_provider_name(), "embedded") ||
        cosmo_webgpu_adapter_is_software() != 1 || !*cosmo_webgpu_adapter_name() ||
        cosmo_webgpu_native_loader_open_count() != 0) {
        fputs("GGML probe did not initialize the embedded software WebGPU backend\n", stderr);
        goto done;
    }
    printf("WEBGPU_ADAPTER name=%s provider=embedded software=1 native_loader_opens=0\n",
           cosmo_webgpu_adapter_name());
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        if (graph_case(backend, 0, iteration) || graph_case(backend, 1, iteration)) goto done;
    }
    failed = 0;
done:
    if (backend) ggml_backend_free(backend);
    if (failed) {
        fputs("WEBGPU_GGML_SELFTEST FAIL\n", stderr);
        return 1;
    }
    puts("WEBGPU_GGML_SELFTEST PASS");
    return 0;
}
