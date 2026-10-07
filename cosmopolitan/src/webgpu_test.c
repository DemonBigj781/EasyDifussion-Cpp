#include "runtime.h"
#include "cosmo-webgpu.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include <webgpu.h>

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

static int graph_case(ggml_backend_t backend, int quantized, unsigned iteration, int strict_embedded) {
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
        (strict_embedded && cosmo_webgpu_native_loader_open_count() != 0)) {
        fputs("WebGPU graph lacks actual compute/submission/readback evidence\n", stderr);
        goto done;
    }
    const struct cosmo_webgpu_device_info *info = cosmo_webgpu_device_metadata(ggml_backend_dev_name(device));
    if (!info || (strict_embedded && (strcmp(info->provider, "embedded") || info->software != 1))) goto done;
    printf("WEBGPU_GGML_EXECUTION type=%s iteration=%u provider=%s software=%d "
           "graphs=%" PRIu64 " submissions=%" PRIu64 " dispatches=%" PRIu64
           " matmuls=%" PRIu64 " readbacks=%" PRIu64 " native_loader_opens=%lu cpu_fallback=0\n",
           type, iteration, info->provider, info->software,
           after.graphs - before.graphs, after.submissions - before.submissions,
           after.dispatches - before.dispatches, after.matmuls - before.matmuls,
           after.readbacks - before.readbacks, cosmo_webgpu_native_loader_open_count());
    failed = 0;
done:
    free(qa);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    if (failed) fprintf(stderr, "WEBGPU_GGML_CASE type=%s iteration=%u FAIL\n", type, iteration);
    return failed;
}

static int group_norm_case(ggml_backend_t backend, int inplace) {
    /* Uneven channel groups, two batches, and idle lanes exercise indexing and
       reduction barriers at the workgroup boundary. */
    enum { NE0 = 7, NE1 = 3, NE2 = 10, NE3 = 2, GROUPS = 3, ELEMENTS = NE0 * NE1 * NE2 * NE3 };
    const float eps = 1e-5f;
    const int channels_per_group = (NE2 + GROUPS - 1) / GROUPS;
    int failed = 1;
    float input[ELEMENTS], actual[ELEMENTS];
    double expected[ELEMENTS];
    ggml_backend_buffer_t buffer = NULL;
    struct ggml_context *ctx = NULL;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    ctx = ggml_init(params);
    if (!ctx) goto done;

    for (int i = 0; i < ELEMENTS; ++i)
        input[i] = (float)((i * 13 + 5) % 47 - 23) / 16.0f;

    struct ggml_tensor *src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NE0, NE1, NE2, NE3);
    ggml_set_name(src, "group-norm-input");
    struct ggml_tensor *dst = inplace ? ggml_group_norm_inplace(ctx, src, GROUPS, eps)
                                      : ggml_group_norm(ctx, src, GROUPS, eps);
    ggml_set_name(dst, "group-norm-output");
    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, dst);

    if (!device || !ggml_backend_dev_supports_op(device, dst)) {
        fprintf(stderr, "WEBGPU_GROUP_NORM_CHECK inplace=%d supported=0 FAIL\n", inplace);
        goto done;
    }

    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || ggml_backend_buffer_is_host(buffer) ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) != device) {
        fputs("WebGPU GroupNorm probe requires tensors owned by its non-host buffer type\n", stderr);
        goto done;
    }
    ggml_backend_tensor_set(src, input, 0, sizeof(input));

    for (int batch = 0; batch < NE3; ++batch) {
        for (int group = 0; group < GROUPS; ++group) {
            const int channel_begin = group * channels_per_group;
            const int channel_end = channel_begin + channels_per_group < NE2
                ? channel_begin + channels_per_group : NE2;
            const int group_channels = channel_end - channel_begin;
            const size_t group_elements = (size_t)NE0 * NE1 * group_channels;
            double sum = 0.0;
            for (int channel = channel_begin; channel < channel_end; ++channel)
                for (int row = 0; row < NE1; ++row)
                    for (int col = 0; col < NE0; ++col) {
                        const size_t i = (((size_t)batch * NE2 + channel) * NE1 + row) * NE0 + col;
                        sum += input[i];
                    }
            const double mean = sum / group_elements;
            double squared_error = 0.0;
            for (int channel = channel_begin; channel < channel_end; ++channel)
                for (int row = 0; row < NE1; ++row)
                    for (int col = 0; col < NE0; ++col) {
                        const size_t i = (((size_t)batch * NE2 + channel) * NE1 + row) * NE0 + col;
                        const double centered = input[i] - mean;
                        squared_error += centered * centered;
                    }
            const double inverse_stddev = 1.0 / sqrt(squared_error / group_elements + eps);
            for (int channel = channel_begin; channel < channel_end; ++channel)
                for (int row = 0; row < NE1; ++row)
                    for (int col = 0; col < NE0; ++col) {
                        const size_t i = (((size_t)batch * NE2 + channel) * NE1 + row) * NE0 + col;
                        expected[i] = (input[i] - mean) * inverse_stddev;
                    }
        }
    }

    const struct execution_counts before = snapshot();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) goto done;
    ggml_backend_tensor_get(dst, actual, 0, sizeof(actual));
    double maximum_error = 0.0;
    for (size_t i = 0; i < ELEMENTS; ++i) {
        const double error = fabs((double)actual[i] - expected[i]);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) || error > 5e-4 * (1.0 + fabs(expected[i]))) {
            fprintf(stderr, "WEBGPU_GROUP_NORM_CHECK inplace=%d index=%zu actual=%.9g expected=%.9g FAIL\n",
                    inplace, i, (double)actual[i], expected[i]);
            goto done;
        }
        if (error > maximum_error) maximum_error = error;
    }
    const struct execution_counts after = snapshot();
    if (after.graphs <= before.graphs || after.submissions <= before.submissions ||
        after.dispatches <= before.dispatches || after.readbacks <= before.readbacks) {
        fputs("WebGPU GroupNorm did not establish direct-device execution evidence\n", stderr);
        goto done;
    }
    printf("WEBGPU_GROUP_NORM_CHECK inplace=%d groups=%d shape=%dx%dx%dx%d values=%d "
           "max_error=%.9g dispatches=%" PRIu64 " PASS\n",
           inplace, GROUPS, NE0, NE1, NE2, NE3, ELEMENTS, maximum_error,
           after.dispatches - before.dispatches);
    failed = 0;
done:
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
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
        if (graph_case(backend, 0, iteration, 1) || graph_case(backend, 1, iteration, 1)) goto done;
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

/* Called on the original main thread, including through the native HTTP lane.
   All tensors remain on the selected WebGPU device; no scheduler exists here. */
int cosmo_webgpu_device_graph_probe(const char *selector, int require_hardware) {
    ggml_backend_t backend = NULL;
    int result = 1;
    ggml_backend_reg_t registry = ggml_backend_reg_by_name(GGML_WEBGPU_NAME);
    if (!registry || !ggml_backend_reg_dev_count(registry) ||
        cosmo_webgpu_select_device(selector)) {
        fprintf(stderr, "WEBGPU_DEVICE_TEST FAIL: %s\n", cosmo_webgpu_last_error());
        return 1;
    }
    const struct cosmo_webgpu_device_info *info = cosmo_webgpu_device_metadata("auto");
    ggml_backend_dev_t device = info ? ggml_backend_dev_by_name(info->selector) : NULL;
    if (!device || ggml_backend_dev_backend_reg(device) != registry) {
        fputs("WEBGPU_DEVICE_TEST FAIL: selected device is not registered by WebGPU\n", stderr);
        return 1;
    }
    const int hardware = !strcmp(info->provider, "native") && !info->software &&
        (info->adapter_type == WGPUAdapterType_DiscreteGPU || info->adapter_type == WGPUAdapterType_IntegratedGPU);
    const char *kind = info->software ? "software-cpu" : info->adapter_type == WGPUAdapterType_DiscreteGPU ? "discrete-gpu" :
        info->adapter_type == WGPUAdapterType_IntegratedGPU ? "integrated-gpu" : "unknown";
    if (require_hardware && !hardware) {
        fprintf(stderr, "WEBGPU_DEVICE_TEST FAIL: hardware required, selected %s provider=%s type=%s\n",
                info->selector, info->provider, kind);
        return 1;
    }
    backend = ggml_backend_dev_init(device, NULL);
    if (!backend) goto done;
    const struct execution_counts before = snapshot();
    for (unsigned iteration = 0; iteration < 2; ++iteration)
        if (graph_case(backend, 0, iteration, 0) || graph_case(backend, 1, iteration, 0)) goto done;
    if (group_norm_case(backend, 0) || group_norm_case(backend, 1)) goto done;
    const struct execution_counts after = snapshot();
    printf("WEBGPU_DEVICE_ADAPTER %s\n", info->name);
    printf("WEBGPU_DEVICE_EXECUTION selector=%s provider=%s software=%d adapter_type=%" PRIu32
           " device_kind=%s require_hardware=%d graphs=%" PRIu64 " submissions=%" PRIu64
           " dispatches=%" PRIu64 " matmuls=%" PRIu64 " readbacks=%" PRIu64
           " native_loader_opens=%lu cpu_fallback=0\n",
           info->selector, info->provider, info->software, info->adapter_type, kind, require_hardware,
           after.graphs - before.graphs, after.submissions - before.submissions,
           after.dispatches - before.dispatches, after.matmuls - before.matmuls,
           after.readbacks - before.readbacks, cosmo_webgpu_native_loader_open_count());
    result = 0;
done:
    if (backend) ggml_backend_free(backend);
    puts(result ? "WEBGPU_DEVICE_TEST FAIL" : "WEBGPU_DEVICE_TEST PASS");
    return result;
}
