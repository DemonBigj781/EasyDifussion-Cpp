/* SPDX-License-Identifier: MIT */
#include "runtime.h"
#include "cosmo-webgpu.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-webgpu.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { SIDE = 4, CHANNELS = 8, GROUPS = 4, SPATIAL = SIDE * SIDE,
       ELEMENTS = SPATIAL * CHANNELS, OUTPUTS = 3, GRAPH_SIZE = 64 };

struct inplace_counts {
    uint64_t graphs, submissions, dispatches, matmuls, readbacks;
};

static struct inplace_counts counts(void) {
    return (struct inplace_counts) {
        cosmo_webgpu_graph_count(), cosmo_webgpu_submission_count(),
        cosmo_webgpu_dispatch_count(), cosmo_webgpu_matmul_dispatch_count(),
        cosmo_webgpu_readback_count(),
    };
}

static struct ggml_context * context(void) {
    const struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * GRAPH_SIZE +
                    ggml_graph_overhead_custom(GRAPH_SIZE, false),
        .no_alloc = true,
    };
    return ggml_init(params);
}

static int on_device(const struct ggml_tensor * tensor, ggml_backend_t backend, int host) {
    if (!tensor->buffer || !tensor->data ||
        ggml_backend_buffer_is_host(tensor->buffer) != (bool)host) return 0;
    ggml_backend_buffer_type_t type = ggml_backend_buffer_get_type(tensor->buffer);
    /* This GGML snapshot's ordinary CPU buffer type deliberately has device
       NULL. Its exact default type and CPU support identify that allocation;
       GPU buffers must still identify the actual WebGPU device. */
    return host ? type == ggml_backend_get_default_buffer_type(backend) &&
                      ggml_backend_supports_buft(backend, type)
                : ggml_backend_buft_get_device(type) == ggml_backend_get_device(backend);
}

static void describe_tensor(ggml_backend_sched_t scheduler, const char * label,
                            struct ggml_tensor * tensor, ggml_backend_t expected) {
    ggml_backend_t assigned = ggml_backend_sched_get_tensor_backend(scheduler, tensor);
    ggml_backend_buffer_type_t type = tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : NULL;
    ggml_backend_dev_t buffer_device = type ? ggml_backend_buft_get_device(type) : NULL;
    ggml_backend_dev_t expected_device = ggml_backend_get_device(expected);
    fprintf(stderr, "WEBGPU_INPLACE_STORAGE label=%s op=%s assigned=%s(%p) expected=%s(%p) "
            "buffer=%p buft=%s(%p) host=%d buft_device=%s(%p) expected_device=%s(%p) "
            "data=%p view_src=%p view_offset=%zu\n", label, ggml_op_name(tensor->op),
            assigned ? ggml_backend_name(assigned) : "none", (void *)assigned,
            ggml_backend_name(expected), (void *)expected, (void *)tensor->buffer,
            type ? ggml_backend_buft_name(type) : "none", (void *)type,
            tensor->buffer ? (int)ggml_backend_buffer_is_host(tensor->buffer) : -1,
            buffer_device ? ggml_backend_dev_name(buffer_device) : "none", (void *)buffer_device,
            expected_device ? ggml_backend_dev_name(expected_device) : "none", (void *)expected_device,
            tensor->data, (void *)tensor->view_src, tensor->view_offs);
}

static int compare(const char * stage, const float * actual, const double * expected,
                   size_t length, double tolerance, int preallocated, unsigned iteration) {
    double maximum = 0;
    for (size_t i = 0; i < length; ++i) {
        const double error = fabs((double)actual[i] - expected[i]);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            error > tolerance * (1 + fabs(expected[i]))) {
            fprintf(stderr, "WEBGPU_INPLACE_CHECK owner=%s iteration=%u stage=%s index=%zu "
                    "actual=%.9g expected=%.9g FAIL\n",
                    preallocated ? "preallocated" : "scheduler", iteration,
                    stage, i, (double)actual[i], expected[i]);
            return 1;
        }
        if (error > maximum) maximum = error;
    }
    printf("WEBGPU_INPLACE_CHECK owner=%s iteration=%u stage=%s values=%zu max_error=%.9g PASS\n",
           preallocated ? "preallocated" : "scheduler", iteration, stage, length, maximum);
    return 0;
}

/* Match the public operations in scalar C. Group normalization accumulates in
   double, with F32 intermediate normalization/affine/SILU results. The preserved
   WebGPU register-tiled matrix kernel stages both operands through F16. */
static void reference(const float * input, const float * scale, const float * bias,
                      const float * projection, double * activation, double * product) {
    const int group_elements = ELEMENTS / GROUPS;
    for (int group = 0; group < GROUPS; ++group) {
        double sum = 0, squares = 0;
        const int start = group * group_elements;
        for (int i = start; i < start + group_elements; ++i) sum += input[i];
        const float mean = (float)(sum / group_elements);
        for (int i = start; i < start + group_elements; ++i) {
            const float centered = input[i] - mean;
            squares += (float)(centered * centered);
        }
        const float variance = (float)(squares / group_elements);
        const float inverse = 1.0f / sqrtf(variance + 1e-5f);
        for (int i = start; i < start + group_elements; ++i) {
            const int channel = i / SPATIAL;
            float value = (input[i] - mean) * inverse;
            value *= scale[channel];
            value += bias[channel];
            activation[i] = value / (1.0f + expf(-value));
        }
    }
    for (int channel = 0; channel < CHANNELS; ++channel) {
        for (int output = 0; output < OUTPUTS; ++output) {
            double sum = 0;
            for (int k = 0; k < SPATIAL; ++k) {
                const float left = ggml_fp16_to_fp32(ggml_fp32_to_fp16(projection[output * SPATIAL + k]));
                const float right = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float)activation[channel * SPATIAL + k]));
                sum += (double)left * right;
            }
            product[channel * OUTPUTS + output] = sum;
        }
    }
}

static int graph_case(ggml_backend_t webgpu, ggml_backend_t cpu, int preallocated, unsigned iteration) {
    int failed = 1;
    struct ggml_context * ctx = context(), * weights = context();
    ggml_backend_sched_t scheduler = NULL;
    ggml_backend_buffer_t weight_buffer = NULL, owner_buffer = NULL;
    float input[ELEMENTS], scale[CHANNELS], bias[CHANNELS], projection[SPATIAL * OUTPUTS];
    float actual_activation[ELEMENTS], actual_product[CHANNELS * OUTPUTS];
    double expected_activation[ELEMENTS], expected_product[CHANNELS * OUTPUTS];
    if (!ctx || !weights) goto done;
    for (int i = 0; i < ELEMENTS; ++i)
        input[i] = (float)((i * 13 + 5 + (int)iteration * 7) % 43 - 21) / 16;
    for (int i = 0; i < CHANNELS; ++i) {
        scale[i] = (float)(i + 3) / 8;
        bias[i] = (float)(i - 4 + (int)iteration) / 16;
    }
    for (int i = 0; i < SPATIAL * OUTPUTS; ++i)
        projection[i] = (float)((i * 7 + 3 + (int)iteration) % 19 - 9) / 16;
    reference(input, scale, bias, projection, expected_activation, expected_product);

    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, SIDE, SIDE, CHANNELS);
    struct ggml_tensor * normalized = ggml_group_norm(ctx, a, GROUPS, 1e-5f);
    ggml_set_name(a, "inplace_input");
    ggml_set_name(normalized, "inplace_owner_group_norm");
    ggml_set_input(a);
    /* Allocate only the input and normalization owner in this variant. Later
       views must honor that actual CPU allocation despite conflicting pins. */
    if (preallocated) {
        owner_buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        if (!owner_buffer || !on_device(normalized, cpu, 1)) goto done;
    }
    struct ggml_tensor * multiplier = ggml_new_tensor_3d(weights, GGML_TYPE_F32, 1, 1, CHANNELS);
    struct ggml_tensor * offset = ggml_new_tensor_3d(weights, GGML_TYPE_F32, 1, 1, CHANNELS);
    struct ggml_tensor * matrix = ggml_new_tensor_2d(weights, GGML_TYPE_F32, SPATIAL, OUTPUTS);
    struct ggml_tensor * multiplied = ggml_mul_inplace(ctx, normalized, multiplier);
    struct ggml_tensor * added = ggml_add_inplace(ctx, multiplied, offset);
    struct ggml_tensor * activated = ggml_silu_inplace(ctx, added);
    struct ggml_tensor * flat = ggml_reshape_2d(ctx, activated, SPATIAL, CHANNELS);
    struct ggml_tensor * product = ggml_mul_mat(ctx, matrix, flat);
    struct ggml_tensor * aliases[] = {normalized, multiplied, added, activated, flat};
    const char * names[] = {"group_norm", "mul_inplace", "add_inplace", "silu_inplace", "reshape"};
    ggml_set_name(product, "inplace_out_of_place_matmul");
    ggml_set_output(product);
    ggml_set_output(activated); /* Keep the shared storage available for its numerical assertion. */
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, product);
    if (ggml_backend_supports_op(webgpu, normalized) || !ggml_backend_supports_op(cpu, normalized) ||
        !ggml_backend_supports_op(webgpu, product)) {
        fputs("WEBGPU_INPLACE required CPU-only group norm / WebGPU matmul capabilities changed\n", stderr);
        goto done;
    }
    for (size_t i = 1; i < sizeof(aliases) / sizeof(aliases[0]); ++i) {
        if (aliases[i]->view_src != normalized || aliases[i]->view_offs != 0) {
            fputs("WEBGPU_INPLACE graph does not expose the expected common storage owner\n", stderr);
            goto done;
        }
    }
    weight_buffer = ggml_backend_alloc_ctx_tensors(weights, webgpu);
    if (!weight_buffer || !on_device(multiplier, webgpu, 0) || !on_device(offset, webgpu, 0) ||
        !on_device(matrix, webgpu, 0)) goto done;
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(multiplier, scale, 0, sizeof(scale));
    ggml_backend_tensor_set(offset, bias, 0, sizeof(bias));
    ggml_backend_tensor_set(matrix, projection, 0, sizeof(projection));
    ggml_backend_t backends[] = {webgpu, cpu};
    scheduler = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, true);
    if (!scheduler) goto done;
    ggml_backend_sched_set_tensor_backend(scheduler, a, cpu);
    /* This reproduces the inconsistent placement that previously sent a CPU
       owner pointer to WebGPU's tensor_buf() as a native WebGPU buffer. */
    for (size_t i = 1; i < sizeof(aliases) / sizeof(aliases[0]); ++i)
        ggml_backend_sched_set_tensor_backend(scheduler, aliases[i], webgpu);
    ggml_backend_sched_set_tensor_backend(scheduler, product, webgpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) goto done;
    for (size_t i = 0; i < sizeof(aliases) / sizeof(aliases[0]); ++i) {
        if (ggml_backend_sched_get_tensor_backend(scheduler, aliases[i]) != cpu ||
            !on_device(aliases[i], cpu, 1) || aliases[i]->buffer != normalized->buffer ||
            aliases[i]->data != normalized->data) {
            fprintf(stderr, "WEBGPU_INPLACE alias %s has inconsistent CPU placement/storage FAIL\n", names[i]);
            for (size_t j = 0; j < sizeof(aliases) / sizeof(aliases[0]); ++j)
                describe_tensor(scheduler, names[j], aliases[j], cpu);
            goto done;
        }
    }
    /* Placement alone is insufficient: CPU operators need CPU source buffers,
       and the later WebGPU consumer needs a real copied non-host activation. */
    if (!on_device(normalized->src[0], cpu, 1) ||
        !on_device(multiplied->src[0], cpu, 1) || !on_device(multiplied->src[1], cpu, 1) ||
        !on_device(added->src[0], cpu, 1) || !on_device(added->src[1], cpu, 1) ||
        !on_device(activated->src[0], cpu, 1) ||
        ggml_backend_sched_get_tensor_backend(scheduler, product) != webgpu ||
        !on_device(product, webgpu, 0) || !on_device(product->src[0], webgpu, 0) ||
        !on_device(product->src[1], webgpu, 0) || product->view_src ||
        product->src[1]->data == flat->data || !on_device(multiplier, webgpu, 0) ||
        !on_device(offset, webgpu, 0)) {
        fputs("WEBGPU_INPLACE operator sources or final consumer have invalid residency FAIL\n", stderr);
        describe_tensor(scheduler, "groupnorm-input", normalized->src[0], cpu);
        describe_tensor(scheduler, "multiply-input", multiplied->src[0], cpu);
        describe_tensor(scheduler, "multiply-weight-copy", multiplied->src[1], cpu);
        describe_tensor(scheduler, "add-input", added->src[0], cpu);
        describe_tensor(scheduler, "add-weight-copy", added->src[1], cpu);
        describe_tensor(scheduler, "silu-input", activated->src[0], cpu);
        describe_tensor(scheduler, "matmul", product, webgpu);
        describe_tensor(scheduler, "matmul-matrix", product->src[0], webgpu);
        describe_tensor(scheduler, "matmul-activation-copy", product->src[1], webgpu);
        goto done;
    }
    ggml_backend_tensor_set(a, input, 0, sizeof(input));
    const struct inplace_counts before = counts();
    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) goto done;
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_tensor_get(activated, actual_activation, 0, sizeof(actual_activation));
    ggml_backend_tensor_get(product, actual_product, 0, sizeof(actual_product));
    if (compare("cpu-alias-activation", actual_activation, expected_activation, ELEMENTS, 3e-6,
                preallocated, iteration) ||
        compare("webgpu-projection-f16", actual_product, expected_product, CHANNELS * OUTPUTS, 2e-4,
                preallocated, iteration)) goto done;
    const struct inplace_counts after = counts();
    if (after.graphs <= before.graphs || after.submissions <= before.submissions ||
        after.dispatches <= before.dispatches || after.matmuls <= before.matmuls ||
        after.readbacks <= before.readbacks || cosmo_webgpu_native_loader_open_count() != 0) goto done;
    printf("WEBGPU_INPLACE_EXECUTION owner=%s iteration=%u alias_nodes=5 alias_backend=CPU "
           "consumer_backend=WebGPU consumer_host_buffer=0 cpu_fallback=1 "
           "graphs=%" PRIu64 " submissions=%" PRIu64 " dispatches=%" PRIu64
           " matmuls=%" PRIu64 " readbacks=%" PRIu64 " native_loader_opens=0 PASS\n",
           preallocated ? "preallocated" : "scheduler", iteration,
           after.graphs - before.graphs, after.submissions - before.submissions,
           after.dispatches - before.dispatches, after.matmuls - before.matmuls,
           after.readbacks - before.readbacks);
    failed = 0;
done:
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
    if (owner_buffer) ggml_backend_buffer_free(owner_buffer);
    if (weights) ggml_free(weights);
    if (ctx) ggml_free(ctx);
    if (failed) fprintf(stderr, "WEBGPU_INPLACE_CASE owner=%s iteration=%u FAIL\n",
                        preallocated ? "preallocated" : "scheduler", iteration);
    return failed;
}

static int gpu_owner_case(ggml_backend_t webgpu, ggml_backend_t cpu, unsigned iteration) {
    int failed = 1;
    struct ggml_context * ctx = context();
    ggml_backend_buffer_t buffer = NULL;
    ggml_backend_sched_t scheduler = NULL;
    float input[ELEMENTS], projection[SPATIAL * OUTPUTS];
    float actual_activation[ELEMENTS], actual_product[CHANNELS * OUTPUTS];
    double expected_activation[ELEMENTS], expected_product[CHANNELS * OUTPUTS];
    if (!ctx) goto done;
    for (int i = 0; i < ELEMENTS; ++i) {
        input[i] = (float)((i * 11 + 7 + (int)iteration * 3) % 37 - 18) / 8;
        expected_activation[i] = input[i] / (1.0f + expf(-input[i]));
    }
    for (int i = 0; i < SPATIAL * OUTPUTS; ++i)
        projection[i] = (float)((i * 3 + 1 + (int)iteration) % 17 - 8) / 16;
    for (int channel = 0; channel < CHANNELS; ++channel) {
        for (int output = 0; output < OUTPUTS; ++output) {
            double sum = 0;
            for (int k = 0; k < SPATIAL; ++k) {
                const float right = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                    (float)expected_activation[channel * SPATIAL + k]));
                sum += (double)projection[output * SPATIAL + k] * right;
            }
            expected_product[channel * OUTPUTS + output] = sum;
        }
    }
    struct ggml_tensor * owner = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SPATIAL, CHANNELS);
    struct ggml_tensor * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SPATIAL, OUTPUTS);
    ggml_set_name(owner, "inplace_preallocated_webgpu_owner");
    buffer = ggml_backend_alloc_ctx_tensors(ctx, webgpu);
    if (!buffer || !on_device(owner, webgpu, 0) || !on_device(matrix, webgpu, 0)) goto done;
    ggml_backend_tensor_set(owner, input, 0, sizeof(input));
    ggml_backend_tensor_set(matrix, projection, 0, sizeof(projection));
    struct ggml_tensor * activated = ggml_silu_inplace(ctx, owner);
    struct ggml_tensor * product = ggml_mul_mat(ctx, matrix, activated);
    /* New views inherit data immediately, but acquire their buffer only during
       view initialization. That state does not imply raw host memory. */
    if (activated->view_src != owner || activated->data != owner->data || activated->buffer != NULL) {
        fputs("WEBGPU_INPLACE preallocated GPU view no longer exposes the regression precondition\n", stderr);
        goto done;
    }
    ggml_set_output(activated);
    ggml_set_output(product);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(graph, product);
    ggml_backend_t backends[] = {webgpu, cpu};
    scheduler = ggml_backend_sched_new(backends, NULL, 2, GRAPH_SIZE, false, true);
    if (!scheduler) goto done;
    ggml_backend_sched_set_allow_cpu_fallback(scheduler, false);
    ggml_backend_sched_set_tensor_backend(scheduler, activated, webgpu);
    ggml_backend_sched_set_tensor_backend(scheduler, product, webgpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) goto done;
    if (ggml_backend_sched_get_tensor_backend(scheduler, owner) != webgpu ||
        ggml_backend_sched_get_tensor_backend(scheduler, activated) != webgpu ||
        ggml_backend_sched_get_tensor_backend(scheduler, product) != webgpu ||
        !on_device(owner, webgpu, 0) || !on_device(activated, webgpu, 0) ||
        !on_device(product, webgpu, 0) || !on_device(product->src[0], webgpu, 0) ||
        !on_device(product->src[1], webgpu, 0) || product->view_src ||
        owner->data != activated->data || owner->buffer != activated->buffer) {
        fputs("WEBGPU_INPLACE preallocated GPU owner moved to incompatible storage FAIL\n", stderr);
        describe_tensor(scheduler, "preallocated-gpu-owner", owner, webgpu);
        describe_tensor(scheduler, "gpu-silu-view", activated, webgpu);
        describe_tensor(scheduler, "gpu-matmul", product, webgpu);
        describe_tensor(scheduler, "gpu-matmul-matrix", product->src[0], webgpu);
        describe_tensor(scheduler, "gpu-matmul-activation", product->src[1], webgpu);
        goto done;
    }
    const struct inplace_counts before = counts();
    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) goto done;
    ggml_backend_sched_synchronize(scheduler);
    ggml_backend_tensor_get(activated, actual_activation, 0, sizeof(actual_activation));
    ggml_backend_tensor_get(product, actual_product, 0, sizeof(actual_product));
    if (compare("webgpu-owner-silu", actual_activation, expected_activation, ELEMENTS, 3e-6, 1, iteration) ||
        compare("webgpu-owner-projection-f16", actual_product, expected_product,
                CHANNELS * OUTPUTS, 2e-4, 1, iteration)) goto done;
    const struct inplace_counts after = counts();
    if (after.graphs <= before.graphs || after.submissions <= before.submissions ||
        after.dispatches < before.dispatches + 2 || after.matmuls <= before.matmuls ||
        after.readbacks < before.readbacks + 2 || cosmo_webgpu_native_loader_open_count() != 0) goto done;
    printf("WEBGPU_INPLACE_EXECUTION owner=preallocated-webgpu iteration=%u alias_nodes=2 "
           "alias_backend=WebGPU consumer_backend=WebGPU consumer_host_buffer=0 cpu_fallback=0 "
           "graphs=%" PRIu64 " submissions=%" PRIu64 " dispatches=%" PRIu64
           " matmuls=%" PRIu64 " readbacks=%" PRIu64 " native_loader_opens=0 PASS\n",
           iteration, after.graphs - before.graphs, after.submissions - before.submissions,
           after.dispatches - before.dispatches, after.matmuls - before.matmuls,
           after.readbacks - before.readbacks);
    failed = 0;
done:
    if (scheduler) ggml_backend_sched_free(scheduler);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    if (failed) fprintf(stderr, "WEBGPU_INPLACE_CASE owner=preallocated-webgpu iteration=%u FAIL\n", iteration);
    return failed;
}

int cosmo_webgpu_inplace_selftest(void) {
    int failed = 1;
    ggml_backend_t webgpu = NULL, cpu = NULL;
    if (cosmo_webgpu_initialize()) goto done;
    webgpu = ggml_backend_webgpu_init();
    cpu = ggml_backend_cpu_init();
    if (!webgpu || !cpu || strcmp(cosmo_webgpu_provider_name(), "embedded") ||
        cosmo_webgpu_adapter_is_software() != 1 || cosmo_webgpu_native_loader_open_count()) goto done;
    ggml_backend_cpu_set_n_threads(cpu, 2);
    /* Rebuild each graph: splitting rewrites its cross-backend source edges. */
    for (int preallocated = 0; preallocated < 2; ++preallocated)
        for (unsigned iteration = 0; iteration < 2; ++iteration)
            if (graph_case(webgpu, cpu, preallocated, iteration)) goto done;
    for (unsigned iteration = 0; iteration < 2; ++iteration)
        if (gpu_owner_case(webgpu, cpu, iteration)) goto done;
    failed = 0;
done:
    if (cpu) ggml_backend_free(cpu);
    if (webgpu) ggml_backend_free(webgpu);
    puts(failed ? "WEBGPU_INPLACE_SELFTEST FAIL" : "WEBGPU_INPLACE_SELFTEST PASS");
    return failed;
}
