#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if GGML_MAX_NAME != 160
#error The shared GGML and both application consumers require GGML_MAX_NAME=160
#endif

static struct ggml_context * shared_context(void) {
    const struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    return ggml_init(params);
}

static int shared_compute(struct ggml_context * ctx, struct ggml_tensor * output, int threads) {
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    return ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS;
}

static int shared_values(const char * name, const float * actual,
                          const double * expected, size_t count) {
    double maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        const double error = fabs((double) actual[i] - expected[i]);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            error > 2e-5 * (1 + fabs(expected[i]))) {
            fprintf(stderr, "SHARED_GGML_CHECK %s[%zu]: %.9g != %.9g FAIL\n",
                    name, i, (double) actual[i], expected[i]);
            return 1;
        }
        if (error > maximum) maximum = error;
    }
    printf("SHARED_GGML_CHECK %s values=%zu max_error=%.9g PASS\n", name, count, maximum);
    return 0;
}

static int shared_clamp(int threads) {
    const float original[] = {-3, -1, 0, 0.25f, 0.5f, 2, -0.75f, 4};
    const double expected[] = {-0.5, -0.5, 0, 0.25, 0.5, 1, -0.5, 1};
    struct ggml_context * ctx = shared_context();
    if (!ctx) return 1;
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    memcpy(a->data, original, sizeof(original));
    struct ggml_tensor * result = ggml_clamp(ctx, a, -0.5f, 1);
    int failed = result->view_src != NULL || result->data == a->data;
    if (!failed) failed = shared_compute(ctx, result, threads);
    if (!failed) failed = memcmp(a->data, original, sizeof(original)) != 0;
    if (!failed) failed = shared_values("clamp-out-of-place", result->data, expected, 8);
    struct ggml_tensor * inplace = ggml_clamp_inplace(ctx, a, -0.5f, 1);
    if (!failed) failed = inplace->view_src != a || inplace->data != a->data;
    if (!failed) failed = shared_compute(ctx, inplace, threads);
    if (!failed) failed = shared_values("clamp-in-place", a->data, expected, 8);
    ggml_free(ctx);
    return failed;
}

static int shared_rope(int threads, int mode) {
    enum { WIDTH = 10, HEADS = 2, TOKENS = 3, DIMS = 4, OFFSET = 2, COUNT = WIDTH * HEADS * TOKENS };
    const int32_t positions[TOKENS] = {0, 2, 7};
    float original[COUNT];
    double expected[COUNT], restored[COUNT];
    struct ggml_context * ctx = shared_context();
    if (!ctx) return 1;
    for (int i = 0; i < COUNT; ++i) {
        original[i] = (float) ((i * 7 + 3) % 29 - 14) / 8;
        restored[i] = expected[i] = original[i];
    }
    for (int token = 0; token < TOKENS; ++token) {
        for (int head = 0; head < HEADS; ++head) {
            const int base = (token * HEADS + head) * WIDTH + OFFSET;
            for (int pair = 0; pair < DIMS / 2; ++pair) {
                const double angle = positions[token] * pow(10000.0, -2.0 * pair / DIMS);
                const int first = base + (mode == GGML_ROPE_TYPE_NORMAL ? pair * 2 : pair);
                const int second = first + (mode == GGML_ROPE_TYPE_NORMAL ? 1 : DIMS / 2);
                expected[first] = original[first] * cos(angle) - original[second] * sin(angle);
                expected[second] = original[first] * sin(angle) + original[second] * cos(angle);
            }
        }
    }
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, WIDTH, HEADS, TOKENS);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, TOKENS);
    memcpy(a->data, original, sizeof(original));
    memcpy(pos->data, positions, sizeof(positions));
    struct ggml_tensor * rotated = ggml_rope_set_offset(
        ggml_rope_ext(ctx, a, pos, NULL, DIMS, mode, 0, 10000, 1, 0, 1, 0, 0), OFFSET);
    struct ggml_tensor * inverse = ggml_rope_set_offset(
        ggml_rope_ext_back(ctx, rotated, pos, NULL, DIMS, mode, 0, 10000, 1, 0, 1, 0, 0), OFFSET);
    int failed = shared_compute(ctx, inverse, threads);
    if (!failed) failed = shared_values(mode == GGML_ROPE_TYPE_NORMAL ? "rope-offset-normal" : "rope-offset-neox",
                                       rotated->data, expected, COUNT);
    if (!failed) failed = shared_values("rope-offset-inverse", inverse->data, restored, COUNT);
    if (!failed) failed = memcmp(a->data, original, sizeof(original)) != 0;
    if (!failed) {
        struct ggml_tensor * identity = ggml_rope_set_offset(
            ggml_rope_ext(ctx, a, pos, NULL, 0, mode, 0, 10000, 1, 0, 1, 0, 0), OFFSET);
        failed = shared_compute(ctx, identity, threads);
        if (!failed) failed = shared_values("rope-zero-dims", identity->data, restored, COUNT);
    }
    if (!failed) {
        struct ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, WIDTH, HEADS, TOKENS);
        double expected_gradient[COUNT];
        float * w = weights->data;
        for (int i = 0; i < COUNT; ++i) {
            w[i] = (float) ((i * 5 + 1) % 17 - 8) / 8;
            expected_gradient[i] = w[i];
        }
        for (int token = 0; token < TOKENS; ++token) {
            for (int head = 0; head < HEADS; ++head) {
                const int base = (token * HEADS + head) * WIDTH + OFFSET;
                for (int pair = 0; pair < DIMS / 2; ++pair) {
                    const double angle = positions[token] * pow(10000.0, -2.0 * pair / DIMS);
                    const int first = base + (mode == GGML_ROPE_TYPE_NORMAL ? pair * 2 : pair);
                    const int second = first + (mode == GGML_ROPE_TYPE_NORMAL ? 1 : DIMS / 2);
                    expected_gradient[first] = w[first] * cos(angle) + w[second] * sin(angle);
                    expected_gradient[second] = -w[first] * sin(angle) + w[second] * cos(angle);
                }
            }
        }
        ggml_set_param(a);
        struct ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, rotated, weights));
        ggml_set_loss(loss);
        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, NULL);
        ggml_graph_reset(graph);
        failed = ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS;
        struct ggml_tensor * gradient = ggml_graph_get_grad(graph, a);
        if (!failed) failed = gradient == NULL;
        if (!failed) failed = shared_values("rope-offset-gradient", gradient->data, expected_gradient, COUNT);
    }
    ggml_free(ctx);
    return failed;
}

static int shared_ssm(int threads, int history) {
    enum { STATE = 3, DIM = 2, HEADS = 4, GROUPS = 2, TOKENS = 3, SEQS = 2, STORED = 3,
           STATE_SEQ = STATE * DIM * HEADS, X_COUNT = DIM * HEADS * TOKENS * SEQS,
           BC_COUNT = STATE * GROUPS * TOKENS * SEQS, DT_COUNT = HEADS * TOKENS * SEQS };
    const int32_t ids[SEQS] = {2, 0};
    float initial[STATE_SEQ * STORED], x[X_COUNT], dt[DT_COUNT], decay[HEADS];
    float b[BC_COUNT], c[BC_COUNT];
    double expected[X_COUNT + TOKENS * STATE_SEQ * SEQS];
    struct ggml_context * ctx = shared_context();
    if (!ctx) return 1;
    for (int i = 0; i < STATE_SEQ * STORED; ++i) initial[i] = (float) ((i * 3 + 1) % 19 - 9) / 16;
    for (int i = 0; i < X_COUNT; ++i) x[i] = (float) ((i * 5 + 2) % 17 - 8) / 12;
    for (int i = 0; i < DT_COUNT; ++i) dt[i] = (float) ((i * 7 + 2) % 13 - 6) / 10;
    for (int i = 0; i < HEADS; ++i) decay[i] = -(float) (i + 1) / 8;
    for (int i = 0; i < BC_COUNT; ++i) {
        b[i] = (float) ((i * 3 + 4) % 11 - 5) / 8;
        c[i] = (float) ((i * 7 + 1) % 13 - 6) / 8;
    }
    const int output_count = X_COUNT + history * STATE_SEQ * SEQS;
    for (int i = 0; i < output_count; ++i) expected[i] = NAN;

    // Evaluate the recurrence in double precision, retaining the last K states
    // in newest-first order. Reordered IDs exercise the separate state pool.
    for (int seq = 0; seq < SEQS; ++seq) {
        double state[STATE_SEQ];
        for (int i = 0; i < STATE_SEQ; ++i) state[i] = initial[ids[seq] * STATE_SEQ + i];
        for (int token = 0; token < TOKENS; ++token) {
            for (int head = 0; head < HEADS; ++head) {
                const int group = head / (HEADS / GROUPS);
                const double step = log1p(exp(dt[(seq * TOKENS + token) * HEADS + head]));
                const double attenuation = exp(step * decay[head]);
                for (int dim = 0; dim < DIM; ++dim) {
                    const int output = ((seq * TOKENS + token) * HEADS + head) * DIM + dim;
                    double sum = 0;
                    for (int channel = 0; channel < STATE; ++channel) {
                        const int index = (head * DIM + dim) * STATE + channel;
                        const int bc = ((seq * TOKENS + token) * GROUPS + group) * STATE + channel;
                        state[index] = attenuation * state[index] + b[bc] * step * x[output];
                        sum += c[bc] * state[index];
                    }
                    expected[output] = sum;
                }
            }
            const int slot = TOKENS - token - 1;
            if (slot < history) {
                for (int i = 0; i < STATE_SEQ; ++i) {
                    expected[X_COUNT + (slot * SEQS + seq) * STATE_SEQ + i] = state[i];
                }
            }
        }
    }
    struct ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, STATE, DIM, HEADS, STORED);
    struct ggml_tensor * tx = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DIM, HEADS, TOKENS, SEQS);
    struct ggml_tensor * tdt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEADS, TOKENS, SEQS);
    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, HEADS);
    struct ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, STATE, GROUPS, TOKENS, SEQS);
    struct ggml_tensor * tc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, STATE, GROUPS, TOKENS, SEQS);
    struct ggml_tensor * ti = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, SEQS);
    memcpy(ts->data, initial, sizeof(initial));
    memcpy(tx->data, x, sizeof(x));
    memcpy(tdt->data, dt, sizeof(dt));
    memcpy(ta->data, decay, sizeof(decay));
    memcpy(tb->data, b, sizeof(b));
    memcpy(tc->data, c, sizeof(c));
    memcpy(ti->data, ids, sizeof(ids));
    struct ggml_tensor * result = ggml_ssm_scan(ctx, ts, tx, tdt, ta, tb, tc, ti, history);
    int failed = ggml_nelements(result) != output_count;
    if (!failed) failed = shared_compute(ctx, result, threads);
    if (!failed) failed = shared_values(history == 1 ? "ssm-k1" : "ssm-k3-history", result->data, expected, output_count);
    if (!failed) failed = memcmp(ts->data, initial, sizeof(initial)) != 0;
    ggml_free(ctx);
    return failed;
}

int cosmo_shared_ggml_selftest(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 1;
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    const int wrong_caps = !props.caps.mmap_support;
    ggml_backend_free(backend);
    if (wrong_caps) {
        fprintf(stderr, "SHARED_GGML_CHECK CPU mmap capability FAIL\n");
        return 1;
    }
    const int counts[] = {1, 4};
    for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        const int threads = counts[i];
        printf("SHARED_GGML_CASE threads=%d\n", threads);
        if (shared_clamp(threads) || shared_rope(threads, GGML_ROPE_TYPE_NORMAL) ||
            shared_rope(threads, GGML_ROPE_TYPE_NEOX) || shared_ssm(threads, 1) ||
            shared_ssm(threads, 3)) {
            fprintf(stderr, "SHARED_GGML_SELFTEST FAIL\n");
            return 1;
        }
    }
    puts("SHARED_GGML_SELFTEST PASS");
    return 0;
}

#ifdef GGML_SHARED_TEST_STANDALONE
int main(void) {
    return cosmo_shared_ggml_selftest();
}
#endif
