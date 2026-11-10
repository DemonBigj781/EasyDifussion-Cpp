#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "runtime/token-merge.hpp"

static void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "token merging: %s\n", what); std::exit(1); }
}

int main(int argc, char** argv) {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    require(backend != nullptr, "requested backend unavailable");
    for (int tokens : {16, 1088}) for (float ratio : {0.f, .25f, .5f}) {
        auto context = ggml_init({16 * 1024 * 1024, nullptr, true});
        GGMLRunnerContext runner;
        runner.ggml_ctx = context;
        runner.backend = backend;
        std::map<ggml_tensor*, const void*> inputs;
        runner.set_backend_tensor_data = [&](ggml_tensor* tensor, const void* data) {
            ggml_set_input(tensor); inputs[tensor] = data;
        };
        constexpr int channels = 4, frames = 2;
        std::vector<float> original(tokens * channels * frames), metric(original.size());
        for (int f = 0; f < frames; ++f) for (int t = 0; t < tokens; ++t) for (int c = 0; c < channels; ++c) {
            original[(f * tokens + t) * channels + c] = f * 10000.f + t + c;
            metric[(f * tokens + t) * channels + c] = c == (t / 4) % channels ? 1.f : 0.f;
        }
        auto x = ggml_new_tensor_3d(context, GGML_TYPE_F32, channels, tokens, frames);
        auto m = ggml_dup_tensor(context, x);
        runner.bind_backend_tensor_data(x, original.data());
        runner.bind_backend_tensor_data(m, metric.data());
        SpatialTokenMerge merger;
        auto plan = merger.merge(&runner, m, x, ratio);
        if (ratio == 0.f) require(plan.values == x && plan.windows.empty(), "zero ratio must be an identity path");
        else require(plan.values->ne[1] == tokens - int(tokens * ratio), "token reduction count");
        auto restored = merger.unmerge(&runner, plan, plan.values);
        require(ggml_are_same_shape(restored, x), "restored original token/frame layout");
        auto output = ggml_dup(context, restored);
        auto graph = ggml_new_graph_custom(context, 8192, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                std::fprintf(stderr, "unsupported op: %s %lldx%lldx%lld\n", ggml_op_name(node->op),
                             (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2]);
                return 1;
            }
        }
        auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
        require(buffer != nullptr, "graph allocation");
        for (auto& [tensor, data] : inputs) ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph execution");
        std::vector<float> result(original.size()), unchanged(original.size());
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        ggml_backend_tensor_get(x, unchanged.data(), 0, unchanged.size() * sizeof(float));
        require(unchanged == original, "merging must not mutate the residual input");
        for (int f = 0; f < frames; ++f) for (int c = 0; c < channels; ++c) {
            double before = 0, after = 0;
            for (int t = 0; t < tokens; ++t) {
                float value = result[(f * tokens + t) * channels + c];
                require(std::isfinite(value), "finite reconstructed values");
                require(value >= f * 10000.f && value <= f * 10000.f + tokens + channels, "no frame mixing");
                before += original[(f * tokens + t) * channels + c]; after += value;
            }
            const double tolerance = 2 * std::numeric_limits<float>::epsilon() * std::max(1., std::abs(before));
            if (std::abs(before - after) > tolerance) std::fprintf(stderr,
                "tokens=%d ratio=%g frame=%d channel=%d before=%.9f after=%.9f delta=%.9f\n",
                tokens, ratio, f, c, before, after, after-before);
            require(std::abs(before - after) <= tolerance, "weighted merging preserves per-frame sums within FP32 rounding");
        }
        if (ratio == 0.f) require(result == original, "zero ratio is exact");
        ggml_backend_buffer_free(buffer); ggml_free(context);
    }
    std::printf("Token merge/unmerge identity, averaging, layout, frame isolation: PASS (%s)\n", ggml_backend_name(backend));
    ggml_backend_free(backend);
}
