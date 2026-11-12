#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "ggml-cpu.h"
#include "core/ggml_extend.hpp"

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using namespace sd::ggml_graph_cut;

struct ContextFixture {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free};
    ggml_tensor *input, *weight, *a, *b, *c, *output;
    ggml_cgraph* graph;
    explicit ContextFixture(bool inplace = false) {
        require(ctx != nullptr, "context unavailable");
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        ggml_set_name(input, "input"); ggml_set_name(weight, "weight");
        ggml_set_f32(input, 1.f); ggml_set_f32(weight, 2.f);
        a = ggml_add(ctx.get(), input, weight);
        b = inplace ? ggml_scale_bias_inplace(ctx.get(), a, 2.f, 1.f) : ggml_scale_bias(ctx.get(), a, 2.f, 1.f);
        c = ggml_silu(ctx.get(), b);
        output = ggml_scale(ctx.get(), c, 1.f);
        mark_graph_cut(a, "first", "out"); mark_graph_cut(c, "second", "out");
        ggml_set_name(output, "ggml_runner_final_result_tensor");
        graph = ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph, output);
    }
};

static bool weight_is_param(ggml_cgraph* graph, const Plan& plan, ggml_tensor* weight) {
    for (const auto& segment : plan.segments)
        for (const auto& input : segment.input_refs)
            if (input_tensor(graph, input) == weight) return input.type == Segment::INPUT_PARAM;
    throw std::runtime_error("weight missing from plan inputs");
}

class BackendRunner : public GGMLRunner {
public:
    explicit BackendRunner(ggml_backend_t backend) : GGMLRunner(backend) {}
    std::string get_desc() override { return "cache-context-test"; }
    void prime() { graph_cut_plan_cache_.graph_cut_plan.available = true; graph_cut_plan_cache_.budgeted_graph_cut_plan.available = true; }
    bool cached() const { return graph_cut_plan_cache_.graph_cut_plan.available || graph_cut_plan_cache_.budgeted_graph_cut_plan.available; }
};

static void benchmark(ggml_backend_t cpu) {
    ContextFixture f;
    PlanCache cache;
    const std::unordered_set<const ggml_tensor*> params{f.weight};
    resolve_plan(cpu, f.graph, &cache, 1024 * 1024, params, nullptr);
    std::vector<double> timings;
    for (int run = 0; run < 5; ++run) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 5000; ++i) {
            auto plan = resolve_plan(cpu, f.graph, &cache, 1024 * 1024, params, nullptr);
            require(plan.valid, "benchmark plan invalid");
        }
        timings.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 5000);
    }
    std::sort(timings.begin(), timings.end());
    std::cout << "Median small-graph cached resolve: " << timings[2] << " us/call (5 x 5000 calls)\n";
}

int main(int argc, char**) {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    auto cpu2 = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu && cpu2, "CPU backend unavailable");
        benchmark(cpu.get());
        if (argc > 1) return 0;
        for (size_t budget : {size_t(0), size_t(1), size_t(1024 * 1024)}) {
            ContextFixture f;
            PlanCache cache;
            auto initial = resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr);
            require(!weight_is_param(f.graph, initial, f.weight), "initial parameter classification invalid");
            auto changed = resolve_plan(cpu.get(), f.graph, &cache, budget, {f.weight}, nullptr);
            require(weight_is_param(f.graph, changed, f.weight), "changed parameter set reused old classification");
            auto restored = resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr);
            require(!weight_is_param(f.graph, restored, f.weight), "removed parameter membership reused old classification");
            cache.graph_cut_plan.segments.front().group_name = "stale-backend";
            if (cache.budgeted_graph_cut_plan.available) cache.budgeted_graph_cut_plan.segments.front().group_name = "stale-backend";
            const auto same = resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr);
            require(same.segments.front().group_name == "stale-backend", "unchanged context did not reuse cache");
            auto moved = resolve_plan(cpu2.get(), f.graph, &cache, budget, {}, nullptr);
            require(moved.segments.front().group_name != "stale-backend", "backend change reused cached measurements");
        }
        {
            BackendRunner runner(cpu.get());
            runner.prime(); require(runner.set_runtime_backend(cpu.get()) && runner.cached(), "no-op backend update cleared cache");
            require(runner.set_runtime_backend(cpu2.get()) && !runner.cached(), "runner backend update retained cache");
            runner.prime(); runner.set_runtime_backends({cpu.get()}); require(!runner.cached(), "extra backend update retained cache");
            runner.prime(); runner.set_runtime_backends({cpu.get()}); require(runner.cached(), "unchanged extra backends cleared cache");
        }
        for (bool inplace : {false, true}) {
            for (size_t budget : {size_t(0), size_t(1), size_t(1024 * 1024)}) {
                ContextFixture mono(inplace), split(inplace), opposite(!inplace);
                PlanCache cache;
                auto plan = resolve_plan(cpu.get(), split.graph, &cache, budget, {split.weight}, nullptr);
                require(budget > 1 ? plan.segments.size() == 1 : plan.segments.size() > 1,
                        "budget case did not exercise expected segmentation");
                require(!plan_matches_graph(opposite.graph, plan), "in-place/out-of-place layout mismatch ignored");
                const float params[2] = {3.f, -2.f};
                std::memcpy(mono.b->op_params, params, sizeof(params));
                std::memcpy(split.b->op_params, params, sizeof(params));
                require(plan_matches_graph(split.graph, plan), "SCALE bias change rejected");
                require(ggml_graph_compute_with_ctx(mono.ctx.get(), mono.graph, 1) == GGML_STATUS_SUCCESS, "monolithic CPU failed");
                for (const auto& segment : plan.segments) {
                    ggml_context* segment_ctx = nullptr;
                    auto graph = build_segment_graph(split.graph, segment, &segment_ctx);
                    const auto status = ggml_graph_compute_with_ctx(segment_ctx, graph, 1);
                    ggml_free(segment_ctx);
                    require(status == GGML_STATUS_SUCCESS, "segmented CPU graph failed");
                }
                require(std::abs(ggml_get_f32_1d(mono.output, 0) - 7.f / (1.f + std::exp(-7.f))) < 1e-6f, "SCALE bias analytic mismatch");
                for (int i = 0; i < 64; ++i)
                    require(std::abs(ggml_get_f32_1d(mono.output, i) - ggml_get_f32_1d(split.output, i)) < 1e-6f,
                            "segmented numerical mismatch");
            }
        }
        std::cout << "PASS parameter/backend cache context, runner transitions, SCALE bias and segmented in-place/out-of-place CPU parity\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
