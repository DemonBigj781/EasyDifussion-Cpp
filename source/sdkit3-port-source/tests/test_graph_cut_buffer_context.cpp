#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "ggml-cpu.h"
#include "core/ggml_graph_cut.h"

static size_t alignment_multiplier = 1;
static size_t maximum_reduction = 0;
extern "C" size_t __real_ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t);
extern "C" size_t __real_ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t);
extern "C" size_t __wrap_ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t type) {
    return __real_ggml_backend_buft_get_alignment(type) * alignment_multiplier;
}
extern "C" size_t __wrap_ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t type) {
    return __real_ggml_backend_buft_get_max_size(type) - maximum_reduction;
}
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using namespace sd::ggml_graph_cut;

struct BufferFixture {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free};
    ggml_cgraph* graph;
    explicit BufferFixture(int nodes) {
        require(ctx != nullptr, "context unavailable");
        auto value = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_set_name(value, "input");
        graph = ggml_new_graph_custom(ctx.get(), nodes + 8, false);
        for (int i = 0; i < nodes; ++i) {
            value = ggml_scale(ctx.get(), value, 1.f);
            if ((i + 1) % 64 == 0 || i == nodes - 1) mark_graph_cut(value, "group-" + std::to_string(i), "out");
        }
        ggml_build_forward_expand(graph, value);
    }
};

static void benchmark(ggml_backend_t cpu, int nodes) {
    BufferFixture f(nodes);
    PlanCache cache;
    resolve_plan(cpu, f.graph, &cache, 0, {}, nullptr);
    std::vector<double> timings;
    for (int run = 0; run < 5; ++run) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 500; ++i) {
            auto plan = resolve_plan(cpu, f.graph, &cache, 0, {}, nullptr);
            require(plan.valid, "benchmark cache invalid");
        }
        timings.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 500);
    }
    std::sort(timings.begin(), timings.end());
    std::cout << nodes << " nodes: " << timings[2] << " us/cached resolve (median 5 x 500 calls)\n";
}

int main(int argc, char**) {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        for (size_t budget : {size_t(0), size_t(1), size_t(1024 * 1024)}) {
            BufferFixture f(128);
            PlanCache cache;
            for (int change = 0; change < 3; ++change) {
                auto initial = resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr);
                require(initial.valid, "initial plan invalid");
                cache.graph_cut_plan.segments.front().group_name = "stale-property";
                if (cache.budgeted_graph_cut_plan.available) cache.budgeted_graph_cut_plan.segments.front().group_name = "stale-property";
                require(resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr).segments.front().group_name == "stale-property",
                        "unchanged buffer properties did not reuse cache");
                if (change == 0) alignment_multiplier = 2;
                if (change == 1) maximum_reduction = 1024;
                if (change == 2) { alignment_multiplier = 1; maximum_reduction = 0; }
                auto changed = resolve_plan(cpu.get(), f.graph, &cache, budget, {}, nullptr);
                require(changed.valid && changed.segments.front().group_name != "stale-property", "changed buffer properties reused plan");
            }
        }
        if (argc == 1) {
            for (int nodes : {4, 256, 1024}) benchmark(cpu.get(), nodes);
        }
        std::cout << "PASS injected alignment/limit/restoration invalidation on unchanged CPU backend\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
