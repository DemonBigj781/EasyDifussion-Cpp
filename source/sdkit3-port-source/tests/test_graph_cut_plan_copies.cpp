#include <cstdio>
#include <cstdlib>
#include <new>
#include <memory>
#include <type_traits>
#include "core/ggml_graph_cut.h"
#include "ggml-cpu.h"

static size_t allocations = 0;
void* operator new(size_t size) {
    ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    using namespace sd::ggml_graph_cut;
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    require(cpu && ctx, "CPU/context unavailable");
    auto graph = ggml_new_graph(ctx.get());
    auto input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
    ggml_set_name(input, "input");
    ggml_set_f32(input, 2.f);
    auto value = input;
    for (int i = 0; i < 8; ++i) {
        value = ggml_scale(ctx.get(), value, 1.f);
        if (i % 2) mark_graph_cut(value, "group-" + std::to_string(i), "output");
    }
    ggml_build_forward_expand(graph, value);
    static_assert(!std::is_reference<decltype(resolve_plan(cpu.get(), graph, nullptr, 0, {}, nullptr))>::value,
                  "the fork must continue returning an independently owned plan");
    for (size_t budget : {size_t(0), size_t(1), size_t(1024 * 1024)}) {
        Plan snapshot;
        {
            PlanCache cache;
            snapshot = resolve_plan(cpu.get(), graph, &cache, budget, {}, nullptr);
            require(snapshot.valid && !snapshot.segments.empty(), "cold plan invalid");
            const auto before_copy = allocations;
            auto copy = snapshot;
            const auto copy_cost = allocations - before_copy;
            const auto before_hit = allocations;
            auto hit = resolve_plan(cpu.get(), graph, &cache, budget, {}, nullptr);
            const auto hit_cost = allocations - before_hit;
            std::printf("budget %zu: returned-plan copy %zu allocations; cached resolve %zu\n", budget, copy_cost, hit_cost);
            require(hit_cost <= copy_cost + 8, "cached resolve made redundant deep plan copies");
            require(hit.layout == snapshot.layout && hit.segments.size() == snapshot.segments.size(), "warm plan differs");
            hit.layout[0] ^= 1;
            hit.segments.front().group_name = "caller-modified";
            hit.cut_markers.clear();
            auto again = resolve_plan(cpu.get(), graph, &cache, budget, {}, nullptr);
            require(again.layout == snapshot.layout && again.cut_markers == snapshot.cut_markers &&
                    again.segments.front().group_name != "caller-modified", "caller modified cached plan");
            cache.graph_cut_plan = {};
            cache.budgeted_graph_cut_plan = {};
            require(plan_matches_graph(graph, copy), "cache reset invalidated independent plan");
        }
        require(plan_matches_graph(graph, snapshot), "cache destruction invalidated returned plan");
        for (const auto& segment : snapshot.segments) {
            ggml_context* segment_ctx = nullptr;
            auto part = build_segment_graph(graph, segment, &segment_ctx);
            auto status = ggml_graph_compute_with_ctx(segment_ctx, part, 1);
            ggml_free(segment_ctx);
            require(status == GGML_STATUS_SUCCESS, "owned plan CPU execution failed");
        }
        require(ggml_get_f32_1d(value, 0) == 2.f, "owned plan CPU result changed");
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) ggml_set_name(ggml_graph_node(graph, i), "");
    {
        PlanCache cache;
        auto plain = resolve_plan(cpu.get(), graph, &cache, 1024 * 1024, {}, nullptr);
        require(plain.valid && !plain.has_cuts && !cache.budgeted_graph_cut_plan.available,
                "uncut graph incorrectly used budgeted cache");
        auto empty = ggml_new_graph(ctx.get());
        auto invalid = resolve_plan(cpu.get(), empty, &cache, 1024 * 1024, {}, nullptr);
        require(invalid.n_nodes == 0 && invalid.segments.empty() && !invalid.has_cuts,
                "empty graph retained previous plan");
    }
    std::puts("PASS bounded plan copies, caller/cache isolation, post-cache-destruction CPU execution");
}
