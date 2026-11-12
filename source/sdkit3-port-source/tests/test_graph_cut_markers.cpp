#include <iostream>
#include <memory>
#include <stdexcept>
#include "ggml-cpu.h"
#include "core/ggml_graph_cut.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace sd::ggml_graph_cut;
    auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(ctx && cpu, "CPU setup failed");
        auto input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        ggml_set_name(input, "input");
        auto a = ggml_scale(ctx.get(), input, 2.f);
        auto b = ggml_scale(ctx.get(), a, 3.f);
        auto c = ggml_scale(ctx.get(), b, 4.f);
        mark_graph_cut(a, "first", "out");
        mark_graph_cut(c, "last", "out");
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, c);
        const std::unordered_set<const ggml_tensor*> params;
        const auto plan = build_plan(cpu.get(), graph, params, nullptr);
        require(plan.valid && plan.has_cuts && plan.segments.size() == 2, "baseline plan invalid");
        require(plan_matches_graph(graph, plan), "unchanged plan rejected");

        mark_graph_cut(a, "renamed", "out");
        require(!plan_matches_graph(graph, plan), "renamed cut incorrectly reused");
        mark_graph_cut(a, "first", "other-output");
        require(!plan_matches_graph(graph, plan), "renamed cut output incorrectly reused");
        mark_graph_cut(a, "first", "out");
        ggml_set_name(a, "ordinary");
        require(!plan_matches_graph(graph, plan), "removed cut incorrectly reused");
        mark_graph_cut(b, "first", "out");
        require(!plan_matches_graph(graph, plan), "moved cut incorrectly reused");
        mark_graph_cut(a, "first", "out");
        require(!plan_matches_graph(graph, plan), "added cut incorrectly reused");
        ggml_set_name(b, "ordinary");
        require(plan_matches_graph(graph, plan), "restored cuts rejected");

        for (size_t budget : {size_t(0), size_t(1), size_t(1024 * 1024)}) {
            PlanCache cache;
            const auto initial = resolve_plan(cpu.get(), graph, &cache, budget, params, nullptr);
            require(initial.valid && plan_matches_graph(graph, initial), "initial cached plan invalid");
            input->ne[0] = 8;
            require(!plan_matches_graph(graph, initial), "budgeted plan lost input shape guard");
            input->ne[0] = 16;
            mark_graph_cut(a, "changed", "out");
            const auto changed = resolve_plan(cpu.get(), graph, &cache, budget, params, nullptr);
            require(changed.valid && plan_matches_graph(graph, changed), "replacement cached plan invalid");
            require(!plan_matches_graph(graph, initial), "original plan accepted after cache replacement");
            require(!changed.segments.empty() && changed.segments.front().group_name.find("changed") != std::string::npos,
                    "cache did not rebuild changed cut group");
            mark_graph_cut(a, "first", "out");
        }

        ggml_set_name(a, "ordinary");
        ggml_set_name(c, "ordinary");
        const auto uncut = build_plan(cpu.get(), graph, params, nullptr);
        require(!uncut.has_cuts && plan_matches_graph(graph, uncut), "uncut plan rejected");
        mark_graph_cut(b, "new", "out");
        require(!plan_matches_graph(graph, uncut), "new cut reused uncut plan");
        ggml_set_name(b, "ggml_runner_cut:incomplete");
        require(plan_matches_graph(graph, uncut), "incomplete marker treated as cut");
        std::cout << "PASS cut rename/output/remove/move/add/restore and zero/tiny/large budget cache invalidation\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
