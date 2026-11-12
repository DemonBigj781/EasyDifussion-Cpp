#include <cstring>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "ggml-cpu.h"
#include "core/ggml_graph_cut.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct LayoutFixture {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free};
    ggml_tensor *input, *weight, *a, *b, *c, *output;
    ggml_cgraph* graph;
    LayoutFixture() {
        require(ctx != nullptr, "context unavailable");
        input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_set_name(input, "input");
        ggml_set_name(weight, "weight");
        ggml_set_f32(input, 1.f);
        ggml_set_f32(weight, 2.f);
        a = ggml_add(ctx.get(), input, weight);
        b = ggml_scale(ctx.get(), a, 2.f);
        c = ggml_silu(ctx.get(), b);
        output = ggml_scale(ctx.get(), c, 1.f);
        sd::ggml_graph_cut::mark_graph_cut(a, "first", "out");
        sd::ggml_graph_cut::mark_graph_cut(c, "second", "out");
        ggml_set_name(output, "ggml_runner_final_result_tensor");
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
    }
    std::unordered_set<const ggml_tensor*> params() const { return {weight}; }
};

int main() {
    using namespace sd::ggml_graph_cut;
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        Plan preserved;
        PlanCache cache;
        {
            LayoutFixture f;
            preserved = resolve_plan(cpu.get(), f.graph, &cache, 1024 * 1024, f.params(), nullptr);
            require(plan_matches_graph(f.graph, preserved), "baseline does not match");
            const auto stride = f.input->nb[1];
            f.input->nb[1] += sizeof(float);
            require(!plan_matches_graph(f.graph, preserved), "changed stride reused plan");
            f.input->nb[1] = stride;
            const auto op = f.c->op;
            f.c->op = GGML_OP_DUP;
            require(!plan_matches_graph(f.graph, preserved), "changed operation reused plan");
            f.c->op = op;
            const auto unary = f.c->op_params[0];
            f.c->op_params[0] = GGML_UNARY_OP_GELU;
            require(!plan_matches_graph(f.graph, preserved), "changed operation parameter reused plan");
            f.c->op_params[0] = unary;
            auto source = f.b->src[0];
            f.b->src[0] = f.input;
            require(!plan_matches_graph(f.graph, preserved), "changed edge reused plan");
            f.b->src[0] = source;
            f.b->view_src = f.weight;
            require(!plan_matches_graph(f.graph, preserved), "changed view source reused plan");
            f.b->view_src = nullptr;
            f.b->view_offs = sizeof(float);
            require(!plan_matches_graph(f.graph, preserved), "changed view offset reused plan");
            f.b->view_offs = 0;
            const auto flags = f.input->flags;
            f.input->flags ^= GGML_TENSOR_FLAG_INPUT;
            require(!plan_matches_graph(f.graph, preserved), "changed tensor flags reused plan");
            f.input->flags = flags;
            f.weight->ne[0] = 2;
            require(!plan_matches_graph(f.graph, preserved), "changed parameter shape reused plan");
            f.weight->ne[0] = 4;
            f.weight->type = GGML_TYPE_F16;
            require(!plan_matches_graph(f.graph, preserved), "changed parameter type reused plan");
            f.weight->type = GGML_TYPE_F32;
            ggml_set_name(f.input, "renamed-input");
            require(!plan_matches_graph(f.graph, preserved), "renamed leaf reused plan");
            ggml_set_name(f.input, "input");
            ggml_set_name(f.b, "ggml_runner_final_result_tensor");
            require(!plan_matches_graph(f.graph, preserved), "changed fork final-output marker reused plan");
            ggml_set_name(f.b, "ordinary");
            require(plan_matches_graph(f.graph, preserved), "restored graph rejected");
        }
        // No tensor pointers from the destroyed graph may be dereferenced.
        for (int repeat = 0; repeat < 8; ++repeat) {
            LayoutFixture f;
            ggml_set_f32(f.input, float(repeat));
            require(plan_matches_graph(f.graph, preserved), "equivalent recreated graph rejected");
            const auto resolved = resolve_plan(cpu.get(), f.graph, &cache, 1024 * 1024, f.params(), nullptr);
            require(plan_matches_graph(f.graph, resolved), "cache reuse after context destruction failed");
            float scale = 3.f;
            std::memcpy(f.b->op_params, &scale, sizeof(scale));
            require(plan_matches_graph(f.graph, preserved), "layout-neutral SCALE update rejected");
            require(ggml_graph_compute_with_ctx(f.ctx.get(), f.graph, 1) == GGML_STATUS_SUCCESS, "CPU graph failed");
            const float value = (float(repeat) + 2.f) * scale;
            const float expected = value / (1.f + std::exp(-value));
            require(std::abs(ggml_get_f32_1d(f.output, 0) - expected) < 1e-5f, "recreated graph numerical mismatch");
        }
        std::cout << "PASS structural cache invalidation, SCALE reuse and eight destroyed/recreated CPU contexts\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
