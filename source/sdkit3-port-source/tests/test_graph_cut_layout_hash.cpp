#include <cstdio>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <chrono>
#include "core/ggml_graph_cut.cpp"

static size_t allocations = 0;
void* operator new(size_t size) {
    ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

    static std::vector<uint64_t> legacy_layout(ggml_cgraph* graph,
                                               const std::unordered_set<const ggml_tensor*>* params = nullptr,
                                               std::vector<uint8_t>* parameter_roles = nullptr) {
        std::vector<const ggml_tensor*> tensors;
        std::unordered_map<const ggml_tensor*, size_t> indices;
        auto add = [&](const ggml_tensor* tensor) {
            if (tensor != nullptr && indices.emplace(tensor, tensors.size() + 1).second) {
                tensors.push_back(tensor);
            }
        };
        for (int i = 0; i < graph->n_leafs; ++i) {
            add(graph->leafs[i]);
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            add(graph->nodes[i]);
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            add(tensors[i]->view_src);
            for (auto source : tensors[i]->src) {
                add(source);
            }
        }
        if (parameter_roles != nullptr) {
            GGML_ASSERT(params != nullptr);
            parameter_roles->clear();
            parameter_roles->reserve(tensors.size());
            for (const auto* tensor : tensors) {
                parameter_roles->push_back(sd::ggml_graph_cut::is_params_tensor(*params, tensor));
            }
        }
        std::vector<uint64_t> signature;
        signature.reserve(tensors.size() * 24);
        signature.push_back(graph->n_nodes);
        signature.push_back(graph->n_leafs);
        for (int i = 0; i < graph->n_leafs; ++i) {
            signature.push_back(indices.at(graph->leafs[i]));
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            signature.push_back(indices.at(graph->nodes[i]));
        }
        for (auto tensor : tensors) {
            signature.push_back(tensor->op);
            signature.push_back(tensor->type);
            signature.push_back(tensor->flags);
            signature.push_back(tensor->view_offs);
            signature.push_back(std::strcmp(tensor->name, "ggml_runner_final_result_tensor") == 0);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                signature.push_back(tensor->ne[d]);
                signature.push_back(tensor->nb[d]);
            }
            signature.push_back(tensor->view_src == nullptr ? 0 : indices.at(tensor->view_src));
            for (auto source : tensor->src) {
                signature.push_back(source == nullptr ? 0 : indices.at(source));
            }
            // SCALE coefficients change values, not layout or allocation size.
            if (tensor->op != GGML_OP_SCALE) {
                for (int value : tensor->op_params) {
                    signature.push_back(static_cast<uint32_t>(value));
                }
            }
        }
        return signature;
    }

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static void check_layout(size_t count, int roots, bool scale) {
    std::vector<ggml_tensor> tensors(count);
    std::vector<ggml_tensor*> nodes, leaves;
    std::unordered_set<const ggml_tensor*> params;
    for (size_t i = 0; i < count; ++i) {
        auto& t = tensors[i];
        t.type = i % 2 ? GGML_TYPE_F32 : GGML_TYPE_F16;
        t.op = scale ? GGML_OP_SCALE : GGML_OP_ADD;
        t.flags = i % 3;
        t.view_offs = i * 4;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) { t.ne[d] = i + d + 1; t.nb[d] = (i + 1) * (d + 4); }
        for (int j = 0; j < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++j) t.op_params[j] = int(i + j);
        if (i) { t.src[0] = &tensors[i - 1]; t.src[1] = &tensors[i / 2]; }
        if (i && i % 3 == 0) t.view_src = &tensors[i - 1];
        if (i % 5 == 0) params.insert(&t);
        if (i % 7 == 0) ggml_set_name(&t, "ggml_runner_final_result_tensor");
        if (roots == 0 || (roots == 1 && i % 2 == 0)) nodes.push_back(&t);
        if (roots == 0 && i % 3 == 0) leaves.push_back(&t);
    }
    if (count && roots >= 2) {
        nodes.push_back(&tensors.back());
        if (roots == 3) {
            // Closure traversal must terminate even when references revisit the root.
            tensors.front().src[0] = &tensors.back();
            nodes.push_back(&tensors.back());
            leaves.push_back(&tensors.back());
        }
    }
    ggml_cgraph graph{};
    graph.n_nodes = nodes.size(); graph.nodes = nodes.data();
    graph.n_leafs = leaves.size(); graph.leafs = leaves.data();
    std::vector<uint8_t> expected_roles, roles;
    const size_t before = allocations;
    auto expected = legacy_layout(&graph, &params, &expected_roles);
    const size_t old_allocations = allocations - before;
    const size_t after = allocations;
    auto observed = sd::ggml_graph_cut::graph_layout(&graph, &params, &roles);
    const size_t new_allocations = allocations - after;
    require(expected == observed, "layout serialization changed");
    require(expected_roles == roles, "parameter classification changed");
    require(sd::ggml_graph_cut::graph_layout(&graph) == expected, "optional roles changed layout");
    if (count >= 256) require(new_allocations < old_allocations / 4, "layout allocation reduction missing");
    if (count == 1024 && roots == 0 && !scale)
        std::printf("1024 tensor layout: %zu -> %zu allocations; signature identical\n", old_allocations, new_allocations);
}

static void check_collisions() {
    using sd::ggml_graph_cut::GraphLayoutTensors;
    std::vector<ggml_tensor> pool(8192);
    std::vector<std::vector<ggml_tensor*>> buckets(64);
    for (auto& tensor : pool) {
        size_t hash = reinterpret_cast<uintptr_t>(&tensor) >> 4;
        hash ^= hash >> 16;
        buckets[hash & 63].push_back(&tensor);
    }
    auto& bucket = *std::max_element(buckets.begin(), buckets.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    require(bucket.size() >= 32, "collision fixture insufficient");
    GraphLayoutTensors table(32);
    table.add(nullptr);
    for (size_t i = 0; i < 32; ++i) table.add(bucket[i]);
    for (size_t i = 32; i-- > 0;) {
        require(table.index(bucket[i]) == i + 1, "collision lookup lost index");
        table.add(bucket[i]);
    }
    require(table.tensors.size() == 32 && table.index(nullptr) == 0, "duplicate/null handling changed");
    for (auto& tensor : pool) table.add(&tensor);
    for (size_t i = 0; i < table.tensors.size(); ++i)
        require(table.index(table.tensors[i]) == i + 1, "rehash lost stable index");
}

int main() {
    for (size_t count : {size_t(0), size_t(1), size_t(4), size_t(256), size_t(1024)})
        for (int roots = 0; roots < 4; ++roots)
            for (bool scale : {false, true}) check_layout(count, roots, scale);
    check_collisions();
    std::puts("PASS 40 exact legacy-layout/role comparisons, collision chains, closure growth and stable indices");
}
