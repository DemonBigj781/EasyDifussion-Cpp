#include <iostream>
#include <memory>
#include <stdexcept>

#include "ggml-cpu.h"
#include "core/ggml_extend.hpp"

static bool fail_reserve = false;
static bool fail_cache = false;
static ggml_context* tracked_context = nullptr;
static ggml_backend_buffer_t tracked_buffer = nullptr;
static int context_frees = 0;
static int buffer_frees = 0;

extern "C" bool __real_ggml_gallocr_reserve(ggml_gallocr_t, ggml_cgraph*);
extern "C" bool __wrap_ggml_gallocr_reserve(ggml_gallocr_t allocator, ggml_cgraph* graph) {
    return fail_reserve ? false : __real_ggml_gallocr_reserve(allocator, graph);
}
extern "C" ggml_backend_buffer_t __real_ggml_backend_alloc_ctx_tensors(ggml_context*, ggml_backend_t);
extern "C" ggml_backend_buffer_t __wrap_ggml_backend_alloc_ctx_tensors(ggml_context* ctx, ggml_backend_t backend) {
    return fail_cache ? nullptr : __real_ggml_backend_alloc_ctx_tensors(ctx, backend);
}
extern "C" void __real_ggml_free(ggml_context*);
extern "C" void __wrap_ggml_free(ggml_context* ctx) {
    if (ctx != nullptr && ctx == tracked_context) ++context_frees;
    __real_ggml_free(ctx);
}
extern "C" void __real_ggml_backend_buffer_free(ggml_backend_buffer_t);
extern "C" void __wrap_ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer != nullptr && buffer == tracked_buffer) ++buffer_frees;
    __real_ggml_backend_buffer_free(buffer);
}

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

class StatusRunner : public GGMLRunner {
    sd::Tensor<float> values{{4}, {1.f, 2.f, 3.f, 4.f}};
    sd::Tensor<int32_t> integers{{4}, {1, 2, 3, 4}};
public:
    explicit StatusRunner(ggml_backend_t backend) : GGMLRunner(backend) {}
    std::string get_desc() override { return "status-test"; }
    std::optional<sd::Tensor<float>> run(bool with_cache = false, bool bad_output = false) {
        return compute<float>([&]() {
            auto graph = new_graph_custom(64);
            auto result = bad_output ? ggml_dup(compute_ctx, make_input(integers))
                                     : ggml_scale(compute_ctx, make_input(values), 2.f);
            if (with_cache) cache("saved-output", result);
            ggml_build_forward_expand(graph, result);
            return graph;
        }, 1);
    }
    std::optional<sd::Tensor<float>> run_cached() {
        return compute<float>([&]() {
            auto graph = new_graph_custom(64);
            auto saved = get_cache_tensor_by_name("saved-output");
            require(saved != nullptr, "cached input unavailable");
            auto result = ggml_scale(compute_ctx, to_backend(saved), 3.f);
            cache("saved-output", result);
            ggml_build_forward_expand(graph, result);
            return graph;
        }, 1);
    }
    std::vector<float> cache_values() {
        auto saved = get_cache_tensor_by_name("saved-output");
        if (!saved) return {};
        std::vector<float> values(ggml_nelements(saved));
        ggml_backend_tensor_get(saved, values.data(), 0, ggml_nbytes(saved));
        return values;
    }
    ggml_context* cache_context() const { return cache_ctx; }
    ggml_backend_buffer_t cached_buffer() const { return cache_buffer; }
};

int main() {
    auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    auto backend2 = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(backend != nullptr, "CPU backend unavailable");
        StatusRunner runner(backend.get());
        auto success = runner.run();
        require(success && success->values() == std::vector<float>({2, 4, 6, 8}) && runner.last_compute_status() == GGML_STATUS_SUCCESS,
                "normal CPU result/status failed");
        fail_reserve = true;
        require(!runner.run() && runner.last_compute_status() == GGML_STATUS_ALLOC_FAILED, "reserve failure not classified as allocation");
        fail_reserve = false;
        runner.runner_done();
        require(runner.last_compute_status() == GGML_STATUS_ALLOC_FAILED, "cleanup erased failure status");
        require(runner.run() && runner.last_compute_status() == GGML_STATUS_SUCCESS, "allocation failure leaked into next run");
        require(!runner.run(false, true) && runner.last_compute_status() == GGML_STATUS_FAILED, "readback failure reused successful status");
        runner.set_execution_cancel_callback([] { return true; });
        require(!runner.run() && runner.last_compute_status() == GGML_STATUS_ABORTED, "cancellation misclassified");
        int cancellation_checks = 0;
        runner.set_execution_cancel_callback([&] { return ++cancellation_checks > 1; });
        require(!runner.run() && runner.last_compute_status() == GGML_STATUS_ABORTED, "dispatch-time cancellation misclassified");
        runner.set_execution_cancel_callback({});
        require(runner.run(true) && runner.last_compute_status() == GGML_STATUS_SUCCESS, "initial cached result failed");
        tracked_context = runner.cache_context();
        tracked_buffer = runner.cached_buffer();
        require(tracked_context != nullptr && tracked_buffer != nullptr, "no existing cache to replace");
        fail_cache = true;
        require(!runner.run(true) && runner.last_compute_status() == GGML_STATUS_ALLOC_FAILED, "cache allocation did not return recoverable failure");
        fail_cache = false;
        require(context_frees == 1 && buffer_frees == 1, "old cache ownership leaked or freed twice");
        require(runner.cache_context() == nullptr && runner.cached_buffer() == nullptr, "failed replacement retained new cache ownership");
        tracked_context = nullptr;
        tracked_buffer = nullptr;
        require(runner.run(true) && runner.last_compute_status() == GGML_STATUS_SUCCESS, "cache did not recover after failure");
        std::vector<float> expected{2, 4, 6, 8};
        for (int cycle = 0; cycle < 4; ++cycle) {
            tracked_context = runner.cache_context(); tracked_buffer = runner.cached_buffer();
            context_frees = buffer_frees = 0;
            for (auto& value : expected) value *= 3.f;
            auto result = runner.run_cached();
            require(result && result->values() == expected && runner.cache_values() == expected,
                    "cached input copy/rebind produced incorrect values");
            require(context_frees == 1 && buffer_frees == 1, "cached replacement lifetime incorrect");
            tracked_context = nullptr; tracked_buffer = nullptr;
        }
        require(backend2 != nullptr, "second CPU backend unavailable");
        tracked_context = runner.cache_context(); tracked_buffer = runner.cached_buffer();
        context_frees = buffer_frees = 0;
        require(runner.set_runtime_backend(backend2.get()), "CPU backend switch failed");
        require(context_frees == 1 && buffer_frees == 1 && runner.cache_values().empty(), "backend switch did not release cache");
        tracked_context = nullptr; tracked_buffer = nullptr;
        require(runner.run(true) && runner.cache_values() == std::vector<float>({2, 4, 6, 8}), "new backend cache did not recover");
        tracked_context = runner.cache_context(); tracked_buffer = runner.cached_buffer();
        context_frees = buffer_frees = 0;
        runner.free_cache_ctx_and_buffer(); runner.free_cache_ctx_and_buffer();
        require(context_frees == 1 && buffer_frees == 1 && runner.cache_values().empty(), "explicit cache cleanup was not idempotent");
        tracked_context = nullptr; tracked_buffer = nullptr;
        require(runner.set_runtime_backend(backend.get()), "return to original backend failed");
        std::cout << "PASS CPU runner success/allocation/readback/cancellation status, cleanup retention, cache replacement failure ownership, repeated copy/rebind, backend switching and idempotent cleanup\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
