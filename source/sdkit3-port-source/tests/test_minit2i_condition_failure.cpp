#include <cstdio>
#include <stdexcept>
#include "conditioning/conditioner.hpp"
#include "ggml-cpu.h"

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct T5TestWeights : RunnerWeightManager {
    int prepares = 0, fail_at = 0;
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return ++prepares != fail_at; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
};
static ggml_backend_buffer_t initialize_weights(ggml_context* ctx, ggml_backend_t cpu) {
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    require(buffer != nullptr, "fixture parameter allocation failed");
    for (auto t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16, "unsupported fixture type");
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> values(ggml_nelements(t), 0.1f);
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> values(ggml_nelements(t), ggml_fp32_to_fp16(0.1f));
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    return buffer;
}
static String2TensorStorage t5_fixture() {
    String2TensorStorage out;
    auto add = [&](const std::string& suffix, int64_t a, int64_t b) {
        TensorStorage t; t.name = "test.t5." + suffix; t.type = GGML_TYPE_F32;
        t.n_dims = 2; t.ne[0] = a; t.ne[1] = b; out[t.name] = t;
    };
    add("shared.weight", 4, 8);
    add("encoder.block.0.layer.0.SelfAttention.q.weight", 4, 64);
    add("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", 4, 8);
    return out;
}
struct TinyConditionT5 : T5Runner {
    ggml_backend_buffer_t storage;
    TinyConditionT5(ggml_backend_t cpu, const std::shared_ptr<T5TestWeights>& weights)
        : T5Runner(cpu, t5_fixture(), "test.t5", false, weights) {
        storage = initialize_weights(params_ctx, cpu);
    }
    ~TinyConditionT5() { ggml_backend_buffer_free(storage); }
};
static void equal(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
    require(a.shape() == b.shape(), "successful tensor shape changed");
    for (int64_t i = 0; i < a.numel(); ++i) require(std::isfinite(a[i]) && a[i] == b[i], "recovery changed CPU result");
}
int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        auto weights=std::make_shared<T5TestWeights>();
        auto runner=std::make_shared<TinyConditionT5>(cpu.get(),weights);
        MiniT2IConditioner embed(cpu.get());
        ConditionerParams params;
        embed.prompt_length=4;
        auto absent=embed.get_learned_condition(1,params);
        require(absent.c_crossattn.shape() == std::vector<int64_t>({1024,4}), "absent dimensions changed");
        for (int64_t i=0;i<absent.c_crossattn.numel();++i) require(absent.c_crossattn[i] == 0, "absent hidden not zero");
        for (int64_t i=0;i<absent.c_vector.numel();++i) require(absent.c_vector[i] == 0, "absent mask not zero");
        embed.t5=runner;
        for (int count : {1,2,4,6}) {
            params.text.clear();
            for(int i=0;i<count;++i) params.text += "</s>";
            auto expected_tokens=embed.tokenizer.encode(params.text);
            weights->prepares=weights->fail_at=0;
            auto baseline=embed.get_learned_condition(1,params);
            embed.runner_done();
            require(baseline.c_crossattn.shape()[0] == 4 && baseline.c_crossattn.shape()[1] == 4, "fixed prompt length changed");
            for (int i=0;i<4;++i) require(baseline.c_vector[i] == (i < expected_tokens.size() ? 1.f : 0.f), "mask padding or truncation changed");
            for (bool cancel : {false,true}) {
                weights->prepares=0; weights->fail_at=cancel ? 0 : 1;
                static_cast<Conditioner&>(embed).set_execution_cancel_callback(cancel ? std::function<bool()>([] { return true; }) : nullptr);
                require(embed.get_learned_condition(1,params).empty(), "failure returned partial MiniT2I conditioning");
                embed.runner_done();
                weights->prepares=weights->fail_at=0;
                static_cast<Conditioner&>(embed).set_execution_cancel_callback(nullptr);
                auto recovered=embed.get_learned_condition(1,params);
                embed.runner_done();
                equal(recovered.c_crossattn,baseline.c_crossattn);
                equal(recovered.c_vector,baseline.c_vector);
            }
        }
        std::puts("PASS MiniT2I CPU failure, runner cancellation, recovery and fixed prompt mask contracts");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
