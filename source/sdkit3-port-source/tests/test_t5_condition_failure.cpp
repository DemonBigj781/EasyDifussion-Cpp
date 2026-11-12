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
        auto weights = std::make_shared<T5TestWeights>();
        auto runner = std::make_shared<TinyConditionT5>(cpu.get(), weights);
        T5CLIPEmbedder embed(cpu.get());
        embed.chunk_len = 4;
        auto absent = embed.get_learned_condition_common(1, {}, 1);
        require(absent.c_crossattn.shape() == std::vector<int64_t>({4096,256}), "absent encoder dimensions changed");
        for (int64_t i=0; i<absent.c_crossattn.numel(); ++i) require(absent.c_crossattn[i] == 0, "absent encoder not zero");
        for (int64_t i=0; i<absent.c_vector.numel(); ++i) require(absent.c_vector[i] == -HUGE_VALF, "absent encoder mask changed");
        embed.t5 = runner;
        for (int chunks : {1,2,3}) for (bool use_mask : {false,true})
        for (bool zero_masked : {false,true}) for (int pad : {0,1,8}) {
            embed.use_mask = use_mask; embed.mask_pad = pad;
            std::vector<int> ids(chunks*4,1);
            std::vector<float> weights_vec(chunks*4,1.25f), mask(chunks*4,0.f);
            for (int i=1; i<chunks*4; i+=2) mask[i] = -HUGE_VALF;
            auto tokens = std::make_tuple(ids,weights_vec,mask);
            weights->prepares = weights->fail_at = 0;
            auto baseline = embed.get_learned_condition_common(1,tokens,1,zero_masked);
            embed.runner_done();
            require(baseline.c_crossattn.shape()[0] == 4 && baseline.c_crossattn.shape()[1] == chunks*4, "chunk shape changed");
            int attended = 0;
            for (int i=0; i<chunks*4; ++i) {
                float expected = mask[i];
                if (std::isinf(expected) && attended < pad) { expected=0; ++attended; }
                require(baseline.c_vector[i] == expected, "padding attention changed");
                if (zero_masked && std::isinf(mask[i]))
                    for (int j=0; j<4; ++j) require(baseline.c_crossattn[i*4+j] == 0, "masked output not zero");
            }
            for (int failure=1; failure<=chunks; ++failure) {
                weights->prepares=0; weights->fail_at=failure;
                require(embed.get_learned_condition_common(1,tokens,1,zero_masked).empty(), "failed chunk returned partial condition");
                require(weights->prepares == failure, "continued beyond failed chunk");
                embed.runner_done();
                weights->prepares=weights->fail_at=0;
                auto recovered=embed.get_learned_condition_common(1,tokens,1,zero_masked);
                embed.runner_done(); equal(recovered.c_crossattn,baseline.c_crossattn);
            }
            embed.set_execution_cancel_callback([] { return true; });
            require(embed.get_learned_condition_common(1,tokens,1,zero_masked).empty(), "cancel returned partial condition");
            embed.runner_done();
            embed.set_execution_cancel_callback(nullptr);
            auto recovered=embed.get_learned_condition_common(1,tokens,1,zero_masked);
            embed.runner_done(); equal(recovered.c_crossattn,baseline.c_crossattn);
            require(recovered.c_vector.shape() == baseline.c_vector.shape(), "mask shape changed after recovery");
            for (int64_t i=0;i<baseline.c_vector.numel();++i) require(recovered.c_vector[i] == baseline.c_vector[i], "mask changed after recovery");
        }
        std::puts("PASS T5-only chunk failure/cancellation/recovery, masking, padding and absent encoder contracts");
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
