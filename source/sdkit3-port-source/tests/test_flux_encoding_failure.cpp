#include <cstdio>
#include <stdexcept>
#include "conditioning/conditioner.hpp"
#include "ggml-cpu.h"

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct FluxTestWeights : RunnerWeightManager {
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
struct TinyFluxClipModel : CLIPTextModel {
    TinyFluxClipModel() {
        blocks.clear(); params.clear();
        hidden_size = 4; vocab_size = 8; n_token = 4; n_head = 1; n_layer = 1; intermediate_size = 8;
        blocks["embeddings"] = std::make_shared<CLIPEmbeddings>(hidden_size, vocab_size, n_token, true);
        blocks["encoder"] = std::make_shared<CLIPEncoder>(n_layer, hidden_size, n_head, intermediate_size);
        blocks["final_layer_norm"] = std::make_shared<LayerNorm>(hidden_size);
    }
};
struct TinyFluxClip : CLIPTextModelRunner {
    ggml_backend_buffer_t storage;
    TinyFluxClip(ggml_backend_t cpu, const std::shared_ptr<FluxTestWeights>& weights)
        : CLIPTextModelRunner(cpu, {}, "test.clip", OPENAI_CLIP_VIT_L_14, true, true, weights) {
        model = TinyFluxClipModel();
        free_params_ctx(); alloc_params_ctx();
        model.init(params_ctx);
        storage = initialize_weights(params_ctx, cpu);
    }
    ~TinyFluxClip() { ggml_backend_buffer_free(storage); }
};
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
struct TinyFluxT5 : T5Runner {
    ggml_backend_buffer_t storage;
    TinyFluxT5(ggml_backend_t cpu, const std::shared_ptr<FluxTestWeights>& weights)
        : T5Runner(cpu, t5_fixture(), "test.t5", false, weights) {
        storage = initialize_weights(params_ctx, cpu);
    }
    ~TinyFluxT5() { ggml_backend_buffer_free(storage); }
};
static void equal(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
    require(a.shape() == b.shape(), "successful tensor shape changed");
    for (int64_t i = 0; i < a.numel(); ++i) require(std::isfinite(a[i]) && a[i] == b[i], "recovery changed CPU result");
}
int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        auto clip_weights = std::make_shared<FluxTestWeights>();
        auto t5_weights = std::make_shared<FluxTestWeights>();
        auto clip = std::make_shared<TinyFluxClip>(cpu.get(), clip_weights);
        auto t5 = std::make_shared<TinyFluxT5>(cpu.get(), t5_weights);
        FluxCLIPEmbedder embed(cpu.get());
        embed.chunk_len = 4;
        for (bool have_clip : {false, true}) for (bool have_t5 : {false, true}) {
            embed.clip_l = have_clip ? clip : nullptr;
            embed.t5 = have_t5 ? t5 : nullptr;
            for (int chunks : {1, 2, 3}) for (bool masked : {false, true}) {
                std::vector<std::pair<std::vector<int>, std::vector<float>>> tokens(2);
                if (have_clip) tokens[0] = {{1,2,3,1}, {1.f,1.f,1.f,1.f}};
                if (have_t5) {
                    tokens[1].first.assign(chunks * 4, 1);
                    tokens[1].second.assign(chunks * 4, 1.25f);
                }
                clip_weights->prepares = t5_weights->prepares = 0;
                clip_weights->fail_at = t5_weights->fail_at = 0;
                auto baseline = embed.get_learned_condition_common(1, tokens, 1, masked);
                embed.runner_done();
                require(baseline.empty() == (!have_clip && !have_t5), "absent encoder contract changed");
                if (!have_clip && !have_t5) {
                    embed.set_execution_cancel_callback([] { return true; });
                    require(embed.get_learned_condition_common(1, tokens, 1, masked).empty(), "empty encoder contract changed under cancellation");
                    embed.runner_done();
                    embed.set_execution_cancel_callback(nullptr);
                    continue;
                }
                require(baseline.c_crossattn.shape()[0] == (have_t5 ? 4 : 4096) &&
                        baseline.c_vector.shape()[0] == (have_clip ? 4 : 768), "optional encoder fallback dimensions changed");
                require(baseline.c_crossattn.shape()[1] == (have_t5 ? chunks * 4 : 4), "chunk count changed");
                if (masked || !have_t5)
                    for (int64_t i = 0; i < baseline.c_crossattn.numel(); ++i)
                        require(baseline.c_crossattn[i] == 0.f, "masked/missing T5 not zero");
                if (!have_clip)
                    for (int64_t i = 0; i < baseline.c_vector.numel(); ++i)
                        require(baseline.c_vector[i] == 0.f, "missing CLIP fallback not zero");
                for (int failure = 0; failure <= (have_t5 ? chunks : 0); ++failure) {
                    if (failure == 0 && !have_clip) continue;
                    clip_weights->prepares = t5_weights->prepares = 0;
                    clip_weights->fail_at = failure == 0 ? 1 : 0;
                    t5_weights->fail_at = failure;
                    require(embed.get_learned_condition_common(1, tokens, 1, masked).empty(),
                            "encoder failure returned partial conditioning");
                    require(t5_weights->prepares == failure, "encoding continued after the first failed component/chunk");
                    embed.runner_done();
                    clip_weights->prepares = t5_weights->prepares = 0;
                    clip_weights->fail_at = t5_weights->fail_at = 0;
                    auto recovered = embed.get_learned_condition_common(1, tokens, 1, masked);
                    embed.runner_done();
                    equal(recovered.c_crossattn, baseline.c_crossattn);
                    equal(recovered.c_vector, baseline.c_vector);
                    clip_weights->prepares = t5_weights->prepares = 0;
                    auto target = failure == 0 ? clip_weights : t5_weights;
                    int cancel_at = failure == 0 ? 1 : failure;
                    Conditioner& interface = embed;
                    interface.set_execution_cancel_callback([target, cancel_at] {
                        return target->prepares >= cancel_at;
                    });
                    require(embed.get_learned_condition_common(1, tokens, 1, masked).empty(),
                            "conditioner cancellation returned partial conditioning");
                    require(target->prepares == cancel_at, "encoding continued beyond cancellation");
                    embed.runner_done();
                    interface.set_execution_cancel_callback(nullptr);
                    clip_weights->prepares = t5_weights->prepares = 0;
                    recovered = embed.get_learned_condition_common(1, tokens, 1, masked);
                    embed.runner_done();
                    equal(recovered.c_crossattn, baseline.c_crossattn);
                    equal(recovered.c_vector, baseline.c_vector);
                }
            }
        }
        std::puts("PASS actual Flux CLIP/T5 failure at each chunk, recovery, masking and optional-encoder fallbacks");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
