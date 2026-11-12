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
        hidden_size = 4; vocab_size = 8; n_token = 77; n_head = 1; n_layer = 1; intermediate_size = 8;
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
    add("shared.weight", 4096, 8);
    add("encoder.block.0.layer.0.SelfAttention.q.weight", 4096, 64);
    add("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", 4096, 8);
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
    auto cpu=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    try {
        require(cpu != nullptr,"CPU unavailable");
        auto lw=std::make_shared<FluxTestWeights>(), gw=std::make_shared<FluxTestWeights>(), tw=std::make_shared<FluxTestWeights>();
        auto l=std::make_shared<TinyFluxClip>(cpu.get(),lw), g=std::make_shared<TinyFluxClip>(cpu.get(),gw);
        auto t=std::make_shared<TinyFluxT5>(cpu.get(),tw);
        SD3CLIPEmbedder embed(cpu.get());
        for(auto tokenizer : {&embed.clip_l_tokenizer,&embed.clip_g_tokenizer}) {
            tokenizer->BOS_TOKEN_ID=0; tokenizer->EOS_TOKEN_ID=1; tokenizer->PAD_TOKEN_ID=0;
        }
        std::vector<std::shared_ptr<FluxTestWeights>> managers={lw,gw,tw};
        for(int present=0;present<8;++present) for(int chunks : {1,2}) for(bool masked : {false,true}) {
            embed.clip_l=(present&1) ? l : nullptr;
            embed.clip_g=(present&2) ? g : nullptr;
            embed.t5=(present&4) ? t : nullptr;
            std::vector<std::pair<std::vector<int>,std::vector<float>>> tokens(3);
            for(int i=0;i<3;++i) if(present&(1<<i)) {
                tokens[i].first.assign(chunks*77,1); tokens[i].second.assign(chunks*77,1.25f);
            }
            auto reset=[&] { for(auto w : managers) w->prepares=w->fail_at=0; };
            auto run=[&] { return embed.get_learned_condition_common(1,tokens,1,masked); };
            reset(); auto baseline=run(); embed.runner_done();
            require(baseline.empty() == (present == 0),"no-encoder contract changed");
            if(!present) {
                embed.set_execution_cancel_callback([] { return true; });
                require(run().empty(), "absent encoders changed under cancellation");
                embed.runner_done();
                embed.set_execution_cancel_callback(nullptr);
                continue;
            }
            require(baseline.c_crossattn.shape()[0] == 4096 && baseline.c_crossattn.shape()[1] == chunks*154,"SD3 chunk dimensions changed");
            require(baseline.c_vector.numel() == ((present&1) ? 4 : 768)+((present&2) ? 4 : 1280),"optional pooled dimensions changed");
            if(masked) for(int64_t i=0;i<baseline.c_crossattn.numel();++i) require(baseline.c_crossattn[i] == 0,"masked outputs not zero");
            if(!(present&4)) for(int chunk=0;chunk<chunks;++chunk)
                for(int64_t i=(chunk*154+77)*4096;i<(chunk+1)*154*4096;++i) require(baseline.c_crossattn[i] == 0,"missing T5 fallback changed");
            if(!(present&1)) for(int i=0;i<768;++i) require(baseline.c_vector[i] == 0,"missing L pooled fallback changed");
            if(!(present&2)) for(int64_t i=(present&1) ? 4 : 768;i<baseline.c_vector.numel();++i) require(baseline.c_vector[i] == 0,"missing G pooled fallback changed");
            for(int component=0;component<3;++component) if(present&(1<<component)) {
                for(int failure=1;failure<=chunks+(component<2 ? 1 : 0);++failure) {
                    reset(); managers[component]->fail_at=failure;
                    require(run().empty(),"active encoder failure returned partial/fallback conditioning");
                    require(managers[component]->prepares == failure,"continued failed encoder");
                    embed.runner_done(); reset();
                    auto recovered=run(); embed.runner_done();
                    equal(recovered.c_crossattn,baseline.c_crossattn); equal(recovered.c_vector,baseline.c_vector);
                }
                for (int cancel_at : {1, chunks + (component < 2 ? 1 : 0)}) {
                    reset();
                    Conditioner& interface = embed;
                    interface.set_execution_cancel_callback([&, component, cancel_at] {
                        return managers[component]->prepares >= cancel_at;
                    });
                    require(run().empty(), "conditioner cancellation returned partial conditioning");
                    require(managers[component]->prepares == cancel_at, "cancelled encoder continued execution");
                    embed.runner_done();
                    interface.set_execution_cancel_callback(nullptr);
                    reset();
                    auto recovered=run(); embed.runner_done();
                    equal(recovered.c_crossattn,baseline.c_crossattn);
                    equal(recovered.c_vector,baseline.c_vector);
                }
            }
        }
        embed.clip_l=l; embed.clip_g=g; embed.t5=t;
        for (int shorter=0;shorter<3;++shorter) {
            std::vector<std::pair<std::vector<int>,std::vector<float>>> tokens(3);
            for (int i=0;i<3;++i) {
                tokens[i].first.assign(i == shorter ? 77 : 154,1);
                tokens[i].second.assign(tokens[i].first.size(),1.25f);
            }
            for(auto w : managers) w->prepares=w->fail_at=0;
            auto automatic=embed.get_learned_condition_common(1,tokens,1);
            embed.runner_done();
            Tokenizer* tokenizers[]={&embed.clip_l_tokenizer,&embed.clip_g_tokenizer,&embed.t5_tokenizer};
            pad_encoder_chunks(*tokenizers[shorter],tokens[shorter].first,tokens[shorter].second,154,77);
            auto explicit_padding=embed.get_learned_condition_common(1,tokens,1);
            embed.runner_done();
            equal(automatic.c_crossattn,explicit_padding.c_crossattn);
            equal(automatic.c_vector,explicit_padding.c_vector);
        }
        std::puts("PASS SD3 eight optional encoder combinations, hidden/pooled/chunk failures, cancellation and recovery");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
