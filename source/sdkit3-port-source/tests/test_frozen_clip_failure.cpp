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
    TinyFluxClipModel(int width, int context) {
        blocks.clear(); params.clear();
        hidden_size = width; vocab_size = 8; n_token = context; n_head = 1; n_layer = 1; intermediate_size = 8;
        blocks["embeddings"] = std::make_shared<CLIPEmbeddings>(hidden_size, vocab_size, n_token, true);
        blocks["encoder"] = std::make_shared<CLIPEncoder>(n_layer, hidden_size, n_head, intermediate_size);
        blocks["final_layer_norm"] = std::make_shared<LayerNorm>(hidden_size);
    }
};
struct TinyFluxClip : CLIPTextModelRunner {
    ggml_backend_buffer_t storage;
    TinyFluxClip(ggml_backend_t cpu, const std::shared_ptr<FluxTestWeights>& weights, int width = 4, int context = 4)
        : CLIPTextModelRunner(cpu, {}, "test.clip", OPENAI_CLIP_VIT_L_14, true, true, weights) {
        model = TinyFluxClipModel(width, context);
        free_params_ctx(); alloc_params_ctx();
        model.init(params_ctx);
        storage = initialize_weights(params_ctx, cpu);
    }
    ~TinyFluxClip() { ggml_backend_buffer_free(storage); }
};
static void equal(const sd::Tensor<float>& a, const sd::Tensor<float>& b) {
    require(a.shape() == b.shape(), "recovered shape changed");
    for (int64_t i=0;i<a.numel();++i) require(std::isfinite(a[i]) && a[i] == b[i], "recovered output changed");
}
int main() {
    auto cpu=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    try {
        require(cpu != nullptr,"CPU unavailable");
        auto lw=std::make_shared<FluxTestWeights>(), gw=std::make_shared<FluxTestWeights>();
        auto l=std::make_shared<TinyFluxClip>(cpu.get(),lw);
        auto g=std::make_shared<TinyFluxClip>(cpu.get(),gw,1280,6);
        FrozenCLIPEmbedderWithCustomWords embed(cpu.get(),{},{},VERSION_SVD);
        embed.text_model=l; embed.text_model2=g;
        embed.tokenizer.EOS_TOKEN_ID=embed.tokenizer2.EOS_TOKEN_ID=1;
        for(bool xl : {false,true}) for(int chunks : {1,2,3}) for(bool masked : {false,true}) {
            embed.version=xl ? VERSION_SDXL : VERSION_SD1;
            std::vector<int> lt(chunks*4,1),gt(chunks*6,1);
            std::vector<float> lv(lt.size(),1.25f),gv(gt.size(),0.75f);
            lw->prepares=gw->prepares=lw->fail_at=gw->fail_at=0;
            auto run=[&] { return embed.get_learned_condition_common(1,lt,lv,1,64,96,masked,&gt,&gv); };
            auto baseline=run(); embed.runner_done();
            require(baseline.c_crossattn.shape()[0] == (xl ? 1284 : 4) && baseline.c_crossattn.shape()[1] == chunks*4,"native context alignment changed");
            if(xl) require(baseline.c_vector.numel() == 2816,"SDXL ADM dimensions changed");
            if(masked) for(int64_t i=0;i<baseline.c_crossattn.numel();++i) require(baseline.c_crossattn[i] == 0,"masked output not zero");
            for(int component=0;component<(xl ? 2 : 1);++component) {
                int calls=chunks+(component == 1 ? 1 : 0);
                for(int failure=1;failure<=calls;++failure) {
                    lw->prepares=gw->prepares=0;
                    lw->fail_at=component == 0 ? failure : 0;
                    gw->fail_at=component == 1 ? failure : 0;
                    require(run().empty(),"failed hidden/pooled output returned partial condition");
                    require((component == 0 ? lw->prepares : gw->prepares) == failure,"continued past failure");
                    if(component == 0) require(gw->prepares == 0,"G ran after L failure");
                    embed.runner_done();
                    lw->prepares=gw->prepares=lw->fail_at=gw->fail_at=0;
                    auto recovered=run(); embed.runner_done();
                    equal(recovered.c_crossattn,baseline.c_crossattn); equal(recovered.c_vector,baseline.c_vector);
                }
            }
            for(int component=0;component<(xl ? 2 : 1);++component) {
                auto target=component == 0 ? lw : gw;
                for(int cancel_at : {1,chunks+(component == 1 ? 1 : 0)}) {
                    lw->prepares=gw->prepares=lw->fail_at=gw->fail_at=0;
                    Conditioner& interface=embed;
                    interface.set_execution_cancel_callback([target,cancel_at] { return target->prepares >= cancel_at; });
                    require(run().empty(),"public cancellation returned partial condition");
                    require(target->prepares == cancel_at,"cancelled runner continued");
                    embed.runner_done();
                    interface.set_execution_cancel_callback(nullptr);
                    lw->prepares=gw->prepares=0;
                    auto recovered=run(); embed.runner_done();
                    equal(recovered.c_crossattn,baseline.c_crossattn);
                    equal(recovered.c_vector,baseline.c_vector);
                }
            }
        }
        // The helper must discard earlier hidden AND pooled outputs after a later failed chunk.
        for(bool pooled : {false,true}) {
            std::vector<int> ids(12,1); std::vector<float> weights(12,1.f);
            for(int failure=1;failure<=3+(pooled ? 1 : 0);++failure) {
                lw->prepares=0; lw->fail_at=failure;
                auto result=encode_clip_token_chunks(*l,ids,weights,1,1,0,nullptr,1,pooled,true);
                require(result.hidden_states.empty() && result.pooled.empty(),"helper returned partial output pair");
                l->runner_done();
            }
        }
        FrozenCLIPEmbedderWithCustomWords svd(cpu.get(),{},{},VERSION_SVD);
        ConditionerParams params;
        Conditioner& svd_interface=svd;
        svd_interface.set_execution_cancel_callback([] { return true; });
        require(svd.get_learned_condition(1,params).empty(),"SVD missing text encoder behavior changed");
        svd.runner_done();
        svd_interface.set_execution_cancel_callback(nullptr);
        require(svd.get_learned_condition(1,params).empty(),"SVD reset behavior changed");
        std::puts("PASS FrozenCLIP CPU hidden/pooled failures, native contexts, SDXL ADM, cancellation and recovery");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
