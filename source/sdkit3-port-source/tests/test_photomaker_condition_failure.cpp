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
        hidden_size = 4; vocab_size = 50000; n_token = 77; n_head = 1; n_layer = 1; intermediate_size = 8;
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
#include "extensions/generation_extension.h"
#include "model/adapter/pmid.hpp"
struct PhotoMakerSpy {
    int calls=0, cleanups=0;
    float style_strength=0;
    PMVersion version=PM_VERSION_1;
    template<class... Args> PhotoMakerSpy(Args&&...) {}
    PMVersion get_version() const { return version; }
    void get_param_tensors(std::map<std::string,ggml_tensor*>&,const std::string&) {}
    void runner_done() { ++cleanups; }
    sd::Tensor<float> compute(int,const sd::Tensor<float>&,const sd::Tensor<float>& prompt,
                              const sd::Tensor<float>&,std::vector<bool>&) {
        ++calls;
        return prompt;
    }
};
#define PhotoMakerIDEncoder PhotoMakerSpy
#define PhotoMakerExtension TestedPhotoMakerExtension
#define create_photomaker_extension create_test_photomaker_extension
#include "extensions/photomaker_extension.cpp"
#undef create_photomaker_extension
#undef PhotoMakerExtension
#undef PhotoMakerIDEncoder
int main() {
    auto cpu=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    try {
        require(cpu != nullptr,"CPU unavailable");
        auto weights=std::make_shared<FluxTestWeights>();
        auto runner=std::make_shared<TinyFluxClip>(cpu.get(),weights);
        FrozenCLIPEmbedderWithCustomWords embed(cpu.get(),{},{},VERSION_SVD);
        embed.version=VERSION_SD1; embed.text_model=runner;
        TestedPhotoMakerExtension extension;
        extension.enabled=true;
        extension.pmid_model=std::make_shared<PhotoMakerSpy>();
        std::vector<uint8_t> pixels(4*4*3,128);
        sd_image_t image{}; image.width=4; image.height=4; image.channel=3; image.data=pixels.data();
        sd_pm_params_t pm{}; pm.id_images_count=1; pm.id_images=&image; pm.style_strength=25;
        sd_pulid_params_t pulid{};
        ConditionerParams params; params.clip_skip=1; params.width=params.height=64;
        GenerationExtensionConditionContext ctx{&embed,params,pm,pulid,1,20};
        SDCondition ordinary; ordinary.c_crossattn=sd::Tensor<float>::full({4,77},0.25f);
        for(bool cancel : {false,true}) for(PMVersion version : {PM_VERSION_1,PM_VERSION_2}) {
            extension.pmid_model->version=version;
            extension.id_condition=ordinary; extension.start_merge_step=0;
            params.text="a img"; weights->prepares=0; weights->fail_at=cancel ? 0 : 1;
            static_cast<Conditioner&>(embed).set_execution_cancel_callback(cancel ? std::function<bool()>([] { return true; }) : nullptr);
            int calls=extension.pmid_model->calls;
            require(!extension.prepare_condition(ctx),"failed text was accepted");
            require(extension.pmid_model->calls == calls,"ID encoder received failed text conditioning");
            require(extension.id_condition.empty() && extension.start_merge_step == -1,"stale identity condition retained");
            require(params.text == "a img","failed preparation changed prompt");
            require(&extension.before_condition(10,ordinary) == &ordinary,"stale extension overrides ordinary condition");
            extension.runner_done(); embed.runner_done();
            static_cast<Conditioner&>(embed).set_execution_cancel_callback(nullptr); weights->prepares=weights->fail_at=0;
            // PM1 success uses a downstream spy; no learned ID encoder is claimed.
            extension.pmid_model->version=PM_VERSION_1; params.text="a img";
            require(extension.prepare_condition(ctx),"recovery failed");
            require(extension.pmid_model->calls == calls+1,"recovery did not reach ID encoder exactly once");
            require(!extension.id_condition.empty() && extension.start_merge_step == 5,"recovery lost merge behavior");
            require(&extension.before_condition(5,ordinary) == &ordinary,"merge started too early");
            require(&extension.before_condition(6,ordinary) == &extension.id_condition,"merge condition not selected");
            extension.runner_done(); embed.runner_done();
        }
        std::puts("PASS actual PhotoMaker prepare boundary with CPU text failure/cancel, downstream spy and recovery");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
