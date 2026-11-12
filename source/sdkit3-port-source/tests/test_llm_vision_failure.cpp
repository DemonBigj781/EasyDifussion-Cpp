#include <cstdio>
#include <stdexcept>
#include "conditioning/conditioner.hpp"
#include "ggml-cpu.h"

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct VisionTokenizer : BPETokenizer {
    VisionTokenizer() { PAD_TOKEN_ID = 0; BOS_TOKEN_ID = 0; EOS_TOKEN_ID = 1; }
    std::vector<int> encode(const std::string& text, on_new_token_cb_t) override {
        return std::vector<int>(128 + text.size(), 1);
    }
};
struct VisionWeights : RunnerWeightManager {
    int prepares = 0, fail_at = 0;
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return ++prepares != fail_at; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
};
static String2TensorStorage vision_fixture(bool deepstack) {
    String2TensorStorage out;
    auto add = [&](const std::string& suffix, std::initializer_list<int64_t> shape) {
        TensorStorage t; t.name = "text_encoders.llm." + suffix; t.type = GGML_TYPE_F32;
        t.n_dims = shape.size(); int i = 0; for (auto d : shape) t.ne[i++] = d; out[t.name] = t;
    };
    const int patch = deepstack ? 14 : 1;
    add("embed_tokens.weight", {4,4});
    add("visual.patch_embed.proj.weight", {patch,patch,2,3,64});
    add("visual.patch_embed.proj.bias", {64});
    add("visual.blocks.0.mlp.gate_proj.weight", {64,64});
    if (deepstack) {
        add("visual.blocks.23.norm1.weight", {64});
        add("visual.pos_embed.weight", {64,16});
    }
    add("visual.merger.mlp.2.weight", {256,4});
    return out;
}
struct VisionRunner : LLM::LLMRunner {
    ggml_backend_buffer_t storage;
    VisionRunner(ggml_backend_t cpu, const std::shared_ptr<VisionWeights>& weights, bool deepstack)
        : LLMRunner(deepstack ? LLM::LLMArch::QWEN3_VL : LLM::LLMArch::QWEN2_5_VL,
                    cpu, vision_fixture(deepstack), "text_encoders.llm", true, weights) {
        storage = ggml_backend_alloc_ctx_tensors(params_ctx, cpu);
        require(storage != nullptr, "vision parameter allocation failed");
        for (auto t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) {
            require(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16, "unsupported fixture type");
            if (t->type == GGML_TYPE_F32) {
                std::vector<float> values(ggml_nelements(t), 0.01f);
                ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<ggml_fp16_t> values(ggml_nelements(t), ggml_fp32_to_fp16(0.01f));
                ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            }
        }
    }
    ~VisionRunner() { ggml_backend_buffer_free(storage); }
};
int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        auto weights = std::make_shared<VisionWeights>();
        LLMEmbedder embed(cpu.get(), vision_fixture(false), VERSION_QWEN_IMAGE, "", true);
        embed.tokenizer = std::make_shared<VisionTokenizer>();
        embed.llm = std::make_shared<VisionRunner>(cpu.get(), weights, false);
        ConditionerParams params;
        params.text = "test";
        params.ref_image_params.vlm_min_size = 4;
        params.ref_image_params.vlm_max_size = 4;
        std::vector<sd::Tensor<float>> refs(2, sd::Tensor<float>::ones({4,4,3,1}));
        params.ref_images = &refs;
        auto exercise = [&] {
            for (int mode : {1, 0, 2, 0, 3, 0}) {
                weights->prepares = 0; weights->fail_at = mode == 2 ? 1 : (mode == 3 ? 2 : 0);
                embed.set_execution_cancel_callback(mode == 1 ? std::function<bool()>([] { return true; }) : std::function<bool()>());
                auto result = embed.get_learned_condition(1, params);
                require(result.empty() == (mode != 0), "vision cancellation/failure/recovery mismatch");
                embed.runner_done();
            }
        };
        for (auto version : {VERSION_QWEN_IMAGE, VERSION_MAGE_FLOW, VERSION_LINGBOT_VIDEO,
                             VERSION_BOOGU_IMAGE, VERSION_KREA2, VERSION_LONGCAT}) {
            embed.version = version;
            exercise();
        }
        embed.version = VERSION_MINIMAX_H3;
        embed.llm = std::make_shared<VisionRunner>(cpu.get(), weights, true);
        exercise();
        std::vector<MiniMaxH3PresentationItem> presentations;
        params.minimax_h3_references = &presentations;
        for (auto kind : {MiniMaxH3PresentationKind::IMAGE, MiniMaxH3PresentationKind::VIDEO}) {
            presentations.clear();
            MiniMaxH3PresentationItem item;
            item.kind = kind;
            item.frames.assign(kind == MiniMaxH3PresentationKind::IMAGE ? 1 : 4, refs[0]);
            presentations.push_back(item);
            exercise();
        }
        for (bool video : {false, true}) {
            for (int mode : {1, 0, 2, 0}) {
                weights->prepares = 0; weights->fail_at = mode == 2 ? 1 : 0;
                embed.set_execution_cancel_callback(mode == 1 ? std::function<bool()>([] { return true; }) : std::function<bool()>());
                auto outputs = video
                    ? embed.llm->encode_video_block_outputs(1, sd::Tensor<float>::ones({28,28,2,3,1}), false, true, true)
                    : embed.llm->encode_image_outputs(1, sd::Tensor<float>::ones({28,28,3,1}), false, true, true);
                require(outputs.empty() == (mode != 0), "singleton deepstack failure/recovery mismatch");
                if (mode == 0) {
                    require(outputs.size() == 4, "deepstack output count changed");
                    for (const auto& output : outputs)
                        require(output.dim() == 2 && output.shape()[0] == 4 && output.shape()[1] == 1,
                                "singleton vision token dimension was lost");
                }
                embed.runner_done();
            }
        }
        std::puts("PASS all five image-consumer branches, MiniMax image/ref/video: CPU cancellation, fault and recovery");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
