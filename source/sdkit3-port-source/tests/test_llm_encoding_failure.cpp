#include <cstdio>
#include <stdexcept>
#include "conditioning/conditioner.hpp"
#include "ggml-cpu.h"

static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct FailureTokenizer : BPETokenizer {
    std::vector<std::string> seen;
    FailureTokenizer() { PAD_TOKEN_ID = 0; BOS_TOKEN_ID = 0; EOS_TOKEN_ID = 1; }
    std::vector<int> encode(const std::string& text, on_new_token_cb_t) override { seen.push_back(text); return std::vector<int>(128, 1); }
};
struct FailureWeights : RunnerWeightManager {
    int prepares = 0, fail_at = 0;
    bool assign_compute_backend(const std::vector<ggml_tensor*>&, ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return ++prepares != fail_at; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
};
static String2TensorStorage tiny_weights() {
    String2TensorStorage tensors;
    TensorStorage t;
    t.name = "text_encoders.llm.embed_tokens.weight"; t.type = GGML_TYPE_F32;
    t.n_dims = 2; t.ne[0] = 4; t.ne[1] = 4; tensors[t.name] = t;
    return tensors;
}
struct FailureRunner : LLM::LLMRunner {
    ggml_backend_buffer_t storage;
    FailureRunner(ggml_backend_t cpu, const std::shared_ptr<FailureWeights>& weights)
        : LLMRunner(LLM::LLMArch::QWEN3, cpu, tiny_weights(), "text_encoders.llm", false, weights) {
        storage = ggml_backend_alloc_ctx_tensors(params_ctx, cpu);
        require(storage != nullptr, "parameter buffer unavailable");
        for (auto t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) {
            require(t->type == GGML_TYPE_F32, "fixture needs F32 weights");
            std::vector<float> values(ggml_nelements(t), 0.5f);
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    ~FailureRunner() { ggml_backend_buffer_free(storage); }
};
static String2TensorStorage tiny_t5_weights() {
    String2TensorStorage tensors;
    auto add = [&](const std::string& suffix, int64_t a, int64_t b) {
        TensorStorage t;
        t.name = "test.byt5." + suffix; t.type = GGML_TYPE_F32;
        t.n_dims = 2; t.ne[0] = a; t.ne[1] = b; tensors[t.name] = t;
    };
    add("shared.weight", 4, 512);
    add("encoder.block.0.layer.0.SelfAttention.q.weight", 4, 64);
    add("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", 4, 8);
    return tensors;
}
struct FailureT5 : T5Runner {
    ggml_backend_buffer_t storage;
    FailureT5(ggml_backend_t cpu, const std::shared_ptr<FailureWeights>& weights)
        : T5Runner(cpu, tiny_t5_weights(), "test.byt5", false, weights) {
        storage = ggml_backend_alloc_ctx_tensors(params_ctx, cpu);
        require(storage != nullptr, "T5 parameter buffer unavailable");
        for (auto t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) {
            require(t->type == GGML_TYPE_F32, "fixture needs F32 T5 weights");
            std::vector<float> values(ggml_nelements(t), 0.1f);
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
    }
    ~FailureT5() { ggml_backend_buffer_free(storage); }
};
int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        auto weights = std::make_shared<FailureWeights>();
        LLMEmbedder embed(cpu.get(), tiny_weights(), VERSION_Z_IMAGE);
        embed.tokenizer = std::make_shared<FailureTokenizer>();
        embed.llm = std::make_shared<FailureRunner>(cpu.get(), weights);
        ConditionerParams params;
        params.text = "test";
        for (auto version : {VERSION_Z_IMAGE, VERSION_PID, VERSION_MINIMAX_H3, VERSION_HUNYUAN_VIDEO}) {
            embed.version = version;
            weights->prepares = 0; weights->fail_at = 1;
            require(embed.get_learned_condition(1, params).empty(), "primary failure returned partial conditioning");
            embed.runner_done();
            weights->prepares = 0; weights->fail_at = 0;
            require(!embed.get_learned_condition(1, params).empty(), "recovery after failure did not encode");
            embed.runner_done();
            embed.set_execution_cancel_callback([] { return true; });
            require(embed.get_learned_condition(1, params).empty(), "cancellation returned conditioning");
            embed.runner_done();
            embed.set_execution_cancel_callback({});
        }
        embed.version = VERSION_Z_IMAGE;
        params.text = "(cat:1.5) [dog]";
        auto tokenizer = std::static_pointer_cast<FailureTokenizer>(embed.tokenizer);
        std::vector<sd::Tensor<float>> references;
        params.ref_images = &references;
        for (int reference_count : {1, 2, 3}) {
            references.assign(reference_count, sd::Tensor<float>::ones({2,2,3}));
            for (int fail_at = 0; fail_at <= reference_count + 2; ++fail_at) {
                weights->prepares = 0; weights->fail_at = fail_at;
                tokenizer->seen.clear();
                auto result = embed.get_learned_condition(1, params);
                require(result.empty() == (fail_at != 0), "Omni extra failure returned partial conditioning");
                if (fail_at == 0) require(result.extra_c_crossattns.size() == size_t(reference_count + 1),
                                          "Omni extra prompts lost");
                if (fail_at == 0) {
                    const std::string expected = "<|vision_end|>" + params.text +
                        "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>";
                    require(std::find(tokenizer->seen.begin(), tokenizer->seen.end(), expected) != tokenizer->seen.end(),
                            "Omni weighted syntax was rewritten before tokenization");
                    auto encoded = embed.tokenize(expected, {0, 0});
                    for (float weight : std::get<1>(encoded))
                        require(weight == 1.f, "Omni extra prompt no longer neutral");
                }
                embed.runner_done();
            }
        }
        params.ref_images = nullptr;
        embed.version = VERSION_HUNYUAN_VIDEO;
        params.text = "a sign saying \"hello\"";
        auto t5_weights = std::make_shared<FailureWeights>();
        embed.byt5 = std::make_shared<FailureT5>(cpu.get(), t5_weights);
        for (int fail_at : {1, 0, 1, 0}) {
            weights->prepares = 0; weights->fail_at = 0;
            t5_weights->prepares = 0; t5_weights->fail_at = fail_at;
            auto result = embed.get_learned_condition(1, params);
            require(result.empty() == (fail_at != 0), "ByT5 failure returned partial conditioning");
            if (fail_at == 0) require(result.extra_c_crossattns.size() == 1, "ByT5 successful output lost");
            embed.runner_done();
        }
        embed.set_execution_cancel_callback([] { return true; });
        embed.llm->set_execution_cancel_callback({});
        weights->fail_at = t5_weights->fail_at = 0;
        require(embed.get_learned_condition(1, params).empty(), "ByT5 cancellation callback not propagated");
        embed.runner_done();
        embed.set_execution_cancel_callback({});
        std::puts("PASS Omni extra-prompt and ByT5 failure/recovery through real CPU encoders");
        std::puts("PASS primary failure, cancellation and repeated CPU recovery for ZImage/PixelDiT/MiniMax/Hunyuan");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
