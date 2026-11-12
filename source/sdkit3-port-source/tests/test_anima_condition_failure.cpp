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
int main() {
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(), ggml_backend_free);
    try {
        require(cpu != nullptr, "CPU unavailable");
        auto weights=std::make_shared<FailureWeights>();
        AnimaConditioner embed(cpu.get(), tiny_weights());
        auto tokenizer=std::make_shared<FailureTokenizer>();
        embed.qwen_tokenizer=tokenizer;
        embed.llm=std::make_shared<FailureRunner>(cpu.get(),weights);
        ConditionerParams params;
        for (const auto& text : {"</s>", "(</s>:1.5)", "[</s>]"}) {
            params.text=text;
            weights->prepares=weights->fail_at=0;
            auto expected=embed.tokenize(params.text);
            for(float w : std::get<1>(expected)) require(w == 1.f, "Anima Qwen weighting contract changed");
            auto baseline=embed.get_learned_condition(1,params);
            embed.runner_done();
            require(!baseline.empty(), "baseline missing");
            require(baseline.c_t5_ids.numel() == std::get<2>(expected).size(), "T5 IDs shape changed");
            for(int64_t i=0;i<baseline.c_t5_ids.numel();++i) {
                require(baseline.c_t5_ids[i] == std::get<2>(expected)[i], "T5 IDs changed");
                require(baseline.c_t5_weights[i] == std::get<3>(expected)[i], "T5 weights changed");
            }
            for(bool cancel : {false,true}) {
                weights->prepares=0; weights->fail_at=cancel ? 0 : 1;
                static_cast<Conditioner&>(embed).set_execution_cancel_callback(cancel ? std::function<bool()>([] { return true; }) : nullptr);
                auto failed=embed.get_learned_condition(1,params);
                require(failed.empty() && failed.c_t5_ids.empty() && failed.c_t5_weights.empty(), "failure returned partial Anima metadata");
                embed.runner_done();
                weights->prepares=weights->fail_at=0;
                static_cast<Conditioner&>(embed).set_execution_cancel_callback(nullptr);
                auto recovered=embed.get_learned_condition(1,params);
                embed.runner_done();
                require(recovered.c_crossattn.shape() == baseline.c_crossattn.shape(), "recovered hidden shape changed");
                for(int64_t i=0;i<baseline.c_crossattn.numel();++i)
                    require(std::isfinite(recovered.c_crossattn[i]) && recovered.c_crossattn[i] == baseline.c_crossattn[i], "recovery changed output");
                for(int64_t i=0;i<baseline.c_t5_ids.numel();++i) {
                    require(recovered.c_t5_ids[i] == baseline.c_t5_ids[i], "recovered IDs changed");
                    require(recovered.c_t5_weights[i] == baseline.c_t5_weights[i], "recovered T5 weights changed");
                }
            }
        }
        std::puts("PASS Anima CPU LLM failure, runner cancellation, recovery and dual tokenizer metadata contracts");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
