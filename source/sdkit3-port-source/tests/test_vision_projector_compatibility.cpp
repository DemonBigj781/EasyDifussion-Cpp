#include <iostream>
#include <stdexcept>

#include "model/te/llm.hpp"

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static String2TensorStorage weights(const std::string& projector, int64_t output, int64_t hidden = 4096) {
    String2TensorStorage tensors;
    tensors["llm.embed_tokens.weight"].n_dims = 2;
    tensors["llm.embed_tokens.weight"].ne[0] = hidden;
    tensors["llm.embed_tokens.weight"].ne[1] = 32000;
    if (!projector.empty()) {
        tensors["llm.visual." + projector].n_dims = 2;
        tensors["llm.visual." + projector].ne[0] = 1024;
        tensors["llm.visual." + projector].ne[1] = output;
    }
    return tensors;
}

int main() {
    try {
        for (const std::string& key : {std::string("merger.linear_fc2.weight"), std::string("merger.mlp.2.weight")}) {
            bool enabled = true;
            auto config = LLM::LLMConfig::detect_from_weights(weights(key, 4096), "llm.", LLM::LLMArch::QWEN3_VL, enabled);
            require(enabled && config.have_vision_weight && config.hidden_size == 4096 && config.vision.out_hidden_size == 4096,
                    "compatible projector rejected");
            enabled = true;
            config = LLM::LLMConfig::detect_from_weights(weights(key, 3584), "llm.", LLM::LLMArch::QWEN3_VL, enabled);
            require(!enabled && config.vision.out_hidden_size == 3584 && config.hidden_size == 4096,
                    "incompatible projector accepted");
            enabled = false;
            LLM::LLMConfig::detect_from_weights(weights(key, 4096), "llm.", LLM::LLMArch::QWEN3_VL, enabled);
            require(!enabled, "caller-disabled vision enabled");
        }
        bool enabled = true;
        LLM::LLMConfig::detect_from_weights(weights("", 0), "llm.", LLM::LLMArch::QWEN3_VL, enabled);
        require(!enabled, "absent vision weights accepted");
        enabled = true;
        auto config = LLM::LLMConfig::detect_from_weights(weights("pos_embed.weight", 256), "llm.", LLM::LLMArch::QWEN3_VL, enabled);
        require(enabled && config.vision.out_hidden_size != config.hidden_size,
                "undetected default projector dimension incorrectly rejected");
        auto tensors = weights("merger.linear_fc2.weight", 4096);
        tensors["other.visual.merger.linear_fc2.weight"].ne[1] = 1;
        enabled = true;
        LLM::LLMConfig::detect_from_weights(tensors, "llm.", LLM::LLMArch::QWEN3_VL, enabled);
        require(enabled, "unrelated prefix affected projector detection");
        std::cout << "PASS both projector names, matching/mismatching dimensions, disabled/missing vision, undetected defaults and prefix isolation\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
