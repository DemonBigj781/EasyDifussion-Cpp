#include <iostream>
#include <stdexcept>

#include "model/te/llm.hpp"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void add(String2TensorStorage& tensors, const std::string& name, std::initializer_list<int64_t> shape) {
    TensorStorage tensor;
    tensor.name = name;
    tensor.n_dims = shape.size();
    int i = 0;
    for (auto dim : shape) tensor.ne[i++] = dim;
    tensors[name] = tensor;
}
static LLM::LLMConfig detect(const String2TensorStorage& tensors, bool& enabled) {
    return LLM::LLMConfig::detect_from_weights(tensors, "llm.", LLM::LLMArch::QWEN3_VL, enabled);
}
static void check(bool split) {
    String2TensorStorage tensors;
    add(tensors, "llm.embed_tokens.weight", {2304,32000});
    add(tensors, "llm.visual.pos_embed.weight", {768,256});
    add(tensors, "llm.visual.merger.linear_fc2.weight", {1024,2304});
    if (split) {
        add(tensors, "llm.visual.patch_embed.proj.0.weight", {14,14,3,768});
        add(tensors, "llm.visual.patch_embed.proj.1.weight", {768,768});
    } else {
        add(tensors, "llm.visual.patch_embed.proj.weight", {14,14,3,768});
    }
    bool enabled = true;
    auto expected = detect(tensors, enabled);
    require(enabled && expected.hidden_size == 2304 && expected.vocab_size == 32000 && expected.vision.patch_size == 14 &&
                expected.vision.hidden_size == 768 && expected.vision.num_position_embeddings == 256 &&
                expected.vision.split_patch_embed == split, "valid weight dimensions not detected");
    for (const auto& name : {"llm.embed_tokens.weight_scale", "llm.visual.pos_embed.weight_scale",
                              "llm.visual.patch_embed.proj.weight_scale", "llm.visual.patch_embed.proj.0.weight_scale",
                              "llm.visual.patch_embed.proj.1.weight_scale"}) {
        add(tensors, name, {17});
    }
    enabled = true;
    auto actual = detect(tensors, enabled);
    require(enabled && actual.hidden_size == expected.hidden_size && actual.vocab_size == expected.vocab_size &&
                actual.vision.patch_size == expected.vision.patch_size && actual.vision.in_channels == expected.vision.in_channels &&
                actual.vision.hidden_size == expected.vision.hidden_size &&
                actual.vision.num_position_embeddings == expected.vision.num_position_embeddings &&
                actual.vision.split_patch_embed == expected.vision.split_patch_embed,
            "scale metadata overwrote model dimensions or split-patch detection");
    add(tensors, "other.embed_tokens.weight", {1,1});
    enabled = true;
    require(detect(tensors, enabled).hidden_size == 2304 && enabled, "unrelated prefix affected metadata");
}
int main() {
    try {
        check(false);
        check(true);
        String2TensorStorage empty;
        bool enabled = false;
        auto defaults = detect(empty, enabled);
        for (auto shape : {std::vector<int64_t>{123}, {123,456,2}}) {
            String2TensorStorage tensors;
            TensorStorage malformed;
            malformed.n_dims = shape.size();
            for (size_t i = 0; i < shape.size(); ++i) malformed.ne[i] = shape[i];
            tensors["llm.embed_tokens.weight"] = malformed;
            tensors["llm.visual.pos_embed.weight"] = malformed;
            enabled = false;
            auto config = detect(tensors, enabled);
            require(config.hidden_size == defaults.hidden_size && config.vocab_size == defaults.vocab_size &&
                        config.vision.hidden_size == defaults.vision.hidden_size &&
                        config.vision.num_position_embeddings == defaults.vision.num_position_embeddings && !enabled,
                    "wrong-rank embedding metadata changed dimensions");
        }
        std::cout << "PASS LLM/vision weight suffix and rank guards, scale metadata isolation, split/unsplit patch detection and prefix controls\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
