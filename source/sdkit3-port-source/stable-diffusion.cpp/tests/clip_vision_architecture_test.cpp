#include <initializer_list>
#include <iostream>
#include <string>

#include "model/te/clip.hpp"

namespace {

TensorStorage make_tensor(const std::string& name, std::initializer_list<int64_t> dimensions) {
    TensorStorage tensor;
    tensor.name   = name;
    tensor.n_dims = static_cast<int>(dimensions.size());

    int index = 0;
    for (const int64_t dimension : dimensions) {
        tensor.ne[index++] = dimension;
    }
    return tensor;
}

bool expect_version(const char* case_name, int64_t hidden_size, CLIPVersion expected) {
    const std::string prefix = "cond_stage_model.transformer";
    String2TensorStorage tensors;
    TensorStorage tensor = make_tensor(prefix + ".vision_model.embeddings.class_embedding",
                                       {hidden_size});
    tensors.emplace(tensor.name, tensor);

    const CLIPVersion actual = detect_clip_vision_version(tensors, prefix);
    if (actual == expected) {
        return true;
    }

    std::cerr << case_name << ": expected version " << expected << ", got " << actual << '\n';
    return false;
}

}  // namespace

int main() {
    bool passed = true;
    passed &= expect_version("ViT-L", 1024, OPENAI_CLIP_VIT_L_14);
    passed &= expect_version("ViT-H", 1280, OPEN_CLIP_VIT_H_14);
    passed &= expect_version("ViT-bigG", 1664, OPEN_CLIP_VIT_BIGG_14);

    String2TensorStorage tensors;
    TensorStorage patch = make_tensor(
        "cond_stage_model.transformer.vision_model.embeddings.patch_embedding.weight",
        {14, 14, 3, 1664});
    tensors.emplace(patch.name, patch);
    if (detect_clip_vision_version(tensors, "cond_stage_model.transformer") != OPEN_CLIP_VIT_BIGG_14) {
        std::cerr << "patch embedding fallback did not detect ViT-bigG\n";
        passed = false;
    }

    if (detect_clip_vision_version({}, "cond_stage_model.transformer", OPENAI_CLIP_VIT_L_14) !=
        OPENAI_CLIP_VIT_L_14) {
        std::cerr << "explicit fallback was not retained\n";
        passed = false;
    }

    return passed ? 0 : 1;
}
