#include <initializer_list>
#include <iostream>
#include <string>

#include "model_loader.h"

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

void add_tensor(ModelLoader& loader, const TensorStorage& tensor) {
    loader.get_tensor_storage_map().emplace(tensor.name, tensor);
}

bool expect_version(const char* case_name, ModelLoader& loader, SDVersion expected) {
    const SDVersion actual = loader.get_sd_version();
    if (actual == expected) {
        return true;
    }

    std::cerr << case_name << ": expected version " << expected << ", got " << actual << '\n';
    return false;
}

ModelLoader make_prefixed_diffusers_loader(int64_t input_channels) {
    ModelLoader loader;
    add_tensor(loader, make_tensor("model.diffusion_model.down_blocks.0.resnets.0.conv1.weight", {3, 3, 320, 320}));
    add_tensor(loader, make_tensor("model.diffusion_model.mid_block.resnets.1.conv1.weight", {3, 3, 1280, 1280}));
    add_tensor(loader, make_tensor("model.diffusion_model.up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight", {320, 320}));
    add_tensor(loader, make_tensor("model.diffusion_model.conv_in.weight", {3, 3, input_channels, 320}));
    add_tensor(loader, make_tensor("clip_l.text_model.embeddings.token_embedding.weight", {768, 49408}));
    return loader;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3) {
        ModelLoader file_loader;
        if (!file_loader.init_from_file(argv[1], "model.diffusion_model.") ||
            !file_loader.init_from_file(argv[2], "clip_l.")) {
            std::cerr << "failed to load integration-test model files\n";
            return 1;
        }

        bool passed = expect_version("standalone SD1 model files", file_loader, VERSION_SD1);
        const auto& tensors = file_loader.get_tensor_storage_map();
        const auto input = tensors.find("model.diffusion_model.conv_in.weight");
        if (input == tensors.end() || input->second.n_dims != 4 ||
            input->second.ne[0] != 3 || input->second.ne[1] != 3 ||
            input->second.ne[2] != 4 || input->second.ne[3] != 320 ||
            input->second.storage_n_dims != 2 ||
            input->second.storage_ne[0] != 256 || input->second.storage_ne[1] != 45) {
            std::cerr << "standalone SD1 GGUF did not restore the logical conv_in shape\n";
            passed = false;
        }
        return passed ? 0 : 1;
    }

    bool passed = true;

    ModelLoader sd1_loader = make_prefixed_diffusers_loader(4);
    passed &= expect_version("prefixed standalone SD1", sd1_loader, VERSION_SD1);

    ModelLoader inpaint_loader = make_prefixed_diffusers_loader(9);
    passed &= expect_version("prefixed standalone SD1 inpaint", inpaint_loader, VERSION_SD1_INPAINT);

    ModelLoader sdxl_loader = make_prefixed_diffusers_loader(4);
    add_tensor(sdxl_loader, make_tensor("clip_g.text_model.embeddings.token_embedding.weight", {1280, 49408}));
    passed &= expect_version("prefixed standalone SDXL", sdxl_loader, VERSION_SDXL);

    ModelLoader existing_unet_loader;
    add_tensor(existing_unet_loader, make_tensor("unet.down_blocks.0.resnets.0.conv1.weight", {3, 3, 320, 320}));
    add_tensor(existing_unet_loader, make_tensor("unet.mid_block.resnets.1.conv1.weight", {3, 3, 1280, 1280}));
    add_tensor(existing_unet_loader, make_tensor("unet.conv_in.weight", {3, 3, 4, 320}));
    add_tensor(existing_unet_loader, make_tensor("text_model.embeddings.token_embedding.weight", {768, 49408}));
    passed &= expect_version("existing unet prefix", existing_unet_loader, VERSION_SD1);

    return passed ? 0 : 1;
}
