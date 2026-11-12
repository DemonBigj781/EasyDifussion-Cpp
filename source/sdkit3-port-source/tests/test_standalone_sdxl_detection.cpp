#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

#include "model_loader.h"

static void add_tensor(ModelLoader& loader, const std::string& name, std::initializer_list<int64_t> dims = {1}) {
    TensorStorage tensor;
    tensor.name = name;
    tensor.n_dims = static_cast<int>(dims.size());
    int index = 0;
    for (auto dim : dims) tensor.ne[index++] = dim;
    loader.get_tensor_storage_map().emplace(name, tensor);
}

static void expect(ModelLoader& loader, SDVersion version) {
    auto actual = loader.get_sd_version();
    if (actual != version) {
        throw std::runtime_error("expected " + std::to_string(version) + ", got " + std::to_string(actual));
    }
}

static ModelLoader make_unet(bool diffusers, int channels, bool middle = true, bool deep = false) {
    ModelLoader loader;
    const std::string prefix = "model.diffusion_model.";
    add_tensor(loader, prefix + (diffusers ? "down_blocks.0.resnets.0.conv1.weight" : "input_blocks.1.0.in_layers.2.weight"));
    add_tensor(loader, prefix + (diffusers ? "conv_in.weight" : "input_blocks.0.0.weight"), {3, 3, channels, 320});
    if (middle) add_tensor(loader, prefix + (diffusers ? "mid_block.resnets.1.conv1.weight" : "middle_block.1.transformer_blocks.0.attn1.to_q.weight"));
    if (deep) add_tensor(loader, prefix + (diffusers ? "up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q.weight" : "output_blocks.3.1.transformer_blocks.1.attn1.to_q.weight"));
    add_tensor(loader, prefix + (diffusers ? "add_embedding.linear_1.weight" : "label_emb.0.0.weight"));
    return loader;
}

int main() {
    try {
        for (bool diffusers : {false, true}) {
            auto standard = make_unet(diffusers, 4);
            expect(standard, VERSION_SDXL);
            auto inpaint = make_unet(diffusers, 9);
            expect(inpaint, VERSION_SDXL_INPAINT);
            auto pix2pix = make_unet(diffusers, 8);
            expect(pix2pix, VERSION_SDXL_PIX2PIX);
            auto vega = make_unet(diffusers, 4, false);
            expect(vega, VERSION_SDXL_VEGA);
            auto ssd = make_unet(diffusers, 4, false, true);
            expect(ssd, VERSION_SDXL_SSD1B);
            auto clip_l_only = make_unet(diffusers, 4);
            add_tensor(clip_l_only, "clip_l.text_model.embeddings.token_embedding.weight", {768, 49408});
            expect(clip_l_only, VERSION_SDXL);
            auto video = make_unet(diffusers, 4);
            add_tensor(video, "model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor");
            expect(video, VERSION_SVD);
            auto custom = make_unet(diffusers, 4);
            add_tensor(custom, "llm_adapter.blocks.0.cross_attn.q_proj.weight");
            expect(custom, VERSION_ANIMA);
        }
        auto unrelated = make_unet(true, 4);
        unrelated.get_tensor_storage_map().erase("model.diffusion_model.add_embedding.linear_1.weight");
        add_tensor(unrelated, "unrelated.add_embedding.linear_1.weight");
        add_tensor(unrelated, "clip_l.text_model.embeddings.token_embedding.weight", {768, 49408});
        expect(unrelated, VERSION_SD1);
        ModelLoader non_unet;
        add_tensor(non_unet, "model.diffusion_model.add_embedding.linear_1.weight");
        expect(non_unet, VERSION_COUNT);
        std::cout << "PASS standalone SDXL base/inpaint/pix2pix/SSD/Vega, encoder independence, video/custom-family precedence and negative controls\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
