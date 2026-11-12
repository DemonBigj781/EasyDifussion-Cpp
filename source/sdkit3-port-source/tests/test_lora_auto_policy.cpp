#include <iostream>
#include <stdexcept>

#include "model_loader.h"
#include "runtime/lora_apply_policy.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static ModelLoader make_loader() {
    ModelLoader loader;
    const int64_t shape[] = {256, 256};
    const std::string name = "model.diffusion_model.test.weight";
    loader.get_tensor_storage_map()[name] = TensorStorage(name, GGML_TYPE_F32, shape, 2, 0);
    return loader;
}
int main() {
    try {
        auto loader = make_loader();
        auto allowed = [&] { return sd::lora_auto_merge_allowed(loader.get_tensor_storage_map(), false, false, false); };
        require(allowed(), "default float weights should permit immediate auto merge");
        loader.set_wtype_override(GGML_TYPE_Q4_0, "");
        require(loader.get_wtype_stat().count(GGML_TYPE_Q4_0) == 0 && loader.get_wtype_stat().count(GGML_TYPE_F32) == 1,
                "fixture must retain source dtype stats before runtime conversion");
        require(loader.get_tensor_storage_map().begin()->second.expected_type == GGML_TYPE_Q4_0 && !allowed(),
                "global quantized conversion incorrectly allowed immediate merge");
        loader.get_tensor_storage_map().begin()->second.expected_type = GGML_TYPE_COUNT;
        loader.set_wtype_override(GGML_TYPE_COUNT, "test.weight=q8_0");
        require(loader.get_tensor_storage_map().begin()->second.expected_type == GGML_TYPE_Q8_0 && !allowed(),
                "quantized tensor rule incorrectly allowed immediate merge");
        loader.get_tensor_storage_map().begin()->second.expected_type = GGML_TYPE_COUNT;
        loader.set_wtype_override(GGML_TYPE_COUNT, "no_match=q8_0");
        require(allowed(), "unmatched quantization rule changed merge policy");
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_COUNT}) {
            loader.get_tensor_storage_map().begin()->second.expected_type = type;
            require(allowed(), "float/sentinel target unexpectedly disabled merge");
        }
        for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q6_K}) {
            loader.get_tensor_storage_map().begin()->second.expected_type = type;
            require(!allowed(), "quantized target was not detected");
        }
        loader.get_tensor_storage_map().begin()->second.expected_type = GGML_TYPE_COUNT;
        for (int flags = 0; flags < 8; ++flags) {
            require(sd::lora_auto_merge_allowed(loader.get_tensor_storage_map(), flags & 1, flags & 2, flags & 4) == (flags == 0),
                    "existing quantized/streaming/row-split restrictions changed");
        }
        // Quantized rule may follow another tensor in a mixed model.
        const int64_t shape[] = {256,256};
        loader.get_tensor_storage_map()["z.weight"] = TensorStorage("z.weight", GGML_TYPE_F16, shape, 2, 0);
        loader.get_tensor_storage_map()["z.weight"].expected_type = GGML_TYPE_Q5_K;
        require(!allowed(), "later mixed-model override was missed");
        std::cout << "PASS auto LoRA policy: real loader quantization rules, source-vs-target dtype, mixed metadata and existing constraints\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
