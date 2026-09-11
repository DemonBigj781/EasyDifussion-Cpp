#include <iostream>
#include <memory>

#include "model_manager.h"

struct ModelManagerTestAccess {
    static ModelManager::TensorState* seed_loaded_tensor(ModelManager& manager, ggml_tensor* tensor) {
        auto state                      = std::make_unique<ModelManager::TensorState>();
        state->name                     = "test.weight";
        state->tensor                   = tensor;
        state->loaded_to_params_backend = true;
        ModelManager::TensorState* raw  = state.get();
        manager.tensor_states_.push_back(std::move(state));

        auto block = std::make_unique<ModelManager::ParamsStorageBlock>();
        block->states.push_back(raw);
        manager.params_storage_blocks_.push_back(std::move(block));
        return raw;
    }

    static bool is_loaded(const ModelManager::TensorState* state) {
        return state != nullptr && state->loaded_to_params_backend;
    }

    static size_t storage_block_count(const ModelManager& manager) {
        return manager.params_storage_blocks_.size();
    }
};

int main() {
    ggml_init_params params{};
    params.mem_size = 16 * 1024;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::cerr << "failed to create ggml test context" << std::endl;
        return 1;
    }

    bool passed = true;
    {
        ModelManager manager;
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        auto* state = ModelManagerTestAccess::seed_loaded_tensor(manager, tensor);

        ModelManager::LoraSpec first;
        first.path       = "first.safetensors";
        first.multiplier = 1.0f;
        manager.set_loras({first}, VERSION_SDXL);

        if (!ModelManagerTestAccess::is_loaded(state) ||
            ModelManagerTestAccess::storage_block_count(manager) != 1) {
            std::cerr << "initial LoRA selection discarded pristine eager weights" << std::endl;
            passed = false;
        }

        manager.set_loras({first}, VERSION_SDXL);
        if (!ModelManagerTestAccess::is_loaded(state) ||
            ModelManagerTestAccess::storage_block_count(manager) != 1) {
            std::cerr << "unchanged LoRA selection discarded cached weights" << std::endl;
            passed = false;
        }

        ModelManager::LoraSpec replacement = first;
        replacement.multiplier             = 0.5f;
        manager.set_loras({replacement}, VERSION_SDXL);
        if (ModelManagerTestAccess::is_loaded(state) ||
            ModelManagerTestAccess::storage_block_count(manager) != 0) {
            std::cerr << "changed LoRA selection retained mutated weights" << std::endl;
            passed = false;
        }
    }

    ggml_free(ctx);
    return passed ? 0 : 1;
}
