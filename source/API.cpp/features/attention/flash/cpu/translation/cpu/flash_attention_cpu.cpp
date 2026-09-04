#include "features/attention/flash/common/flash_attention.hpp"
#include "features/attention/flash/cpu/definition/cpu/flash_attention_cpu.hpp"

namespace edcpp::api::attention::flash::cpu::translation {
namespace {

ValidationResult validate_cpu(const Request& request) noexcept {
    if (request.compute_params == nullptr) {
        return {false, "CPU FlashAttention requires compute parameters"};
    }
    if (request.destination == nullptr) {
        return {false, "CPU FlashAttention requires a destination tensor"};
    }

    const auto* destination = static_cast<const ggml_tensor*>(request.destination);
    if (!definition::supported(destination)) {
        return {false, "CPU FlashAttention requires GGML_OP_FLASH_ATTN_EXT"};
    }
    return {true, nullptr};
}

bool forward_cpu(const Request& request) noexcept {
    const auto* params = static_cast<const ggml_compute_params*>(request.compute_params);
    auto* destination = static_cast<ggml_tensor*>(request.destination);
    return definition::forward(params, destination);
}

const Translation cpu_translation = [] {
    Translation translation;
    translation.backend = Backend::cpu;
    translation.name = "cpu-ggml-flash-attn-ext";
    translation.capabilities.forward = true;
    translation.capabilities.ggml_flash_attn_ext = true;
    translation.validate = &validate_cpu;
    translation.forward = &forward_cpu;
    return translation;
}();

const bool registered = register_translation(&cpu_translation);

} // namespace
} // namespace edcpp::api::attention::flash::cpu::translation
