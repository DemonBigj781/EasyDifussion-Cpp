#include "features/attention/flash/common/flash_attention.hpp"
#include "features/attention/flash/cpu/definition/cpu/flash_attention_cpu.hpp"

#include "ggml-cpu-impl.h"

#include <array>
#include <cstring>

namespace edcpp::api::attention::flash::cpu::translation {
namespace {

std::size_t element_size(DType dtype) noexcept {
    return dtype == DType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

ggml_type native_type(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return GGML_TYPE_F32;
        case DType::f16: return GGML_TYPE_F16;
        case DType::bf16: return GGML_TYPE_BF16;
    }
    return GGML_TYPE_COUNT;
}

template <typename Tensor>
void set_layout(ggml_tensor& native, const Tensor& tensor) noexcept {
    native.ne[0] = tensor.head_dim;
    native.ne[1] = tensor.tokens;
    native.ne[2] = tensor.heads;
    native.ne[3] = tensor.batch;
    const auto item_size = element_size(tensor.dtype);
    const std::array<std::size_t, 4> contiguous = {
        item_size,
        item_size * static_cast<std::size_t>(tensor.head_dim),
        item_size * static_cast<std::size_t>(tensor.head_dim * tensor.tokens),
        item_size * static_cast<std::size_t>(tensor.head_dim * tensor.tokens * tensor.heads),
    };
    for (std::size_t i = 0; i < 4; ++i) {
        native.nb[i] = tensor.byte_strides[i] == 0 ? contiguous[i] : tensor.byte_strides[i];
    }
}

template <typename Tensor>
ggml_tensor make_tensor(const Tensor& tensor) noexcept {
    ggml_tensor native{};
    native.type = native_type(tensor.dtype);
    native.data = const_cast<void*>(static_cast<const void*>(tensor.data));
    set_layout(native, tensor);
    return native;
}

Result validate_cpu(const Request& request) noexcept {
    if (request.execution.thread_index != 0 || request.execution.thread_count != 1) {
        return {false, "CPU FlashAttention Common route currently supports single-thread execution only"};
    }
    if (request.query.dtype != DType::f32 || request.output.dtype != DType::f32) {
        return {false, "CPU FlashAttention requires F32 query and output tensors"};
    }
    if (request.mask.data != nullptr && request.mask.dtype != DType::f16) {
        return {false, "CPU FlashAttention requires an F16 additive mask"};
    }
    if (request.mask.data != nullptr &&
        (request.mask.head_dim != request.key.tokens || request.mask.tokens != request.query.tokens ||
         request.query.heads % request.mask.heads != 0 || request.query.batch % request.mask.batch != 0)) {
        return {false, "CPU FlashAttention mask shape is not broadcast-compatible"};
    }
    const auto row_contiguous = [](const auto& tensor) {
        return tensor.byte_strides[0] == 0 || tensor.byte_strides[0] == element_size(tensor.dtype);
    };
    if (!row_contiguous(request.query) || !row_contiguous(request.key) || !row_contiguous(request.value) ||
        !row_contiguous(request.output)) {
        return {false, "CPU FlashAttention requires contiguous tensor rows"};
    }
    return {true, nullptr};
}

Result forward_cpu(const Request& request) noexcept {
    auto query = make_tensor(request.query);
    auto key = make_tensor(request.key);
    auto value = make_tensor(request.value);
    auto output = make_tensor(request.output);
    ggml_tensor mask{};
    ggml_tensor* mask_ptr = nullptr;
    if (request.mask.data != nullptr) {
        mask = make_tensor(request.mask);
        mask_ptr = &mask;
    }

    output.op = GGML_OP_FLASH_ATTN_EXT;
    output.src[0] = &query;
    output.src[1] = &key;
    output.src[2] = &value;
    output.src[3] = mask_ptr;
    const float params[] = {request.scale, request.max_bias, request.logit_softcap};
    std::memcpy(output.op_params, params, sizeof(params));
    output.op_params[3] = GGML_PREC_F32;

    ggml_compute_params execution{};
    execution.ith = request.execution.thread_index;
    execution.nth = request.execution.thread_count;
    execution.wdata = request.execution.workspace;
    execution.wsize = request.execution.workspace_size;
    execution.threadpool = nullptr;
    execution.use_ref = request.execution.reference;

    return definition::forward(&execution, &output)
        ? Result{true, nullptr}
        : Result{false, "CPU FlashAttention native execution failed"};
}

const Translation cpu_translation = [] {
    Translation translation;
    translation.backend = Backend::cpu;
    translation.name = "cpu";
    translation.capabilities.forward = true;
    translation.capabilities.additive_mask = true;
    translation.capabilities.alibi_bias = true;
    translation.capabilities.logit_softcap = true;
    translation.capabilities.grouped_query = true;
    translation.capabilities.f32_accumulation = true;
    translation.validate = &validate_cpu;
    translation.forward = &forward_cpu;
    return translation;
}();

const bool registered = register_translation(&cpu_translation);

} // namespace
} // namespace edcpp::api::attention::flash::cpu::translation
