#include "xformers-attention.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "features/attention/xformers/common/xformers.hpp"

#include <atomic>
#include <cmath>
#include <cstring>

namespace xformers_common = edcpp::api::attention::xformers;

namespace {

std::atomic<uint64_t> xformers_launch_count{0};

bool supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 ||
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

xformers_common::DType common_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return xformers_common::DType::f16;
        case GGML_TYPE_BF16: return xformers_common::DType::bf16;
        case GGML_TYPE_F32:
        default: return xformers_common::DType::f32;
    }
}

bool read_parameters(
        const ggml_tensor* dst,
        float& scale,
        float& max_bias,
        float& softcap) {
    std::memcpy(&scale, dst->op_params + 0 * sizeof(float), sizeof(scale));
    std::memcpy(&max_bias, dst->op_params + 1 * sizeof(float), sizeof(max_bias));
    std::memcpy(&softcap, dst->op_params + 2 * sizeof(float), sizeof(softcap));

    int32_t precision = GGML_PREC_DEFAULT;
    std::memcpy(
        &precision, dst->op_params + 3 * sizeof(float), sizeof(precision));
    return std::isfinite(scale) && std::isfinite(max_bias) &&
           std::isfinite(softcap) && max_bias >= 0.0f && softcap >= 0.0f &&
           (precision == GGML_PREC_DEFAULT || precision == GGML_PREC_F32);
}

xformers_common::Tensor4D common_tensor(const ggml_tensor* tensor) {
    xformers_common::Tensor4D result;
    result.data = tensor->data;
    result.batch = tensor->ne[3];
    result.heads = tensor->ne[2];
    result.tokens = tensor->ne[1];
    result.head_dim = tensor->ne[0];
    result.dtype = common_dtype(tensor->type);
    result.byte_strides = {
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]};
    return result;
}

xformers_common::MutableTensor4D common_output(ggml_tensor* tensor) {
    xformers_common::MutableTensor4D result;
    result.data = tensor->data;
    result.batch = tensor->ne[3];
    result.heads = tensor->ne[1];
    result.tokens = tensor->ne[2];
    result.head_dim = tensor->ne[0];
    result.dtype = common_dtype(tensor->type);
    // GGML FLASH_ATTN_EXT stores output as [value, head, token, batch].
    result.byte_strides = {
        tensor->nb[0], tensor->nb[2], tensor->nb[1], tensor->nb[3]};
    return result;
}

xformers_common::AttentionRequest common_request(
        int device,
        ggml_tensor* dst,
        void* stream,
        bool synchronize) {
    xformers_common::AttentionRequest request;
    const ggml_tensor* query = dst->src[0];
    const ggml_tensor* key = dst->src[1];
    const ggml_tensor* value = dst->src[2];
    const ggml_tensor* mask = dst->src[3];
    const ggml_tensor* sinks = dst->src[4];

    request.q = common_tensor(query);
    request.k = common_tensor(key);
    request.v = common_tensor(value);
    request.out = common_output(dst);
    request.dtype = request.q.dtype;

    float max_bias = 0.0f;
    read_parameters(dst, request.scale, max_bias, request.softcap);
    if (mask != nullptr) {
        request.mask.data = mask->data;
        request.mask.batch = mask->ne[3];
        request.mask.heads = mask->ne[2];
        request.mask.query_tokens = mask->ne[1];
        request.mask.key_tokens = mask->ne[0];
        request.mask.dtype = common_dtype(mask->type);
        request.mask.byte_strides = {
            mask->nb[0], mask->nb[1], mask->nb[2], mask->nb[3]};
    }
    if (max_bias > 0.0f) {
        request.alibi.enabled = true;
        request.alibi.max_bias = max_bias;
    }
    if (sinks != nullptr) {
        request.sinks.enabled = true;
        request.sinks.values = static_cast<const float*>(sinks->data);
        request.sinks.value_count = sinks->ne[0];
    }
    request.execution.stream = stream;
    request.execution.synchronize = synchronize;
    request.execution.device_index = device;
    return request;
}

bool basic_contract_supported(int device, const ggml_tensor* dst) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT ||
        device < 0 || device >= ggml_cuda_info().device_count ||
        !GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[device].cc) ||
        ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_PASCAL ||
        dst->src[0] == nullptr || dst->src[1] == nullptr ||
        dst->src[2] == nullptr || !supported_type(dst->src[0]->type) ||
        !supported_type(dst->src[1]->type) ||
        !supported_type(dst->src[2]->type) ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor* mask = dst->src[3];
    const ggml_tensor* sinks = dst->src[4];
    if (mask != nullptr && !supported_type(mask->type)) {
        return false;
    }
    if (sinks != nullptr && sinks->type != GGML_TYPE_F32) {
        return false;
    }
    float scale = 0.0f;
    float max_bias = 0.0f;
    float softcap = 0.0f;
    return read_parameters(dst, scale, max_bias, softcap) &&
           (max_bias == 0.0f || mask != nullptr);
}

} // namespace

uint64_t ggml_cuda_xformers_attn_launch_count() noexcept {
    return xformers_launch_count.load(std::memory_order_relaxed);
}

bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor* dst) {
    if (!basic_contract_supported(device, dst)) {
        return false;
    }
    auto* mutable_dst = const_cast<ggml_tensor*>(dst);
    const auto request = common_request(device, mutable_dst, nullptr, false);
    return xformers_common::validate(
        edcpp::api::Backend::cuda, request).ok;
}

void ggml_cuda_xformers_attn(
        ggml_backend_cuda_context& context, ggml_tensor* dst) {
    GGML_ASSERT(
        ggml_cuda_xformers_attn_supported(context.device, dst));
    auto request = common_request(
        context.device,
        dst,
        reinterpret_cast<void*>(context.stream()),
        false);
    GGML_ASSERT(
        xformers_common::forward(edcpp::api::Backend::cuda, request));
    xformers_launch_count.fetch_add(1, std::memory_order_relaxed);
}

#else

uint64_t ggml_cuda_xformers_attn_launch_count() noexcept {
    return 0;
}

bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor* dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_xformers_attn(
        ggml_backend_cuda_context& context, ggml_tensor* dst) {
    GGML_UNUSED(context);
    GGML_UNUSED(dst);
    GGML_ABORT("xFormers attention is not available for this target");
}

#endif
