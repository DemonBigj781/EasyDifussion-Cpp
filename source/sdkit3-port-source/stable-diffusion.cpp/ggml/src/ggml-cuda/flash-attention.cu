#include "flash-attention.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "features/attention/flash/common/flash_attention.hpp"

#include <atomic>
#include <cmath>
#include <cstring>

namespace flash_common = edcpp::api::attention::flash;

namespace {

std::atomic<uint64_t> flash_common_launch_count{0};

bool supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 ||
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

flash_common::DType common_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return flash_common::DType::f16;
        case GGML_TYPE_BF16: return flash_common::DType::bf16;
        case GGML_TYPE_F32:
        default: return flash_common::DType::f32;
    }
}

bool read_parameters(
        const ggml_tensor* dst,
        float& scale,
        float& max_bias,
        float& logit_softcap) {
    std::memcpy(&scale, dst->op_params + 0 * sizeof(float), sizeof(scale));
    std::memcpy(
        &max_bias, dst->op_params + 1 * sizeof(float), sizeof(max_bias));
    std::memcpy(
        &logit_softcap,
        dst->op_params + 2 * sizeof(float),
        sizeof(logit_softcap));

    int32_t precision = GGML_PREC_DEFAULT;
    std::memcpy(
        &precision, dst->op_params + 3 * sizeof(float), sizeof(precision));
    return std::isfinite(scale) && std::isfinite(max_bias) &&
           std::isfinite(logit_softcap) && scale >= 0.0f &&
           max_bias >= 0.0f && logit_softcap >= 0.0f &&
           (precision == GGML_PREC_DEFAULT || precision == GGML_PREC_F32);
}

flash_common::Tensor4D common_tensor(const ggml_tensor* tensor) {
    flash_common::Tensor4D result;
    result.data = tensor->data;
    result.dtype = common_dtype(tensor->type);
    result.head_dim = tensor->ne[0];
    result.tokens = tensor->ne[1];
    result.heads = tensor->ne[2];
    result.batch = tensor->ne[3];
    result.byte_strides = {
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]};
    return result;
}

flash_common::MutableTensor4D common_output(ggml_tensor* tensor) {
    flash_common::MutableTensor4D result;
    result.data = tensor->data;
    result.dtype = common_dtype(tensor->type);
    result.head_dim = tensor->ne[0];
    result.tokens = tensor->ne[2];
    result.heads = tensor->ne[1];
    result.batch = tensor->ne[3];
    // GGML FLASH_ATTN_EXT stores [value, head, token, batch].
    result.byte_strides = {
        tensor->nb[0], tensor->nb[2], tensor->nb[1], tensor->nb[3]};
    return result;
}

flash_common::Request common_request(
        int device,
        ggml_tensor* dst,
        void* stream,
        bool synchronize) {
    flash_common::Request request;
    request.query = common_tensor(dst->src[0]);
    request.key = common_tensor(dst->src[1]);
    request.value = common_tensor(dst->src[2]);
    request.output = common_output(dst);
    if (dst->src[3] != nullptr) {
        request.mask = common_tensor(dst->src[3]);
    }
    read_parameters(
        dst, request.scale, request.max_bias, request.logit_softcap);
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
        dst->src[2] == nullptr || dst->src[4] != nullptr ||
        ggml_flash_attn_ext_get_sage_attn(dst) ||
        !supported_type(dst->src[0]->type) ||
        !supported_type(dst->src[1]->type) ||
        !supported_type(dst->src[2]->type) ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->src[3] != nullptr && !supported_type(dst->src[3]->type)) {
        return false;
    }
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    return read_parameters(dst, scale, max_bias, logit_softcap) &&
           (max_bias == 0.0f || dst->src[3] != nullptr);
}

} // namespace

uint64_t ggml_cuda_flash_common_attn_launch_count() noexcept {
    return flash_common_launch_count.load(std::memory_order_relaxed);
}

bool ggml_cuda_flash_common_attn_supported(
        int device, const ggml_tensor* dst) {
    if (!basic_contract_supported(device, dst)) {
        return false;
    }
    auto* mutable_dst = const_cast<ggml_tensor*>(dst);
    const auto request = common_request(device, mutable_dst, nullptr, false);
    return flash_common::validate(
        edcpp::api::Backend::cuda, request).ok;
}

void ggml_cuda_flash_common_attn(
        ggml_backend_cuda_context& context, ggml_tensor* dst) {
    GGML_ASSERT(
        ggml_cuda_flash_common_attn_supported(context.device, dst));
    auto request = common_request(
        context.device,
        dst,
        reinterpret_cast<void*>(context.stream()),
        false);
    const flash_common::Result result = flash_common::forward(
        edcpp::api::Backend::cuda, request);
    GGML_ASSERT(result.ok);
    flash_common_launch_count.fetch_add(1, std::memory_order_relaxed);
}

#else

uint64_t ggml_cuda_flash_common_attn_launch_count() noexcept {
    return 0;
}

bool ggml_cuda_flash_common_attn_supported(
        int device, const ggml_tensor* dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_flash_common_attn(
        ggml_backend_cuda_context& context, ggml_tensor* dst) {
    GGML_UNUSED(context);
    GGML_UNUSED(dst);
    GGML_ABORT("normalized CUDA FlashAttention is unavailable for this target");
}

#endif
