#include "sage-attention.cuh"
#include "../../../../flash/cuda/translation/gpu/fattn.cuh"
#include "features/attention/sage/common/sage_attention.hpp"
#include "features/attention/sage/cuda/translation/gpu/support.hpp"

#include <cstring>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// Non-CUDA implementation hook. Until native ROCm/MUSA Sage kernels are wired,
// report unsupported so the shared dispatcher selects an existing attention
// path instead of referencing CUDA-only symbols.
bool ggml_sage_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("SageAttention is not available for this target");
}

#else

#include "sage-attention-sm80.cuh"

namespace {

namespace sage_common = edcpp::api::attention::sage;

sage_common::Tensor4D common_tensor(const ggml_tensor* tensor) {
    sage_common::Tensor4D result;
    result.data = tensor->data;
    result.dtype = tensor->type == GGML_TYPE_F32
                       ? sage_common::DType::f32
                       : sage_common::DType::f16;
    result.head_dim = tensor->ne[0];
    result.tokens = tensor->ne[1];
    result.heads = tensor->ne[2];
    result.batch = tensor->ne[3];
    result.byte_strides = {
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]};
    return result;
}

sage_common::SupportRequest common_support_request(
        int device, ggml_tensor* dst) {
    sage_common::SupportRequest request;
    request.query = common_tensor(dst->src[0]);
    request.key = common_tensor(dst->src[1]);
    request.value = common_tensor(dst->src[2]);
    request.output.data = dst->data;
    request.output.dtype = sage_common::DType::f32;
    request.output.head_dim = dst->ne[0];
    request.output.tokens = dst->ne[2];
    request.output.heads = dst->ne[1];
    request.output.batch = dst->ne[3];
    request.output.byte_strides = {
        dst->nb[0], dst->nb[2], dst->nb[1], dst->nb[3]};

    const int cc = ggml_cuda_info().devices[device].cc;
    request.device = {device, cc / 100, (cc % 100) / 10};
    std::memcpy(&request.scale, dst->op_params, sizeof(request.scale));
    std::memcpy(
        &request.max_bias,
        reinterpret_cast<const float*>(dst->op_params) + 1,
        sizeof(request.max_bias));
    std::memcpy(
        &request.logit_softcap,
        reinterpret_cast<const float*>(dst->op_params) + 2,
        sizeof(request.logit_softcap));
    request.additive_mask = dst->src[3] != nullptr;
    request.attention_sinks = dst->src[4] != nullptr;
    return request;
}

} // namespace

bool ggml_sage_attn_supported(int device, const ggml_tensor * dst) {
    if (!ggml_cuda_sage_attn_sm80_supported(device, dst)) {
        return false;
    }
    namespace sage_translation =
        edcpp::api::attention::sage::cuda::translation::gpu;
    if (!sage_translation::ensure_registered()) {
        return false;
    }
    auto* mutable_dst = const_cast<ggml_tensor*>(dst);
    return sage_common::support(
        edcpp::api::Backend::cuda,
        common_support_request(device, mutable_dst)).ok;
}

void ggml_sage_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_sage_attn_sm80(ctx, dst);
}

#endif

// FlashAttention retention is an API-compatibility contract, not an automatic
// optimization choice. The actual CUDA/HIP kernels remain in fattn.cu; this
// common API owns backend/architecture policy and delegates execution to them.
bool ggml_flash_compat_supported(int device, const ggml_tensor * dst) {
    if (dst == nullptr) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;

#if defined(GGML_USE_HIP)
    // ROCm/HIP uses the shared fattn implementation. Keep architecture-specific
    // kernel selection inside fattn.cu (MFMA/WMMA/tile/vector) and expose only
    // the compatibility contract here.
    if (!GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_supported(device, dst);
#else
    // Pascal gets a dedicated opcode/native implementation later. Do not allow
    // the generic compatibility API to silently treat Pascal as Volta+.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    return ggml_cuda_flash_attn_ext_supported(device, dst);
#endif
}

void ggml_flash_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_flash_compat_supported(ctx.device, dst));
    ggml_cuda_flash_attn_ext(ctx, dst);
}

bool ggml_attention_impl_supported(ggml_attention_impl impl, int device, const ggml_tensor * dst) {
    switch (impl) {
        case GGML_ATTN_IMPL_SAGE:
            return ggml_sage_attn_supported(device, dst);
        case GGML_ATTN_IMPL_XFORMERS:
            // xFormers is selected by the GGML application adapter, which
            // calls the feature Common API. It is not a Sage backend route.
            return false;
        case GGML_ATTN_IMPL_NONE:
        default:
            return false;
    }
}
