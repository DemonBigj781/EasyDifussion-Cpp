#include "features/attention/xformers/common/xformers.hpp"
#include "features/attention/xformers/cuda/definition/gpu/xformers.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <cstdint>

// Static-library builds must retain this translation object so its registration
// initializer can publish the CUDA callbacks to Common. Build systems reference
// this C-linkage symbol; application code does not call the backend directly.
extern "C" void edcpp_xformers_cuda_translation_link_anchor() {}

namespace edcpp::api::attention::xformers::cuda::translation::gpu {
namespace {

namespace native = edcpp::api::attention::xformers::cuda::definition::gpu;

constexpr int threads_per_block = 256;
constexpr int values_per_thread = 2;

struct FusedForwardParams {
    const char* q = nullptr;
    const char* k = nullptr;
    const char* v = nullptr;
    const char* mask = nullptr;
    const char* sinks = nullptr;
    const float* alibi_slopes = nullptr;
    char* out = nullptr;

    std::int64_t q_batches = 0;
    std::int64_t k_batches = 0;
    std::int64_t v_batches = 0;
    std::int64_t q_heads = 0;
    std::int64_t k_heads = 0;
    std::int64_t v_heads = 0;
    std::int64_t q_tokens = 0;
    std::int64_t kv_tokens = 0;
    std::int64_t q_dim = 0;
    std::int64_t v_dim = 0;

    std::int64_t mask_batch = 1;
    std::int64_t mask_heads = 1;
    std::int64_t mask_q_tokens = 1;
    std::int64_t mask_k_tokens = 1;

    std::size_t q_nb0 = 0, q_nb1 = 0, q_nb2 = 0, q_nb3 = 0;
    std::size_t k_nb0 = 0, k_nb1 = 0, k_nb2 = 0, k_nb3 = 0;
    std::size_t v_nb0 = 0, v_nb1 = 0, v_nb2 = 0, v_nb3 = 0;
    std::size_t out_nb0 = 0, out_nb1 = 0, out_nb2 = 0, out_nb3 = 0;
    std::size_t mask_nb0 = 0, mask_nb1 = 0, mask_nb2 = 0, mask_nb3 = 0;

    int mask_dtype = 0;
    float scale = 0.0f;
    float softcap = 0.0f;
    float max_bias = 0.0f;
    float alibi_m0 = 0.0f;
    float alibi_m1 = 0.0f;
    std::uint32_t head_log2 = 1;
    bool causal = false;
    bool explicit_alibi = false;
};

template <typename T>
__device__ __forceinline__ float load_value(const char* pointer) {
    return static_cast<float>(*reinterpret_cast<const T*>(pointer));
}

template <>
__device__ __forceinline__ float load_value<half>(const char* pointer) {
    return __half2float(*reinterpret_cast<const half*>(pointer));
}

template <>
__device__ __forceinline__ float load_value<nv_bfloat16>(const char* pointer) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(*reinterpret_cast<const std::uint16_t*>(pointer)) << 16;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float load_mask_value(
        const char* pointer, int dtype) {
    if (dtype == static_cast<int>(DType::f16)) {
        return load_value<half>(pointer);
    }
    if (dtype == static_cast<int>(DType::bf16)) {
        return load_value<nv_bfloat16>(pointer);
    }
    return load_value<float>(pointer);
}

__device__ __forceinline__ float ggml_alibi_slope(
        const FusedForwardParams& p, std::int64_t head) {
    return head < p.head_log2
        ? powf(p.alibi_m0, static_cast<float>(head + 1))
        : powf(p.alibi_m1, static_cast<float>(2 * (head - p.head_log2) + 1));
}

__device__ __forceinline__ void online_softmax_update(
        float score,
        float& running_max,
        float& running_sum,
        float& alpha,
        float& beta) {
    if (isnan(score)) {
        running_max = score;
        running_sum = score;
        alpha = score;
        beta = score;
        return;
    }

    if (score == -CUDART_INF_F) {
        alpha = 1.0f;
        beta = 0.0f;
        return;
    }

    if (running_sum == 0.0f) {
        running_max = score;
        running_sum = 1.0f;
        alpha = 0.0f;
        beta = 1.0f;
        return;
    }

    if (score == CUDART_INF_F) {
        if (running_max == CUDART_INF_F) {
            alpha = 1.0f;
            beta = 1.0f;
            running_sum += 1.0f;
        } else {
            running_max = CUDART_INF_F;
            running_sum = 1.0f;
            alpha = 0.0f;
            beta = 1.0f;
        }
        return;
    }

    if (running_max == CUDART_INF_F) {
        alpha = 1.0f;
        beta = 0.0f;
        return;
    }

    const float next_max = fmaxf(running_max, score);
    alpha = expf(running_max - next_max);
    beta = expf(score - next_max);
    running_sum = running_sum * alpha + beta;
    running_max = next_max;
}

template <typename QType, typename KType, typename VType>
__global__ void fused_forward_kernel(FusedForwardParams p) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    const std::int64_t rows_per_batch = p.q_heads * p.q_tokens;
    const std::int64_t batch = row / rows_per_batch;
    const std::int64_t row_in_batch = row - batch * rows_per_batch;
    const std::int64_t q_head = row_in_batch / p.q_tokens;
    const std::int64_t q_position = row_in_batch - q_head * p.q_tokens;
    const std::int64_t k_head = q_head / (p.q_heads / p.k_heads);
    const std::int64_t v_head = q_head / (p.q_heads / p.v_heads);

    const std::int64_t k_batch = batch / (p.q_batches / p.k_batches);
    const std::int64_t v_batch = batch / (p.q_batches / p.v_batches);
    const char* q_row = p.q + batch * p.q_nb3 +
        q_head * p.q_nb2 + q_position * p.q_nb1;

    __shared__ float reduction[threads_per_block];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float running_max;
    __shared__ float running_sum;

    if (tid == 0) {
        running_max = -CUDART_INF_F;
        running_sum = 0.0f;
        alpha = 0.0f;
        beta = 0.0f;
    }
    __syncthreads();

    float output_accumulator[values_per_thread] = {0.0f, 0.0f};
    const float applied_scale = p.scale > 0.0f
        ? p.scale
        : rsqrtf(static_cast<float>(p.q_dim));

    for (std::int64_t key_position = 0; key_position < p.kv_tokens; ++key_position) {
        const char* k_row = p.k + k_batch * p.k_nb3 +
            k_head * p.k_nb2 + key_position * p.k_nb1;

        float partial = 0.0f;
        for (std::int64_t dimension = tid; dimension < p.q_dim;
             dimension += threads_per_block) {
            partial +=
                load_value<QType>(q_row + dimension * p.q_nb0) *
                load_value<KType>(k_row + dimension * p.k_nb0);
        }
        reduction[tid] = partial;
        __syncthreads();

        for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = reduction[0] * applied_scale;
            if (p.softcap > 0.0f && isfinite(score)) {
                score = p.softcap * tanhf(score / p.softcap);
            }

            if (p.causal && key_position > q_position) {
                score = -CUDART_INF_F;
            } else {
                if (p.mask != nullptr) {
                    const std::int64_t mask_batch =
                        p.mask_batch == 1 ? 0 : batch % p.mask_batch;
                    const std::int64_t mask_head =
                        p.mask_heads == 1 ? 0 : q_head % p.mask_heads;
                    const std::int64_t mask_query =
                        p.mask_q_tokens == 1 ? 0 : q_position;
                    const std::int64_t mask_key =
                        p.mask_k_tokens == 1 ? 0 : key_position;
                    const char* mask_pointer = p.mask +
                        mask_batch * p.mask_nb3 +
                        mask_head * p.mask_nb2 +
                        mask_query * p.mask_nb1 +
                        mask_key * p.mask_nb0;
                    float mask_value =
                        load_mask_value(mask_pointer, p.mask_dtype);
                    if (p.max_bias > 0.0f) {
                        mask_value *= ggml_alibi_slope(p, q_head);
                    }
                    score += mask_value;
                }
                if (p.explicit_alibi) {
                    score += p.alibi_slopes[q_head] *
                        static_cast<float>(key_position - q_position);
                }
            }
            online_softmax_update(
                score, running_max, running_sum, alpha, beta);
        }
        __syncthreads();

        const char* v_row = p.v + v_batch * p.v_nb3 +
            v_head * p.v_nb2 + key_position * p.v_nb1;
        #pragma unroll
        for (int slot = 0; slot < values_per_thread; ++slot) {
            const std::int64_t dimension = tid + slot * threads_per_block;
            if (dimension < p.v_dim) {
                output_accumulator[slot] =
                    output_accumulator[slot] * alpha +
                    beta * load_value<VType>(
                        v_row + dimension * p.v_nb0);
            }
        }
        __syncthreads();
    }

    if (p.sinks != nullptr) {
        if (tid == 0) {
            const float sink = reinterpret_cast<const float*>(p.sinks)[q_head];
            online_softmax_update(
                sink, running_max, running_sum, alpha, beta);
        }
        __syncthreads();
        #pragma unroll
        for (int slot = 0; slot < values_per_thread; ++slot) {
            output_accumulator[slot] *= alpha;
        }
    }

    char* out_row = p.out + batch * p.out_nb3 +
        q_head * p.out_nb2 + q_position * p.out_nb1;
    const float inverse_sum = running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    #pragma unroll
    for (int slot = 0; slot < values_per_thread; ++slot) {
        const std::int64_t dimension = tid + slot * threads_per_block;
        if (dimension < p.v_dim) {
            *reinterpret_cast<float*>(
                out_row + dimension * p.out_nb0) =
                output_accumulator[slot] * inverse_sum;
        }
    }
}

std::size_t element_size(DType dtype) noexcept {
    return dtype == DType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

std::array<std::size_t, 4> tensor_strides(
        const Tensor4D& tensor) noexcept {
    auto strides = tensor.byte_strides;
    if (strides[0] == 0) strides[0] = element_size(tensor.dtype);
    if (strides[1] == 0) {
        strides[1] = strides[0] * static_cast<std::size_t>(tensor.head_dim);
    }
    if (strides[2] == 0) {
        strides[2] = strides[1] * static_cast<std::size_t>(tensor.tokens);
    }
    if (strides[3] == 0) {
        strides[3] = strides[2] * static_cast<std::size_t>(tensor.heads);
    }
    return strides;
}

std::array<std::size_t, 4> tensor_strides(
        const MutableTensor4D& tensor) noexcept {
    auto strides = tensor.byte_strides;
    if (strides[0] == 0) strides[0] = element_size(tensor.dtype);
    if (strides[1] == 0) {
        strides[1] = strides[0] * static_cast<std::size_t>(tensor.head_dim);
    }
    if (strides[2] == 0) {
        strides[2] = strides[1] * static_cast<std::size_t>(tensor.tokens);
    }
    if (strides[3] == 0) {
        strides[3] = strides[2] * static_cast<std::size_t>(tensor.heads);
    }
    return strides;
}

std::array<std::size_t, 4> mask_strides(
        const MaskView& mask) noexcept {
    auto strides = mask.byte_strides;
    if (strides[0] == 0) strides[0] = element_size(mask.dtype);
    if (strides[1] == 0) {
        strides[1] = strides[0] * static_cast<std::size_t>(mask.key_tokens);
    }
    if (strides[2] == 0) {
        strides[2] = strides[1] * static_cast<std::size_t>(mask.query_tokens);
    }
    if (strides[3] == 0) {
        strides[3] = strides[2] * static_cast<std::size_t>(mask.heads);
    }
    return strides;
}

template <typename QType, typename KType, typename VType>
void launch(
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    fused_forward_kernel<QType, KType, VType>
        <<<rows, threads_per_block, 0, stream>>>(params);
}

template <typename QType, typename KType>
void launch_v(
        DType dtype,
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    switch (dtype) {
        case DType::f32: launch<QType, KType, float>(params, rows, stream); break;
        case DType::f16: launch<QType, KType, half>(params, rows, stream); break;
        case DType::bf16: launch<QType, KType, nv_bfloat16>(params, rows, stream); break;
    }
}

template <typename QType>
void launch_kv(
        DType k_dtype,
        DType v_dtype,
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    switch (k_dtype) {
        case DType::f32: launch_v<QType, float>(v_dtype, params, rows, stream); break;
        case DType::f16: launch_v<QType, half>(v_dtype, params, rows, stream); break;
        case DType::bf16: launch_v<QType, nv_bfloat16>(v_dtype, params, rows, stream); break;
    }
}

native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return native::NativeDType::f32;
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
    }
    return native::NativeDType::f32;
}

native::NativeRequest translate_request(const AttentionRequest& request) noexcept {
    native::NativeRequest result;
    result.q = {request.q.data, request.q.batch, request.q.heads,
                request.q.tokens, request.q.head_dim,
                translate_dtype(request.q.dtype), request.q.byte_strides};
    result.k = {request.k.data, request.k.batch, request.k.heads,
                request.k.tokens, request.k.head_dim,
                translate_dtype(request.k.dtype), request.k.byte_strides};
    result.v = {request.v.data, request.v.batch, request.v.heads,
                request.v.tokens, request.v.head_dim,
                translate_dtype(request.v.dtype), request.v.byte_strides};
    result.out = {request.out.data, request.out.batch, request.out.heads,
                  request.out.tokens, request.out.head_dim,
                  translate_dtype(request.out.dtype), request.out.byte_strides};
    result.dtype = translate_dtype(request.dtype);
    result.scale = request.scale;
    result.softcap = request.softcap;
    result.causal = request.causal;
    result.mask = {request.mask.data, request.mask.batch, request.mask.heads,
                   request.mask.query_tokens, request.mask.key_tokens,
                   translate_dtype(request.mask.dtype), request.mask.byte_strides};
    result.alibi = {request.alibi.enabled, request.alibi.slopes,
                    request.alibi.slope_count, request.alibi.max_bias};
    result.sinks = {request.sinks.enabled, request.sinks.values,
                    request.sinks.value_count};
    result.execution = {request.execution.stream, request.execution.synchronize,
                        request.execution.device_index};
    return result;
}

bool device_accessible_pointer(const void* pointer, int device) noexcept {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attributes.type == cudaMemoryTypeManaged ||
           (attributes.type == cudaMemoryTypeDevice && attributes.device == device);
#else
    return attributes.memoryType == cudaMemoryTypeDevice && attributes.device == device;
#endif
}

ValidationResult validate(const AttentionRequest& request) {
    const native::NativeRequest native_request = translate_request(request);
    const native::NativeValidationResult definition =
        native::validate_forward(native_request);
    if (!definition.ok) {
        return {false, definition.message == nullptr
            ? "CUDA xFormers definition rejected the request"
            : definition.message};
    }

    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess) {
        cudaGetLastError();
        return {false, "CUDA xFormers could not resolve the active CUDA device"};
    }
    const int device = request.execution.device_index < 0
        ? active_device
        : request.execution.device_index;
    if (device != active_device) {
        return {false, "CUDA xFormers execution device must be the active CUDA device"};
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        cudaGetLastError();
        return {false, "CUDA xFormers could not read active-device capabilities"};
    }
    if (properties.major < native::capabilities().minimum_compute_major) {
        return {false, "CUDA Common xFormers requires an NVIDIA Pascal-or-newer device"};
    }

    if (!device_accessible_pointer(request.q.data, device) ||
        !device_accessible_pointer(request.k.data, device) ||
        !device_accessible_pointer(request.v.data, device) ||
        !device_accessible_pointer(request.out.data, device)) {
        return {false, "CUDA xFormers Q, K, V, and output must be device or managed-memory pointers on the active device"};
    }
    if (request.mask.data != nullptr &&
        !device_accessible_pointer(request.mask.data, device)) {
        return {false, "CUDA xFormers additive mask must be device or managed memory"};
    }
    if (request.alibi.enabled &&
        request.alibi.slopes != nullptr &&
        !device_accessible_pointer(request.alibi.slopes, device)) {
        return {false, "CUDA xFormers ALiBi slopes must be device or managed memory"};
    }
    if (request.sinks.enabled &&
        !device_accessible_pointer(request.sinks.values, device)) {
        return {false, "CUDA xFormers attention sinks must be device or managed memory"};
    }
    return {true, {}};
}

bool forward(const AttentionRequest& request) {
    FusedForwardParams params;
    params.q = static_cast<const char*>(request.q.data);
    params.k = static_cast<const char*>(request.k.data);
    params.v = static_cast<const char*>(request.v.data);
    params.mask = static_cast<const char*>(request.mask.data);
    params.sinks = reinterpret_cast<const char*>(request.sinks.values);
    params.alibi_slopes = request.alibi.slopes;
    params.out = static_cast<char*>(request.out.data);
    params.q_batches = request.q.batch;
    params.k_batches = request.k.batch;
    params.v_batches = request.v.batch;
    params.q_heads = request.q.heads;
    params.k_heads = request.k.heads;
    params.v_heads = request.v.heads;
    params.q_tokens = request.q.tokens;
    params.kv_tokens = request.k.tokens;
    params.q_dim = request.q.head_dim;
    params.v_dim = request.v.head_dim;
    params.mask_batch = request.mask.batch;
    params.mask_heads = request.mask.heads;
    params.mask_q_tokens = request.mask.query_tokens;
    params.mask_k_tokens = request.mask.key_tokens;
    const auto q_strides = tensor_strides(request.q);
    const auto k_strides = tensor_strides(request.k);
    const auto v_strides = tensor_strides(request.v);
    const auto out_strides = tensor_strides(request.out);
    const auto normalized_mask_strides = mask_strides(request.mask);
    params.q_nb0 = q_strides[0];
    params.q_nb1 = q_strides[1];
    params.q_nb2 = q_strides[2];
    params.q_nb3 = q_strides[3];
    params.k_nb0 = k_strides[0];
    params.k_nb1 = k_strides[1];
    params.k_nb2 = k_strides[2];
    params.k_nb3 = k_strides[3];
    params.v_nb0 = v_strides[0];
    params.v_nb1 = v_strides[1];
    params.v_nb2 = v_strides[2];
    params.v_nb3 = v_strides[3];
    params.out_nb0 = out_strides[0];
    params.out_nb1 = out_strides[1];
    params.out_nb2 = out_strides[2];
    params.out_nb3 = out_strides[3];
    params.mask_nb0 = normalized_mask_strides[0];
    params.mask_nb1 = normalized_mask_strides[1];
    params.mask_nb2 = normalized_mask_strides[2];
    params.mask_nb3 = normalized_mask_strides[3];
    params.mask_dtype = static_cast<int>(request.mask.dtype);
    params.scale = request.scale;
    params.softcap = request.softcap;
    params.max_bias = request.alibi.max_bias;
    params.causal = request.causal;
    params.explicit_alibi =
        request.alibi.enabled && request.alibi.slopes != nullptr;
    params.head_log2 = 1u << static_cast<std::uint32_t>(
        std::floor(std::log2(static_cast<double>(request.q.heads))));
    params.alibi_m0 = std::pow(
        2.0f, -params.max_bias / static_cast<float>(params.head_log2));
    params.alibi_m1 = std::pow(
        2.0f,
        -(params.max_bias / 2.0f) / static_cast<float>(params.head_log2));

    const auto rows = static_cast<std::uint32_t>(
        request.q.batch * request.q.heads * request.q.tokens);
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(request.execution.stream);
    switch (request.q.dtype) {
        case DType::f32:
            launch_kv<float>(
                request.k.dtype, request.v.dtype, params, rows, stream);
            break;
        case DType::f16:
            launch_kv<half>(
                request.k.dtype, request.v.dtype, params, rows, stream);
            break;
        case DType::bf16:
            launch_kv<nv_bfloat16>(
                request.k.dtype, request.v.dtype, params, rows, stream);
            break;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    return !request.execution.synchronize ||
           cudaStreamSynchronize(stream) == cudaSuccess;
}

const Translation cuda_translation = [] {
    const native::NativeCapabilities native_capabilities = native::capabilities();
    Translation translation;
    translation.backend = Backend::cuda;
    translation.name = "cuda";
    translation.capabilities.forward = native_capabilities.fused_forward;
    translation.capabilities.qkt = native_capabilities.separate_stages;
    translation.capabilities.additive_mask = native_capabilities.additive_mask;
    translation.capabilities.causal_mask = native_capabilities.causal_mask;
    translation.capabilities.alibi = native_capabilities.alibi;
    translation.capabilities.softcap = native_capabilities.softcap;
    translation.capabilities.attention_sinks = native_capabilities.attention_sinks;
    translation.capabilities.gqa = native_capabilities.gqa;
    translation.capabilities.mqa = native_capabilities.mqa;
    translation.capabilities.f32 = native_capabilities.f32;
    translation.capabilities.f16 = native_capabilities.f16;
    translation.capabilities.bf16 = native_capabilities.bf16;
    translation.validate = &validate;
    translation.forward = &forward;
    return translation;
}();

[[maybe_unused]] const bool registered = register_translation(&cuda_translation);

} // namespace
} // namespace edcpp::api::attention::xformers::cuda::translation::gpu
