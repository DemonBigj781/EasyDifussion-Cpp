#include "features/attention/flash/common/flash_attention.hpp"
#include "features/attention/flash/cuda/definition/gpu/flash_attention_cuda.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <array>
#include <cmath>
#include <cstdint>

// Static-library builds retain this translation object through this anchor so
// its initializer can register the CUDA callbacks with Common.
extern "C" void edcpp_flash_attention_cuda_translation_link_anchor() {}

namespace edcpp::api::attention::flash::cuda::translation::gpu {
namespace {

namespace native = edcpp::api::attention::flash::cuda::definition::gpu;

constexpr int threads_per_block = 256;
constexpr int values_per_thread = 2;

struct FusedForwardParams {
    const char* query = nullptr;
    const char* key = nullptr;
    const char* value = nullptr;
    const char* mask = nullptr;
    char* output = nullptr;

    std::int64_t batches = 0;
    std::int64_t query_heads = 0;
    std::int64_t key_heads = 0;
    std::int64_t value_heads = 0;
    std::int64_t query_tokens = 0;
    std::int64_t key_tokens = 0;
    std::int64_t query_dimension = 0;
    std::int64_t value_dimension = 0;

    std::int64_t mask_batches = 1;
    std::int64_t mask_heads = 1;

    std::size_t query_strides[4]{};
    std::size_t key_strides[4]{};
    std::size_t value_strides[4]{};
    std::size_t mask_strides[4]{};
    std::size_t output_strides[4]{};

    int mask_dtype = 0;
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    float alibi_m0 = 1.0f;
    float alibi_m1 = 1.0f;
    std::uint32_t head_log2 = 1;
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

__device__ __forceinline__ float alibi_slope(
        const FusedForwardParams& params, std::int64_t head) {
    return head < params.head_log2
        ? powf(params.alibi_m0, static_cast<float>(head + 1))
        : powf(
              params.alibi_m1,
              static_cast<float>(2 * (head - params.head_log2) + 1));
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

template <typename QueryType, typename KeyType, typename ValueType>
__global__ void fused_forward_kernel(FusedForwardParams params) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const int thread = threadIdx.x;
    const std::int64_t rows_per_batch =
        params.query_heads * params.query_tokens;
    const std::int64_t batch = row / rows_per_batch;
    const std::int64_t row_in_batch = row - batch * rows_per_batch;
    const std::int64_t query_head =
        row_in_batch / params.query_tokens;
    const std::int64_t query_position =
        row_in_batch - query_head * params.query_tokens;
    const std::int64_t key_head =
        query_head / (params.query_heads / params.key_heads);
    const std::int64_t value_head =
        query_head / (params.query_heads / params.value_heads);

    const char* query_row = params.query +
        batch * params.query_strides[3] +
        query_head * params.query_strides[2] +
        query_position * params.query_strides[1];

    __shared__ float reduction[threads_per_block];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float alpha;
    __shared__ float beta;

    if (thread == 0) {
        running_max = -CUDART_INF_F;
        running_sum = 0.0f;
        alpha = 0.0f;
        beta = 0.0f;
    }
    __syncthreads();

    float accumulator[values_per_thread] = {0.0f, 0.0f};
    for (std::int64_t key_position = 0;
         key_position < params.key_tokens;
         ++key_position) {
        const char* key_row = params.key +
            batch * params.key_strides[3] +
            key_head * params.key_strides[2] +
            key_position * params.key_strides[1];

        float partial = 0.0f;
        for (std::int64_t dimension = thread;
             dimension < params.query_dimension;
             dimension += threads_per_block) {
            partial +=
                load_value<QueryType>(
                    query_row + dimension * params.query_strides[0]) *
                load_value<KeyType>(
                    key_row + dimension * params.key_strides[0]);
        }
        reduction[thread] = partial;
        __syncthreads();

        for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                reduction[thread] += reduction[thread + stride];
            }
            __syncthreads();
        }

        if (thread == 0) {
            float score = reduction[0] * params.scale;
            if (params.logit_softcap > 0.0f && isfinite(score)) {
                score = params.logit_softcap *
                    tanhf(score / params.logit_softcap);
            }
            if (params.mask != nullptr) {
                const std::int64_t mask_batch =
                    params.mask_batches == 1
                    ? 0
                    : batch % params.mask_batches;
                const std::int64_t mask_head =
                    params.mask_heads == 1
                    ? 0
                    : query_head % params.mask_heads;
                const char* mask_pointer = params.mask +
                    mask_batch * params.mask_strides[3] +
                    mask_head * params.mask_strides[2] +
                    query_position * params.mask_strides[1] +
                    key_position * params.mask_strides[0];
                float mask_value =
                    load_mask_value(mask_pointer, params.mask_dtype);
                if (params.max_bias > 0.0f) {
                    mask_value *= alibi_slope(params, query_head);
                }
                score += mask_value;
            }
            online_softmax_update(
                score, running_max, running_sum, alpha, beta);
        }
        __syncthreads();

        const char* value_row = params.value +
            batch * params.value_strides[3] +
            value_head * params.value_strides[2] +
            key_position * params.value_strides[1];
        #pragma unroll
        for (int slot = 0; slot < values_per_thread; ++slot) {
            const std::int64_t dimension =
                thread + slot * threads_per_block;
            if (dimension < params.value_dimension) {
                accumulator[slot] = accumulator[slot] * alpha +
                    beta * load_value<ValueType>(
                        value_row + dimension * params.value_strides[0]);
            }
        }
        __syncthreads();
    }

    char* output_row = params.output +
        batch * params.output_strides[3] +
        query_head * params.output_strides[2] +
        query_position * params.output_strides[1];
    const float inverse_sum =
        running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    #pragma unroll
    for (int slot = 0; slot < values_per_thread; ++slot) {
        const std::int64_t dimension =
            thread + slot * threads_per_block;
        if (dimension < params.value_dimension) {
            *reinterpret_cast<float*>(
                output_row + dimension * params.output_strides[0]) =
                accumulator[slot] * inverse_sum;
        }
    }
}

std::size_t element_size(DType dtype) noexcept {
    return dtype == DType::f32 ? sizeof(float) : sizeof(std::uint16_t);
}

template <typename Tensor>
std::array<std::size_t, 4> normalized_strides(
        const Tensor& tensor) noexcept {
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

native::NativeDType translate_dtype(DType dtype) noexcept {
    switch (dtype) {
        case DType::f32: return native::NativeDType::f32;
        case DType::f16: return native::NativeDType::f16;
        case DType::bf16: return native::NativeDType::bf16;
    }
    return static_cast<native::NativeDType>(255);
}

native::NativeRequest translate_request(const Request& request) noexcept {
    native::NativeRequest result;
    result.query = {
        request.query.data, translate_dtype(request.query.dtype),
        request.query.head_dim, request.query.tokens, request.query.heads,
        request.query.batch, request.query.byte_strides};
    result.key = {
        request.key.data, translate_dtype(request.key.dtype),
        request.key.head_dim, request.key.tokens, request.key.heads,
        request.key.batch, request.key.byte_strides};
    result.value = {
        request.value.data, translate_dtype(request.value.dtype),
        request.value.head_dim, request.value.tokens, request.value.heads,
        request.value.batch, request.value.byte_strides};
    result.mask = {
        request.mask.data, translate_dtype(request.mask.dtype),
        request.mask.head_dim, request.mask.tokens, request.mask.heads,
        request.mask.batch, request.mask.byte_strides};
    result.output = {
        request.output.data, translate_dtype(request.output.dtype),
        request.output.head_dim, request.output.tokens, request.output.heads,
        request.output.batch, request.output.byte_strides};
    result.scale = request.scale;
    result.max_bias = request.max_bias;
    result.logit_softcap = request.logit_softcap;
    result.execution = {
        request.execution.stream, request.execution.synchronize,
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
           (attributes.type == cudaMemoryTypeDevice &&
            attributes.device == device);
#else
    return attributes.memoryType == cudaMemoryTypeDevice &&
           attributes.device == device;
#endif
}

Result validate_cuda(const Request& request) noexcept {
    const native::NativeResult definition =
        native::validate_forward(translate_request(request));
    if (!definition.ok) {
        return {false, definition.message == nullptr
            ? "CUDA FlashAttention definition rejected the request"
            : definition.message};
    }
    if (request.execution.thread_index != 0 ||
        request.execution.thread_count != 1) {
        return {false, "CUDA FlashAttention requires one host dispatch thread"};
    }

    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess) {
        cudaGetLastError();
        return {false, "CUDA FlashAttention could not resolve the active CUDA device"};
    }
    const int device = request.execution.device_index < 0
        ? active_device
        : request.execution.device_index;
    if (device != active_device) {
        return {false, "CUDA FlashAttention execution device must be the active CUDA device"};
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        cudaGetLastError();
        return {false, "CUDA FlashAttention could not read active-device capabilities"};
    }
    if (properties.major < native::capabilities().minimum_compute_major) {
        return {false, "CUDA Common FlashAttention requires an NVIDIA Pascal-or-newer device"};
    }

    if (!device_accessible_pointer(request.query.data, device) ||
        !device_accessible_pointer(request.key.data, device) ||
        !device_accessible_pointer(request.value.data, device) ||
        !device_accessible_pointer(request.output.data, device)) {
        return {false, "CUDA FlashAttention Q, K, V, and output must be device or managed-memory pointers on the active device"};
    }
    if (request.mask.data != nullptr &&
        !device_accessible_pointer(request.mask.data, device)) {
        return {false, "CUDA FlashAttention mask must be device or managed memory"};
    }
    return {true, nullptr};
}

template <typename QueryType, typename KeyType, typename ValueType>
void launch(
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    fused_forward_kernel<QueryType, KeyType, ValueType>
        <<<rows, threads_per_block, 0, stream>>>(params);
}

template <typename QueryType, typename KeyType>
void launch_value(
        DType value_dtype,
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    switch (value_dtype) {
        case DType::f32:
            launch<QueryType, KeyType, float>(params, rows, stream);
            break;
        case DType::f16:
            launch<QueryType, KeyType, half>(params, rows, stream);
            break;
        case DType::bf16:
            launch<QueryType, KeyType, nv_bfloat16>(params, rows, stream);
            break;
    }
}

template <typename QueryType>
void launch_key_value(
        DType key_dtype,
        DType value_dtype,
        const FusedForwardParams& params,
        std::uint32_t rows,
        cudaStream_t stream) {
    switch (key_dtype) {
        case DType::f32:
            launch_value<QueryType, float>(
                value_dtype, params, rows, stream);
            break;
        case DType::f16:
            launch_value<QueryType, half>(
                value_dtype, params, rows, stream);
            break;
        case DType::bf16:
            launch_value<QueryType, nv_bfloat16>(
                value_dtype, params, rows, stream);
            break;
    }
}

Result forward_cuda(const Request& request) noexcept {
    FusedForwardParams params;
    params.query = static_cast<const char*>(request.query.data);
    params.key = static_cast<const char*>(request.key.data);
    params.value = static_cast<const char*>(request.value.data);
    params.mask = static_cast<const char*>(request.mask.data);
    params.output = static_cast<char*>(request.output.data);
    params.batches = request.query.batch;
    params.query_heads = request.query.heads;
    params.key_heads = request.key.heads;
    params.value_heads = request.value.heads;
    params.query_tokens = request.query.tokens;
    params.key_tokens = request.key.tokens;
    params.query_dimension = request.query.head_dim;
    params.value_dimension = request.value.head_dim;
    params.mask_batches = request.mask.batch;
    params.mask_heads = request.mask.heads;

    const auto query_strides = normalized_strides(request.query);
    const auto key_strides = normalized_strides(request.key);
    const auto value_strides = normalized_strides(request.value);
    const auto mask_strides = normalized_strides(request.mask);
    const auto output_strides = normalized_strides(request.output);
    for (std::size_t axis = 0; axis < 4; ++axis) {
        params.query_strides[axis] = query_strides[axis];
        params.key_strides[axis] = key_strides[axis];
        params.value_strides[axis] = value_strides[axis];
        params.mask_strides[axis] = mask_strides[axis];
        params.output_strides[axis] = output_strides[axis];
    }
    params.mask_dtype = static_cast<int>(request.mask.dtype);
    params.scale = request.scale;
    params.max_bias = request.max_bias;
    params.logit_softcap = request.logit_softcap;
    params.head_log2 = 1u << static_cast<std::uint32_t>(
        std::floor(std::log2(static_cast<double>(request.query.heads))));
    params.alibi_m0 = std::pow(
        2.0f, -params.max_bias / static_cast<float>(params.head_log2));
    params.alibi_m1 = std::pow(
        2.0f,
        -(params.max_bias / 2.0f) / static_cast<float>(params.head_log2));

    const auto rows = static_cast<std::uint32_t>(
        request.query.batch * request.query.heads * request.query.tokens);
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(request.execution.stream);
    switch (request.query.dtype) {
        case DType::f32:
            launch_key_value<float>(
                request.key.dtype, request.value.dtype,
                params, rows, stream);
            break;
        case DType::f16:
            launch_key_value<half>(
                request.key.dtype, request.value.dtype,
                params, rows, stream);
            break;
        case DType::bf16:
            launch_key_value<nv_bfloat16>(
                request.key.dtype, request.value.dtype,
                params, rows, stream);
            break;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return {false, "CUDA FlashAttention kernel launch failed"};
    }
    if (request.execution.synchronize &&
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return {false, "CUDA FlashAttention kernel execution failed"};
    }
    return {true, nullptr};
}

const Translation cuda_translation = [] {
    const native::NativeCapabilities native_capabilities =
        native::capabilities();
    Translation translation;
    translation.backend = Backend::cuda;
    translation.name = "cuda";
    translation.capabilities.forward = native_capabilities.fused_forward;
    translation.capabilities.additive_mask =
        native_capabilities.additive_mask;
    translation.capabilities.alibi_bias = native_capabilities.alibi_bias;
    translation.capabilities.logit_softcap =
        native_capabilities.logit_softcap;
    translation.capabilities.grouped_query =
        native_capabilities.grouped_query;
    translation.capabilities.f32_accumulation =
        native_capabilities.f32_accumulation;
    translation.validate = &validate_cuda;
    translation.forward = &forward_cuda;
    return translation;
}();

[[maybe_unused]] const bool registered =
    register_translation(&cuda_translation);

} // namespace
} // namespace edcpp::api::attention::flash::cuda::translation::gpu
