#include "xformers-attention.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("xFormers attention is not available for this target");
}

#else

namespace {

constexpr int XFORMERS_THREADS = 256;
constexpr int XFORMERS_VALUES_PER_THREAD = 2;
constexpr int XFORMERS_MAX_VALUE_DIM = XFORMERS_THREADS * XFORMERS_VALUES_PER_THREAD;

struct xformers_forward_params {
    const char * q;
    const char * k;
    const char * v;
    const char * mask;
    const char * sinks;
    char * out;

    int64_t q_dim;
    int64_t v_dim;
    int64_t q_len;
    int64_t kv_len;
    int64_t q_heads;
    int64_t k_heads;
    int64_t v_heads;
    int64_t q_batches;
    int64_t k_batches;
    int64_t v_batches;
    int64_t mask_heads;
    int64_t mask_batches;

    uint64_t q_nb1;
    uint64_t q_nb2;
    uint64_t q_nb3;
    uint64_t k_nb1;
    uint64_t k_nb2;
    uint64_t k_nb3;
    uint64_t v_nb1;
    uint64_t v_nb2;
    uint64_t v_nb3;
    uint64_t mask_nb0;
    uint64_t mask_nb1;
    uint64_t mask_nb2;
    uint64_t mask_nb3;
    uint64_t out_nb0;
    uint64_t out_nb1;
    uint64_t out_nb2;
    uint64_t out_nb3;

    float scale;
    float max_bias;
    float logit_softcap;
    float alibi_m0;
    float alibi_m1;
    uint32_t n_head_log2;
};

template <typename T>
__device__ __forceinline__ float xformers_load(const char * ptr, int64_t index) {
    return static_cast<float>(reinterpret_cast<const T *>(ptr)[index]);
}

template <>
__device__ __forceinline__ float xformers_load<half>(const char * ptr, int64_t index) {
    return __half2float(reinterpret_cast<const half *>(ptr)[index]);
}

template <>
__device__ __forceinline__ float xformers_load<nv_bfloat16>(const char * ptr, int64_t index) {
    const uint32_t bits = static_cast<uint32_t>(reinterpret_cast<const uint16_t *>(ptr)[index]) << 16;
    return __uint_as_float(bits);
}

// Update one online-softmax row. alpha rescales the previous numerator and
// beta is the weight for the newly observed value. Explicit infinity handling
// keeps fully masked rows and multiple +inf logits deterministic.
__device__ __forceinline__ void xformers_online_softmax_update(
        float score,
        float & running_max,
        float & running_sum,
        float & alpha,
        float & beta) {
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
__global__ void xformers_mea_forward_kernel(xformers_forward_params p) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;

    const int64_t rows_per_batch = p.q_heads * p.q_len;
    const int64_t batch = row / rows_per_batch;
    const int64_t row_in_batch = row - batch * rows_per_batch;
    const int64_t q_head = row_in_batch / p.q_len;
    const int64_t q_pos = row_in_batch - q_head * p.q_len;

    const int64_t k_head = q_head / (p.q_heads / p.k_heads);
    const int64_t v_head = q_head / (p.q_heads / p.v_heads);
    const int64_t k_batch = batch / (p.q_batches / p.k_batches);
    const int64_t v_batch = batch / (p.q_batches / p.v_batches);

    const char * q_row = p.q + batch * p.q_nb3 + q_head * p.q_nb2 + q_pos * p.q_nb1;

    __shared__ float reduction[XFORMERS_THREADS];
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

    float out_acc[XFORMERS_VALUES_PER_THREAD] = {0.0f, 0.0f};
    const float slope = get_alibi_slope(
        p.max_bias, static_cast<uint32_t>(q_head), p.n_head_log2, p.alibi_m0, p.alibi_m1);
    const float dot_scale = p.logit_softcap == 0.0f ? p.scale : p.scale / p.logit_softcap;

    for (int64_t key_pos = 0; key_pos < p.kv_len; ++key_pos) {
        const char * k_row = p.k + k_batch * p.k_nb3 + k_head * p.k_nb2 + key_pos * p.k_nb1;

        float partial = 0.0f;
        for (int64_t d = tid; d < p.q_dim; d += XFORMERS_THREADS) {
            partial += xformers_load<QType>(q_row, d) * xformers_load<KType>(k_row, d);
        }
        reduction[tid] = partial;
        __syncthreads();

        for (int stride = XFORMERS_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float score = reduction[0] * dot_scale;
            if (p.logit_softcap != 0.0f) {
                score = p.logit_softcap * tanhf(score);
            }

            if (p.mask != nullptr) {
                const int64_t mask_head = q_head % p.mask_heads;
                const int64_t mask_batch = batch % p.mask_batches;
                const char * mask_value = p.mask +
                    mask_batch * p.mask_nb3 + mask_head * p.mask_nb2 +
                    q_pos * p.mask_nb1 + key_pos * p.mask_nb0;
                score += slope * __half2float(*reinterpret_cast<const half *>(mask_value));
            }

            xformers_online_softmax_update(score, running_max, running_sum, alpha, beta);
        }
        __syncthreads();

        const char * v_row = p.v + v_batch * p.v_nb3 + v_head * p.v_nb2 + key_pos * p.v_nb1;
        #pragma unroll
        for (int slot = 0; slot < XFORMERS_VALUES_PER_THREAD; ++slot) {
            const int64_t d = tid + slot * XFORMERS_THREADS;
            if (d < p.v_dim) {
                out_acc[slot] = out_acc[slot] * alpha + beta * xformers_load<VType>(v_row, d);
            }
        }
        __syncthreads();
    }

    // Attention sinks participate in the softmax denominator but have no V
    // row, so they only rescale the accumulated numerator.
    if (p.sinks != nullptr) {
        if (tid == 0) {
            const float sink = reinterpret_cast<const float *>(p.sinks)[q_head];
            xformers_online_softmax_update(sink, running_max, running_sum, alpha, beta);
        }
        __syncthreads();

        #pragma unroll
        for (int slot = 0; slot < XFORMERS_VALUES_PER_THREAD; ++slot) {
            out_acc[slot] *= alpha;
        }
    }

    char * out_row = p.out + batch * p.out_nb3 + q_pos * p.out_nb2 + q_head * p.out_nb1;
    const float inv_sum = running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    #pragma unroll
    for (int slot = 0; slot < XFORMERS_VALUES_PER_THREAD; ++slot) {
        const int64_t d = tid + slot * XFORMERS_THREADS;
        if (d < p.v_dim) {
            *reinterpret_cast<float *>(out_row + d * p.out_nb0) = out_acc[slot] * inv_sum;
        }
    }
}

static bool xformers_supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static size_t xformers_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return sizeof(float);
        case GGML_TYPE_F16: return sizeof(half);
        case GGML_TYPE_BF16: return sizeof(nv_bfloat16);
        default: return 0;
    }
}

static bool xformers_valid_row_layout(const ggml_tensor * tensor) {
    return tensor != nullptr && xformers_supported_type(tensor->type) &&
           tensor->nb[0] == xformers_type_size(tensor->type);
}

static bool xformers_read_params(
        const ggml_tensor * dst,
        float & scale,
        float & max_bias,
        float & logit_softcap) {
    std::memcpy(&scale, dst->op_params + 0 * sizeof(float), sizeof(scale));
    std::memcpy(&max_bias, dst->op_params + 1 * sizeof(float), sizeof(max_bias));
    std::memcpy(&logit_softcap, dst->op_params + 2 * sizeof(float), sizeof(logit_softcap));

    int32_t precision = GGML_PREC_DEFAULT;
    std::memcpy(&precision, dst->op_params + 3 * sizeof(float), sizeof(precision));

    return std::isfinite(scale) && std::isfinite(max_bias) &&
           std::isfinite(logit_softcap) && logit_softcap >= 0.0f &&
           (precision == GGML_PREC_DEFAULT || precision == GGML_PREC_F32);
}

template <typename QType, typename KType, typename VType>
static void xformers_launch(
        const xformers_forward_params & params,
        cudaStream_t stream,
        uint32_t rows) {
    const dim3 grid(rows, 1, 1);
    const dim3 block(XFORMERS_THREADS, 1, 1);
    xformers_mea_forward_kernel<QType, KType, VType><<<grid, block, 0, stream>>>(params);
}

template <typename QType, typename KType>
static void xformers_launch_v(
        ggml_type v_type,
        const xformers_forward_params & params,
        cudaStream_t stream,
        uint32_t rows) {
    switch (v_type) {
        case GGML_TYPE_F32: xformers_launch<QType, KType, float>(params, stream, rows); break;
        case GGML_TYPE_F16: xformers_launch<QType, KType, half>(params, stream, rows); break;
        case GGML_TYPE_BF16: xformers_launch<QType, KType, nv_bfloat16>(params, stream, rows); break;
        default: GGML_ABORT("unsupported xFormers V type");
    }
}

template <typename QType>
static void xformers_launch_kv(
        ggml_type k_type,
        ggml_type v_type,
        const xformers_forward_params & params,
        cudaStream_t stream,
        uint32_t rows) {
    switch (k_type) {
        case GGML_TYPE_F32: xformers_launch_v<QType, float>(v_type, params, stream, rows); break;
        case GGML_TYPE_F16: xformers_launch_v<QType, half>(v_type, params, stream, rows); break;
        case GGML_TYPE_BF16: xformers_launch_v<QType, nv_bfloat16>(v_type, params, stream, rows); break;
        default: GGML_ABORT("unsupported xFormers K type");
    }
}

} // namespace

bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT ||
        device < 0 || device >= ggml_cuda_info().device_count) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL ||
        !xformers_valid_row_layout(q) || !xformers_valid_row_layout(k) ||
        !xformers_valid_row_layout(v) || dst->type != GGML_TYPE_F32 ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }

    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || q->ne[3] <= 0 ||
        k->ne[0] != q->ne[0] || k->ne[1] <= 0 || k->ne[2] <= 0 || k->ne[3] <= 0 ||
        v->ne[0] <= 0 || v->ne[0] > XFORMERS_MAX_VALUE_DIM ||
        v->ne[1] != k->ne[1] || v->ne[2] <= 0 || v->ne[3] <= 0) {
        return false;
    }

    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 ||
        q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) {
        return false;
    }

    if (dst->ne[0] != v->ne[0] || dst->ne[1] != q->ne[2] ||
        dst->ne[2] != q->ne[1] || dst->ne[3] != q->ne[3]) {
        return false;
    }

    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) ||
            mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] ||
            mask->ne[2] <= 0 || mask->ne[3] <= 0 ||
            q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0) {
            return false;
        }
    }

    if (sinks != nullptr &&
        (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float) || sinks->ne[0] != q->ne[2])) {
        return false;
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!xformers_read_params(dst, scale, max_bias, logit_softcap) ||
        (max_bias > 0.0f && mask == nullptr)) {
        return false;
    }

    const uint64_t rows = static_cast<uint64_t>(q->ne[1]) * q->ne[2] * q->ne[3];
    return rows <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

void ggml_cuda_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_xformers_attn_supported(ctx.device, dst));

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    xformers_forward_params params{};
    params.q = static_cast<const char *>(q->data);
    params.k = static_cast<const char *>(k->data);
    params.v = static_cast<const char *>(v->data);
    params.mask = mask == nullptr ? nullptr : static_cast<const char *>(mask->data);
    params.sinks = sinks == nullptr ? nullptr : static_cast<const char *>(sinks->data);
    params.out = static_cast<char *>(dst->data);

    params.q_dim = q->ne[0];
    params.v_dim = v->ne[0];
    params.q_len = q->ne[1];
    params.kv_len = k->ne[1];
    params.q_heads = q->ne[2];
    params.k_heads = k->ne[2];
    params.v_heads = v->ne[2];
    params.q_batches = q->ne[3];
    params.k_batches = k->ne[3];
    params.v_batches = v->ne[3];
    params.mask_heads = mask == nullptr ? 1 : mask->ne[2];
    params.mask_batches = mask == nullptr ? 1 : mask->ne[3];

    params.q_nb1 = q->nb[1];
    params.q_nb2 = q->nb[2];
    params.q_nb3 = q->nb[3];
    params.k_nb1 = k->nb[1];
    params.k_nb2 = k->nb[2];
    params.k_nb3 = k->nb[3];
    params.v_nb1 = v->nb[1];
    params.v_nb2 = v->nb[2];
    params.v_nb3 = v->nb[3];
    params.mask_nb0 = mask == nullptr ? 0 : mask->nb[0];
    params.mask_nb1 = mask == nullptr ? 0 : mask->nb[1];
    params.mask_nb2 = mask == nullptr ? 0 : mask->nb[2];
    params.mask_nb3 = mask == nullptr ? 0 : mask->nb[3];
    params.out_nb0 = dst->nb[0];
    params.out_nb1 = dst->nb[1];
    params.out_nb2 = dst->nb[2];
    params.out_nb3 = dst->nb[3];

    xformers_read_params(dst, params.scale, params.max_bias, params.logit_softcap);
    params.n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(q->ne[2])));
    params.alibi_m0 = std::pow(2.0f, -params.max_bias / params.n_head_log2);
    params.alibi_m1 = std::pow(2.0f, -(params.max_bias / 2.0f) / params.n_head_log2);

    const uint32_t rows = static_cast<uint32_t>(q->ne[1] * q->ne[2] * q->ne[3]);
    cudaStream_t stream = ctx.stream();

    switch (q->type) {
        case GGML_TYPE_F32: xformers_launch_kv<float>(k->type, v->type, params, stream, rows); break;
        case GGML_TYPE_F16: xformers_launch_kv<half>(k->type, v->type, params, stream, rows); break;
        case GGML_TYPE_BF16: xformers_launch_kv<nv_bfloat16>(k->type, v->type, params, stream, rows); break;
        default: GGML_ABORT("unsupported xFormers Q type");
    }

    CUDA_CHECK(cudaGetLastError());
}

#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
