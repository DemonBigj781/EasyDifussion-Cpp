#include "xformers-attention.cuh"

#include <cfloat>
#include <cmath>
#include <cstring>

namespace {

constexpr int XFORMERS_THREADS = 256;

template <typename T>
__device__ __forceinline__ float xformers_load(const T * ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float xformers_load<half>(const half * ptr) {
    return __half2float(*ptr);
}

template <typename QType>
__global__ void xformers_mea_forward_kernel(
        const QType * __restrict__ q,
        const half * __restrict__ k,
        const half * __restrict__ v,
        const half * __restrict__ mask,
        float * __restrict__ out,
        int d,
        int q_len,
        int kv_len,
        int q_heads,
        int kv_heads,
        int batches,
        int64_t q_s1,
        int64_t q_s2,
        int64_t q_s3,
        int64_t k_s1,
        int64_t k_s2,
        int64_t k_s3,
        int64_t v_s1,
        int64_t v_s2,
        int64_t v_s3,
        int64_t o_s1,
        int64_t o_s2,
        int64_t o_s3,
        int mask_heads,
        int mask_batches,
        int64_t m_s1,
        int64_t m_s2,
        int64_t m_s3,
        float scale) {
    const int q_pos = blockIdx.x;
    const int q_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;

    if (q_pos >= q_len || q_head >= q_heads || batch >= batches) {
        return;
    }

    const int gqa = q_heads / kv_heads;
    const int kv_head = q_head / gqa;

    const QType * q_row = q + static_cast<int64_t>(batch) * q_s3 +
                          static_cast<int64_t>(q_head) * q_s2 +
                          static_cast<int64_t>(q_pos) * q_s1;

    __shared__ float reduction[XFORMERS_THREADS];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float running_max;
    __shared__ float running_sum;

    if (tid == 0) {
        running_max = -CUDART_INF_F;
        running_sum = 0.0f;
    }
    __syncthreads();

    float out_acc = 0.0f;

    for (int key_pos = 0; key_pos < kv_len; ++key_pos) {
        const half * k_row = k + static_cast<int64_t>(batch) * k_s3 +
                             static_cast<int64_t>(kv_head) * k_s2 +
                             static_cast<int64_t>(key_pos) * k_s1;

        float partial = 0.0f;
        if (tid < d) {
            partial = xformers_load(q_row + tid) * __half2float(k_row[tid]);
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
            float score = reduction[0] * scale;
            if (mask != nullptr) {
                const int mh = mask_heads == 1 ? 0 : (q_head % mask_heads);
                const int mb = mask_batches == 1 ? 0 : batch;
                const half * mp = mask + static_cast<int64_t>(mb) * m_s3 +
                                  static_cast<int64_t>(mh) * m_s2 +
                                  static_cast<int64_t>(q_pos) * m_s1 + key_pos;
                score += __half2float(*mp);
            }

            const float next_max = fmaxf(running_max, score);
            alpha = isinf(running_max) ? 0.0f : expf(running_max - next_max);
            beta = expf(score - next_max);
            running_sum = running_sum * alpha + beta;
            running_max = next_max;
        }
        __syncthreads();

        if (tid < d) {
            const half * v_row = v + static_cast<int64_t>(batch) * v_s3 +
                                 static_cast<int64_t>(kv_head) * v_s2 +
                                 static_cast<int64_t>(key_pos) * v_s1;
            out_acc = out_acc * alpha + beta * __half2float(v_row[tid]);
        }
        __syncthreads();
    }

    if (tid < d) {
        float * out_row = out + static_cast<int64_t>(batch) * o_s3 +
                          static_cast<int64_t>(q_head) * o_s2 +
                          static_cast<int64_t>(q_pos) * o_s1;
        out_row[tid] = running_sum > 0.0f ? out_acc / running_sum : 0.0f;
    }
}

static bool xformers_valid_tensor_layout(const ggml_tensor * t, size_t element_size) {
    return t != nullptr && t->nb[0] == element_size;
}

} // namespace

bool ggml_cuda_xformers_attn_supported(int device, const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr || sinks != nullptr) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_PASCAL) {
        return false;
    }

    if ((Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) ||
        K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t d = Q->ne[0];
    if (d <= 0 || d > XFORMERS_THREADS || K->ne[0] != d || V->ne[0] != d) {
        return false;
    }

    if (Q->ne[1] <= 0 || K->ne[1] <= 0 || Q->ne[2] <= 0 || K->ne[2] <= 0 || Q->ne[3] <= 0) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3] || Q->ne[3] != V->ne[3]) {
        return false;
    }
    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2]) {
        return false;
    }

    if (!xformers_valid_tensor_layout(Q, Q->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(half)) ||
        !xformers_valid_tensor_layout(K, sizeof(half)) ||
        !xformers_valid_tensor_layout(V, sizeof(half)) ||
        !xformers_valid_tensor_layout(dst, sizeof(float))) {
        return false;
    }

    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) ||
            mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] ||
            (mask->ne[2] != 1 && mask->ne[2] != Q->ne[2]) ||
            (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3])) {
            return false;
        }
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, reinterpret_cast<const float *>(dst->op_params) + 2, sizeof(logit_softcap));
    return max_bias == 0.0f && logit_softcap == 0.0f;
}

void ggml_cuda_xformers_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_xformers_attn_supported(ctx.device, dst));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, reinterpret_cast<const float *>(dst->op_params), sizeof(scale));

    const int d = static_cast<int>(Q->ne[0]);
    const int q_len = static_cast<int>(Q->ne[1]);
    const int kv_len = static_cast<int>(K->ne[1]);
    const int q_heads = static_cast<int>(Q->ne[2]);
    const int kv_heads = static_cast<int>(K->ne[2]);
    const int batches = static_cast<int>(Q->ne[3]);

    const dim3 grid(q_len, q_heads, batches);
    const dim3 block(XFORMERS_THREADS, 1, 1);
    cudaStream_t stream = ctx.stream();

    const int mask_heads = mask == nullptr ? 1 : static_cast<int>(mask->ne[2]);
    const int mask_batches = mask == nullptr ? 1 : static_cast<int>(mask->ne[3]);
    const int64_t m_s1 = mask == nullptr ? 0 : static_cast<int64_t>(mask->nb[1] / sizeof(half));
    const int64_t m_s2 = mask == nullptr ? 0 : static_cast<int64_t>(mask->nb[2] / sizeof(half));
    const int64_t m_s3 = mask == nullptr ? 0 : static_cast<int64_t>(mask->nb[3] / sizeof(half));

    if (Q->type == GGML_TYPE_F32) {
        xformers_mea_forward_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float *>(Q->data),
            static_cast<const half *>(K->data),
            static_cast<const half *>(V->data),
            mask == nullptr ? nullptr : static_cast<const half *>(mask->data),
            static_cast<float *>(dst->data),
            d, q_len, kv_len, q_heads, kv_heads, batches,
            Q->nb[1] / sizeof(float), Q->nb[2] / sizeof(float), Q->nb[3] / sizeof(float),
            K->nb[1] / sizeof(half), K->nb[2] / sizeof(half), K->nb[3] / sizeof(half),
            V->nb[1] / sizeof(half), V->nb[2] / sizeof(half), V->nb[3] / sizeof(half),
            dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), dst->nb[3] / sizeof(float),
            mask_heads, mask_batches, m_s1, m_s2, m_s3, scale);
    } else {
        xformers_mea_forward_kernel<half><<<grid, block, 0, stream>>>(
            static_cast<const half *>(Q->data),
            static_cast<const half *>(K->data),
            static_cast<const half *>(V->data),
            mask == nullptr ? nullptr : static_cast<const half *>(mask->data),
            static_cast<float *>(dst->data),
            d, q_len, kv_len, q_heads, kv_heads, batches,
            Q->nb[1] / sizeof(half), Q->nb[2] / sizeof(half), Q->nb[3] / sizeof(half),
            K->nb[1] / sizeof(half), K->nb[2] / sizeof(half), K->nb[3] / sizeof(half),
            V->nb[1] / sizeof(half), V->nb[2] / sizeof(half), V->nb[3] / sizeof(half),
            dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), dst->nb[3] / sizeof(float),
            mask_heads, mask_batches, m_s1, m_s2, m_s3, scale);
    }

    CUDA_CHECK(cudaGetLastError());
}
