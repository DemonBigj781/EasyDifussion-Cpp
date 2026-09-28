#include "optimizer.hpp"

static void opt_step_adamw_f32_sycl(float* weight, const float* grad, float* moment1,
                                   float* moment2, const float* params, int64_t n,
                                   queue_ptr stream) {
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const int64_t i = id[0];
            const float alpha = params[0];
            const float beta1 = params[1];
            const float beta2 = params[2];
            const float eps = params[3];
            const float wd = params[4];
            const float beta1h = params[5];
            const float beta2h = params[6];
            const float m = moment1[i] * beta1 + grad[i] * (1.0f - beta1);
            const float v = moment2[i] * beta2 + grad[i] * grad[i] * (1.0f - beta2);
            moment1[i] = m;
            moment2[i] = v;
            const float mh = m * beta1h;
            const float vh = sycl::sqrt(v * beta2h) + eps;
            weight[i] = weight[i] * (1.0f - alpha * wd) - alpha * mh / vh;
        });
    });
}

static void opt_step_sgd_f32_sycl(float* weight, const float* grad, const float* params,
                                 int64_t n, queue_ptr stream) {
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            const int64_t i = id[0];
            const float alpha = params[0];
            const float keep = 1.0f - alpha * params[1];
            weight[i] = weight[i] * keep - alpha * grad[i];
        });
    });
}

void ggml_sycl_op_opt_step_adamw(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32 &&
                dst->src[1]->type == GGML_TYPE_F32 && dst->src[2]->type == GGML_TYPE_F32 &&
                dst->src[3]->type == GGML_TYPE_F32 && dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst->src[2]) &&
                ggml_is_contiguous(dst->src[3]) && ggml_is_contiguous(dst->src[4]));
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    opt_step_adamw_f32_sycl(static_cast<float*>(dst->src[0]->data),
                            static_cast<const float*>(dst->src[1]->data),
                            static_cast<float*>(dst->src[2]->data),
                            static_cast<float*>(dst->src[3]->data),
                            static_cast<const float*>(dst->src[4]->data),
                            ggml_nelements(dst), ctx.stream());
}

void ggml_sycl_op_opt_step_sgd(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32 &&
                dst->src[1]->type == GGML_TYPE_F32 && dst->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst->src[2]));
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    opt_step_sgd_f32_sycl(static_cast<float*>(dst->src[0]->data),
                          static_cast<const float*>(dst->src[1]->data),
                          static_cast<const float*>(dst->src[2]->data),
                          ggml_nelements(dst), ctx.stream());
}
