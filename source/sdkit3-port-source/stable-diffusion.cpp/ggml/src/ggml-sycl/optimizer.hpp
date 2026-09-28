#ifndef GGML_SYCL_OPTIMIZER_HPP
#define GGML_SYCL_OPTIMIZER_HPP

#include "common.hpp"

void ggml_sycl_op_opt_step_adamw(ggml_backend_sycl_context& ctx, ggml_tensor* dst);
void ggml_sycl_op_opt_step_sgd(ggml_backend_sycl_context& ctx, ggml_tensor* dst);

#endif // GGML_SYCL_OPTIMIZER_HPP
