#include "features/attention/sage/common/sage_attention.hpp"
#include "features/attention/sage/cuda/translation/gpu/support.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace sage = edcpp::api::attention::sage;

namespace {

bool check(bool condition, std::string_view message) {
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    return condition;
}

sage::SupportRequest supported_request() {
    sage::SupportRequest request;
    request.query = {nullptr, sage::DType::f32, 64, 77, 8, 1, {}};
    request.key = {nullptr, sage::DType::f16, 64, 77, 8, 1, {}};
    request.value = {nullptr, sage::DType::f16, 64, 77, 8, 1, {}};
    request.output = {
        nullptr,
        sage::DType::f32,
        64,
        77,
        8,
        1,
        {sizeof(float),
         64 * 8 * sizeof(float),
         64 * sizeof(float),
         64 * 8 * 77 * sizeof(float)}};
    request.device = {0, 8, 6};
    request.scale = 0.125f;
    return request;
}

} // namespace

int main() {
    using edcpp::api::Backend;

    bool ok = true;
    ok &= check(
        edcpp::api::attention::sage::cuda::translation::gpu::ensure_registered(),
        "CUDA Sage support registration anchor is retained");
    const auto* translation = sage::translation_for(Backend::cuda);
    ok &= check(translation != nullptr,
                "CUDA Sage support translation is registered with Common");

    const sage::Capabilities capabilities = sage::capabilities(Backend::cuda);
    ok &= check(capabilities.support && !capabilities.forward &&
                    capabilities.int8_qk && capabilities.fp16_pv &&
                    capabilities.grouped_query &&
                    capabilities.minimum_compute_major == 8 &&
                    capabilities.maximum_compute_major_exclusive == 9,
                "capabilities describe the current SM80 INT8-QK/FP16-PV boundary");

    auto request = supported_request();
    ok &= check(sage::support(Backend::cuda, request).ok,
                "normalized support accepts an SM86-compatible request");

    request.device.compute_major = 9;
    ok &= check(!sage::support(Backend::cuda, request).ok,
                "normalized support rejects Hopper until a matching kernel exists");

    request = supported_request();
    request.additive_mask = true;
    ok &= check(!sage::support(Backend::cuda, request).ok,
                "normalized support rejects masks not implemented by the live kernel");

    request = supported_request();
    request.key.dtype = sage::DType::bf16;
    ok &= check(!sage::support(Backend::cuda, request).ok,
                "normalized support rejects unsupported K/V dtypes");

    request = supported_request();
    request.output.head_dim = 128;
    ok &= check(!sage::support(Backend::cuda, request).ok,
                "Common rejects a mismatched output shape");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
