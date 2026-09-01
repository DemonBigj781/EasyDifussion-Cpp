#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xformers::prototype {

enum class DType {
    F32,
    F16,
    BF16,
};

enum class BackendFamily {
    CPU,
    CUDA,
    ROCM,
    ONEAPI,
    VULKAN,
};

enum class DeviceClass {
    CPU,
    GPU,
    NPU,
};

enum class CapabilityState {
    NotCompiled,
    Unavailable,
    Unvalidated,
    Validated,
    Unsupported,
};

struct Tensor4D {
    // Logical layout: [batch, heads, tokens, head_dim].
    const float * data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
};

struct MutableTensor4D {
    float * data = nullptr;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t tokens = 0;
    std::int64_t head_dim = 0;
};

struct MaskView {
    // Optional additive mask. Logical layout may be broadcast from
    // [batch|1, heads|1, query_tokens|1, key_tokens|1].
    const float * data = nullptr;
    std::int64_t batch = 1;
    std::int64_t heads = 1;
    std::int64_t query_tokens = 1;
    std::int64_t key_tokens = 1;
};

struct AlibiConfig {
    bool enabled = false;
    // One slope per query head when non-empty.
    const float * slopes = nullptr;
    std::int64_t slope_count = 0;
};

struct AttentionSinkConfig {
    bool enabled = false;
    // Optional sink bias per query head. This prototype models sinks as an
    // additive logit term; backend definitions may translate native semantics.
    const float * values = nullptr;
    std::int64_t value_count = 0;
};

struct AttentionRequest {
    Tensor4D q{};
    Tensor4D k{};
    Tensor4D v{};
    MutableTensor4D out{};

    DType dtype = DType::F32;
    float scale = 0.0f;              // 0 => 1/sqrt(head_dim)
    float softcap = 0.0f;            // <=0 => disabled
    bool causal = false;

    MaskView mask{};
    AlibiConfig alibi{};
    AttentionSinkConfig sinks{};
};

struct BackendIdentity {
    BackendFamily family = BackendFamily::CPU;
    DeviceClass device_class = DeviceClass::CPU;
    CapabilityState state = CapabilityState::Unvalidated;
    const char * name = "prototype-cpu";
};

struct Capabilities {
    bool forward = true;
    bool qkt = true;
    bool additive_mask = true;
    bool causal_mask = true;
    bool alibi = true;
    bool softcap = true;
    bool attention_sinks = true;
    bool gqa = true;
    bool mqa = true;
    bool f32 = true;
    bool f16 = false;
    bool bf16 = false;
};

struct ValidationResult {
    bool ok = false;
    std::string message;
};

struct ScoreBuffer {
    std::vector<float> values;
    std::int64_t batch = 0;
    std::int64_t heads = 0;
    std::int64_t query_tokens = 0;
    std::int64_t key_tokens = 0;
};

BackendIdentity backend_identity();
Capabilities capabilities();
ValidationResult validate(const AttentionRequest & request);

bool qkt(const AttentionRequest & request, ScoreBuffer & scores);
bool apply_mask_and_bias(const AttentionRequest & request, ScoreBuffer & scores);
bool softmax(ScoreBuffer & scores);
bool av(const AttentionRequest & request, const ScoreBuffer & probabilities);
bool forward(const AttentionRequest & request);

} // namespace xformers::prototype
