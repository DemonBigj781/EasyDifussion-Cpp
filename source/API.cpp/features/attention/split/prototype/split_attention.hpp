#pragma once

// Temporary Split Attention prototype contract.
// This is exploratory code only. It is intentionally outside `definition/`
// and `common/` and must not be treated as the final API.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace split_attention::prototype {

enum class SplitAxis {
    None,
    Query,
    KeyValue,
    Heads,
    Batch,
};

struct AttentionShape {
    std::int64_t batch = 0;
    std::int64_t query_tokens = 0;
    std::int64_t key_value_tokens = 0;
    std::int64_t query_heads = 0;
    std::int64_t key_value_heads = 0;
    std::int64_t head_dim = 0;
    std::size_t element_size = 0;
};

struct PlanRequest {
    AttentionShape shape{};
    std::size_t memory_budget_bytes = 0;
    std::int64_t preferred_chunk_multiple = 1;
    bool allow_unsplit = true;
    bool allow_query_split = true;
    bool allow_key_value_split = true;
    bool allow_head_split = true;
    bool allow_batch_split = true;
};

struct SplitAttentionPlan {
    SplitAxis axis = SplitAxis::None;
    std::int64_t chunk_size = 0;
    std::int64_t chunk_count = 1;
    std::size_t estimated_workspace_bytes = 0;
    bool uses_full_score_matrix = true;
    bool requires_cross_chunk_softmax = false;
    bool valid = false;
};

struct AttentionRequest {
    AttentionShape shape{};

    // Prototype storage is dense row-major float data:
    // Q: [batch, query_heads, query_tokens, head_dim]
    // K: [batch, kv_heads, kv_tokens, head_dim]
    // V: [batch, kv_heads, kv_tokens, head_dim]
    const float * q = nullptr;
    const float * k = nullptr;
    const float * v = nullptr;

    // Optional additive bias/mask with logical shape
    // [batch, query_heads, query_tokens, kv_tokens].
    // nullptr means no additive mask/bias.
    const float * additive_mask = nullptr;

    // Zero requests the conventional 1/sqrt(head_dim) scale.
    float scale = 0.0f;
};

struct AttentionResult {
    // [batch, query_heads, query_tokens, head_dim]
    std::vector<float> output;
};

SplitAttentionPlan make_plan(const PlanRequest & request);

// Backend-neutral CPU reference prototype used to establish Split Attention
// semantics. It supports unsplit, query-split, and exact K/V-split execution.
bool forward(const AttentionRequest & request,
             const SplitAttentionPlan & plan,
             AttentionResult & result);

} // namespace split_attention::prototype
