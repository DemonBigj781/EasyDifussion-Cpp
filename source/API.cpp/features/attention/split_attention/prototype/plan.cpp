// Temporary Split Attention planning prototype.
//
// This file is intentionally outside both `definition/` and `common/`.
// It exists only to explore a normalized planning model before backend
// translations are written. Do not treat this as a stable ABI or production
// implementation.

#include <algorithm>
#include <cstddef>
#include <cstdint>

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

    // Zero means "planner has no explicit memory budget".
    std::size_t memory_budget_bytes = 0;

    // Optional backend/device alignment hint. A backend definition may later
    // replace this with a target-specific value.
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

    // Approximate scratch requirement for the selected logical chunk.
    // This is deliberately backend-neutral and conservative.
    std::size_t estimated_workspace_bytes = 0;

    bool uses_full_score_matrix = true;
    bool valid = false;
};

static std::int64_t ceil_div(std::int64_t value, std::int64_t divisor) {
    return divisor > 0 ? (value + divisor - 1) / divisor : 0;
}

static std::int64_t align_down(std::int64_t value, std::int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return (value / multiple) * multiple;
}

static std::size_t estimate_score_bytes(const AttentionShape & shape,
                                        std::int64_t query_tokens,
                                        std::int64_t key_value_tokens) {
    if (query_tokens <= 0 || key_value_tokens <= 0 ||
        shape.batch <= 0 || shape.query_heads <= 0 || shape.element_size == 0) {
        return 0;
    }

    const std::uint64_t elements =
        static_cast<std::uint64_t>(shape.batch) *
        static_cast<std::uint64_t>(shape.query_heads) *
        static_cast<std::uint64_t>(query_tokens) *
        static_cast<std::uint64_t>(key_value_tokens);

    return static_cast<std::size_t>(elements * shape.element_size);
}

static SplitAttentionPlan make_plan(const PlanRequest & request) {
    SplitAttentionPlan plan{};
    const AttentionShape & shape = request.shape;

    if (shape.batch <= 0 || shape.query_tokens <= 0 ||
        shape.key_value_tokens <= 0 || shape.query_heads <= 0 ||
        shape.head_dim <= 0 || shape.element_size == 0) {
        return plan;
    }

    const std::size_t full_score_bytes =
        estimate_score_bytes(shape, shape.query_tokens, shape.key_value_tokens);

    // If there is no explicit budget, or the full score region fits, preserve
    // the unsplit path. A later backend definition may choose a fused path that
    // never materializes this matrix at all.
    if (request.allow_unsplit &&
        (request.memory_budget_bytes == 0 ||
         full_score_bytes <= request.memory_budget_bytes)) {
        plan.axis = SplitAxis::None;
        plan.chunk_size = shape.query_tokens;
        plan.chunk_count = 1;
        plan.estimated_workspace_bytes = full_score_bytes;
        plan.uses_full_score_matrix = true;
        plan.valid = true;
        return plan;
    }

    // Prototype policy: prefer splitting Q first because each query chunk can
    // preserve a complete K/V softmax domain without requiring a cross-chunk
    // normalization merge. This is a policy candidate, not a final rule.
    if (request.allow_query_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_query_token =
            estimate_score_bytes(shape, 1, shape.key_value_tokens);

        if (bytes_per_query_token > 0 &&
            bytes_per_query_token <= request.memory_budget_bytes) {
            std::int64_t chunk = static_cast<std::int64_t>(
                request.memory_budget_bytes / bytes_per_query_token);
            chunk = std::min(chunk, shape.query_tokens);
            chunk = align_down(chunk, std::max<std::int64_t>(
                                          1, request.preferred_chunk_multiple));
            chunk = std::max<std::int64_t>(1, chunk);

            plan.axis = SplitAxis::Query;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.query_tokens, chunk);
            plan.estimated_workspace_bytes =
                estimate_score_bytes(shape, chunk, shape.key_value_tokens);
            plan.uses_full_score_matrix = false;
            plan.valid = true;
            return plan;
        }
    }

    // Key/value splitting is intentionally only represented as a fallback in
    // this prototype. Exact attention then requires global softmax state across
    // K/V chunks (for example running max/sum and rescaling), which belongs in
    // later translated definitions and merge semantics.
    if (request.allow_key_value_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_kv_token =
            estimate_score_bytes(shape, shape.query_tokens, 1);

        if (bytes_per_kv_token > 0 &&
            bytes_per_kv_token <= request.memory_budget_bytes) {
            std::int64_t chunk = static_cast<std::int64_t>(
                request.memory_budget_bytes / bytes_per_kv_token);
            chunk = std::min(chunk, shape.key_value_tokens);
            chunk = align_down(chunk, std::max<std::int64_t>(
                                          1, request.preferred_chunk_multiple));
            chunk = std::max<std::int64_t>(1, chunk);

            plan.axis = SplitAxis::KeyValue;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.key_value_tokens, chunk);
            plan.estimated_workspace_bytes =
                estimate_score_bytes(shape, shape.query_tokens, chunk);
            plan.uses_full_score_matrix = false;
            plan.valid = true;
            return plan;
        }
    }

    // Head and batch splitting remain represented in the normalized vocabulary
    // but are not selected yet. Backend definitions should determine whether
    // they are useful and semantically safe for their target.
    return plan;
}

} // namespace split_attention::prototype
