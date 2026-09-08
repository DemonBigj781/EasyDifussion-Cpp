// Temporary Split Attention planning prototype.
//
// This file is intentionally outside both `definition/` and `common/`.
// It exists only to explore a normalized planning model before backend
// translations are written. Do not treat this as a stable ABI or production
// implementation.

#include "split_attention.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace split_attention::prototype {

static std::int64_t ceil_div(std::int64_t value, std::int64_t divisor) {
    return divisor > 0 ? (value + divisor - 1) / divisor : 0;
}

static std::int64_t align_down(std::int64_t value, std::int64_t multiple) {
    if (multiple <= 1) {
        return value;
    }
    return (value / multiple) * multiple;
}

static std::size_t saturating_mul(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        return std::numeric_limits<std::size_t>::max();
    }
    return a * b;
}

static std::size_t estimate_score_bytes(const AttentionShape & shape,
                                        std::int64_t batch,
                                        std::int64_t heads,
                                        std::int64_t query_tokens,
                                        std::int64_t key_value_tokens) {
    if (batch <= 0 || heads <= 0 || query_tokens <= 0 ||
        key_value_tokens <= 0 || shape.element_size == 0) {
        return 0;
    }

    std::size_t bytes = static_cast<std::size_t>(batch);
    bytes = saturating_mul(bytes, static_cast<std::size_t>(heads));
    bytes = saturating_mul(bytes, static_cast<std::size_t>(query_tokens));
    bytes = saturating_mul(bytes, static_cast<std::size_t>(key_value_tokens));
    bytes = saturating_mul(bytes, shape.element_size);
    return bytes;
}

static std::int64_t budget_limited_chunk(std::size_t budget,
                                         std::size_t bytes_per_item,
                                         std::int64_t logical_items,
                                         std::int64_t preferred_multiple) {
    if (budget == 0 || bytes_per_item == 0 || logical_items <= 0 ||
        bytes_per_item > budget) {
        return 0;
    }

    std::int64_t chunk = static_cast<std::int64_t>(budget / bytes_per_item);
    chunk = std::min(chunk, logical_items);
    chunk = align_down(chunk, std::max<std::int64_t>(1, preferred_multiple));
    return std::max<std::int64_t>(1, chunk);
}

SplitAttentionPlan make_plan(const PlanRequest & request) {
    SplitAttentionPlan plan{};
    const AttentionShape & shape = request.shape;

    if (shape.batch <= 0 || shape.query_tokens <= 0 ||
        shape.key_value_tokens <= 0 || shape.query_heads <= 0 ||
        shape.key_value_heads <= 0 || shape.head_dim <= 0 ||
        shape.element_size == 0) {
        return plan;
    }

    // Prototype GQA/MQA rule: query heads must map evenly to K/V heads.
    if (shape.query_heads % shape.key_value_heads != 0) {
        return plan;
    }

    const std::size_t full_score_bytes = estimate_score_bytes(
        shape,
        shape.batch,
        shape.query_heads,
        shape.query_tokens,
        shape.key_value_tokens);

    if (request.allow_unsplit &&
        (request.memory_budget_bytes == 0 ||
         full_score_bytes <= request.memory_budget_bytes)) {
        plan.axis = SplitAxis::None;
        plan.chunk_size = shape.query_tokens;
        plan.chunk_count = 1;
        plan.estimated_workspace_bytes = full_score_bytes;
        plan.uses_full_score_matrix = true;
        plan.requires_cross_chunk_softmax = false;
        plan.valid = true;
        return plan;
    }

    // Prefer query splitting. Each query chunk still spans the complete K/V
    // domain, so softmax remains local to each query row and no cross-chunk
    // normalization state is required.
    if (request.allow_query_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_query_token = estimate_score_bytes(
            shape,
            shape.batch,
            shape.query_heads,
            1,
            shape.key_value_tokens);

        const std::int64_t chunk = budget_limited_chunk(
            request.memory_budget_bytes,
            bytes_per_query_token,
            shape.query_tokens,
            request.preferred_chunk_multiple);

        if (chunk > 0) {
            plan.axis = SplitAxis::Query;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.query_tokens, chunk);
            plan.estimated_workspace_bytes = estimate_score_bytes(
                shape,
                shape.batch,
                shape.query_heads,
                chunk,
                shape.key_value_tokens);
            plan.uses_full_score_matrix = false;
            plan.requires_cross_chunk_softmax = false;
            plan.valid = true;
            return plan;
        }
    }

    // K/V splitting can operate at a lower memory ceiling than query splitting,
    // but exact attention requires a running global-softmax state across K/V
    // chunks. The prototype forward path implements that state explicitly.
    if (request.allow_key_value_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_kv_token = estimate_score_bytes(
            shape,
            shape.batch,
            shape.query_heads,
            shape.query_tokens,
            1);

        const std::int64_t chunk = budget_limited_chunk(
            request.memory_budget_bytes,
            bytes_per_kv_token,
            shape.key_value_tokens,
            request.preferred_chunk_multiple);

        if (chunk > 0) {
            plan.axis = SplitAxis::KeyValue;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.key_value_tokens, chunk);
            plan.estimated_workspace_bytes = estimate_score_bytes(
                shape,
                shape.batch,
                shape.query_heads,
                shape.query_tokens,
                chunk);
            plan.uses_full_score_matrix = false;
            plan.requires_cross_chunk_softmax = true;
            plan.valid = true;
            return plan;
        }
    }

    // Prototype support for head and batch splitting. These preserve complete
    // K/V domains for every query row and therefore do not require a special
    // softmax merge. They are considered only after query and K/V splitting.
    if (request.allow_head_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_head = estimate_score_bytes(
            shape,
            shape.batch,
            1,
            shape.query_tokens,
            shape.key_value_tokens);

        const std::int64_t chunk = budget_limited_chunk(
            request.memory_budget_bytes,
            bytes_per_head,
            shape.query_heads,
            1);

        if (chunk > 0) {
            plan.axis = SplitAxis::Heads;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.query_heads, chunk);
            plan.estimated_workspace_bytes = estimate_score_bytes(
                shape,
                shape.batch,
                chunk,
                shape.query_tokens,
                shape.key_value_tokens);
            plan.uses_full_score_matrix = false;
            plan.requires_cross_chunk_softmax = false;
            plan.valid = true;
            return plan;
        }
    }

    if (request.allow_batch_split && request.memory_budget_bytes > 0) {
        const std::size_t bytes_per_batch = estimate_score_bytes(
            shape,
            1,
            shape.query_heads,
            shape.query_tokens,
            shape.key_value_tokens);

        const std::int64_t chunk = budget_limited_chunk(
            request.memory_budget_bytes,
            bytes_per_batch,
            shape.batch,
            1);

        if (chunk > 0) {
            plan.axis = SplitAxis::Batch;
            plan.chunk_size = chunk;
            plan.chunk_count = ceil_div(shape.batch, chunk);
            plan.estimated_workspace_bytes = estimate_score_bytes(
                shape,
                chunk,
                shape.query_heads,
                shape.query_tokens,
                shape.key_value_tokens);
            plan.uses_full_score_matrix = false;
            plan.requires_cross_chunk_softmax = false;
            plan.valid = true;
            return plan;
        }
    }

    return plan;
}

} // namespace split_attention::prototype
