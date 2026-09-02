// Temporary backend-neutral Split Attention execution prototype.
//
// This is a CPU reference implementation used to establish semantics for the
// later backend definition translations. It is intentionally not production
// code and is not part of `common/`.

#include "split_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace split_attention::prototype {

namespace {

static std::size_t q_index(const AttentionShape & s,
                           std::int64_t b,
                           std::int64_t h,
                           std::int64_t q,
                           std::int64_t d) {
    return static_cast<std::size_t>((((b * s.query_heads + h) * s.query_tokens + q) * s.head_dim) + d);
}

static std::size_t kv_index(const AttentionShape & s,
                            std::int64_t b,
                            std::int64_t h,
                            std::int64_t kv,
                            std::int64_t d) {
    return static_cast<std::size_t>((((b * s.key_value_heads + h) * s.key_value_tokens + kv) * s.head_dim) + d);
}

static std::size_t mask_index(const AttentionShape & s,
                              std::int64_t b,
                              std::int64_t h,
                              std::int64_t q,
                              std::int64_t kv) {
    return static_cast<std::size_t>((((b * s.query_heads + h) * s.query_tokens + q) * s.key_value_tokens) + kv);
}

static float attention_scale(const AttentionRequest & request) {
    if (request.scale != 0.0f) {
        return request.scale;
    }
    return 1.0f / std::sqrt(static_cast<float>(request.shape.head_dim));
}

static float score(const AttentionRequest & request,
                   std::int64_t b,
                   std::int64_t qh,
                   std::int64_t kvh,
                   std::int64_t q,
                   std::int64_t kv,
                   float scale) {
    const AttentionShape & s = request.shape;
    float dot = 0.0f;
    for (std::int64_t d = 0; d < s.head_dim; ++d) {
        dot += request.q[q_index(s, b, qh, q, d)] *
               request.k[kv_index(s, b, kvh, kv, d)];
    }
    dot *= scale;
    if (request.additive_mask != nullptr) {
        dot += request.additive_mask[mask_index(s, b, qh, q, kv)];
    }
    return dot;
}

static bool execute_complete_kv_domain(const AttentionRequest & request,
                                       AttentionResult & result,
                                       std::int64_t batch_begin,
                                       std::int64_t batch_end,
                                       std::int64_t head_begin,
                                       std::int64_t head_end,
                                       std::int64_t query_begin,
                                       std::int64_t query_end) {
    const AttentionShape & s = request.shape;
    const float scale = attention_scale(request);
    const std::int64_t q_per_kv = s.query_heads / s.key_value_heads;

    std::vector<float> logits(static_cast<std::size_t>(s.key_value_tokens));

    for (std::int64_t b = batch_begin; b < batch_end; ++b) {
        for (std::int64_t qh = head_begin; qh < head_end; ++qh) {
            const std::int64_t kvh = qh / q_per_kv;
            for (std::int64_t q = query_begin; q < query_end; ++q) {
                float row_max = -std::numeric_limits<float>::infinity();
                for (std::int64_t kv = 0; kv < s.key_value_tokens; ++kv) {
                    const float x = score(request, b, qh, kvh, q, kv, scale);
                    logits[static_cast<std::size_t>(kv)] = x;
                    row_max = std::max(row_max, x);
                }

                float denom = 0.0f;
                for (std::int64_t kv = 0; kv < s.key_value_tokens; ++kv) {
                    const float e = std::exp(logits[static_cast<std::size_t>(kv)] - row_max);
                    logits[static_cast<std::size_t>(kv)] = e;
                    denom += e;
                }

                if (!(denom > 0.0f) || !std::isfinite(denom)) {
                    return false;
                }

                for (std::int64_t d = 0; d < s.head_dim; ++d) {
                    float acc = 0.0f;
                    for (std::int64_t kv = 0; kv < s.key_value_tokens; ++kv) {
                        const float p = logits[static_cast<std::size_t>(kv)] / denom;
                        acc += p * request.v[kv_index(s, b, kvh, kv, d)];
                    }
                    result.output[q_index(s, b, qh, q, d)] = acc;
                }
            }
        }
    }

    return true;
}

// Exact K/V-split attention using an online softmax merge. For each query row:
//   m = running maximum logit
//   l = running exp-sum relative to m
//   o = running weighted V sum relative to m
// Each K/V chunk can then be merged without ever materializing the complete
// score row.
static bool execute_kv_split(const AttentionRequest & request,
                             const SplitAttentionPlan & plan,
                             AttentionResult & result) {
    const AttentionShape & s = request.shape;
    const float scale = attention_scale(request);
    const std::int64_t q_per_kv = s.query_heads / s.key_value_heads;

    for (std::int64_t b = 0; b < s.batch; ++b) {
        for (std::int64_t qh = 0; qh < s.query_heads; ++qh) {
            const std::int64_t kvh = qh / q_per_kv;
            for (std::int64_t q = 0; q < s.query_tokens; ++q) {
                float running_max = -std::numeric_limits<float>::infinity();
                float running_sum = 0.0f;
                std::vector<float> running_value(static_cast<std::size_t>(s.head_dim), 0.0f);

                for (std::int64_t start = 0; start < s.key_value_tokens; start += plan.chunk_size) {
                    const std::int64_t end = std::min(start + plan.chunk_size, s.key_value_tokens);

                    float chunk_max = -std::numeric_limits<float>::infinity();
                    std::vector<float> chunk_logits(static_cast<std::size_t>(end - start));

                    for (std::int64_t kv = start; kv < end; ++kv) {
                        const float x = score(request, b, qh, kvh, q, kv, scale);
                        chunk_logits[static_cast<std::size_t>(kv - start)] = x;
                        chunk_max = std::max(chunk_max, x);
                    }

                    float chunk_sum = 0.0f;
                    std::vector<float> chunk_value(static_cast<std::size_t>(s.head_dim), 0.0f);

                    for (std::int64_t kv = start; kv < end; ++kv) {
                        const float e = std::exp(
                            chunk_logits[static_cast<std::size_t>(kv - start)] - chunk_max);
                        chunk_sum += e;
                        for (std::int64_t d = 0; d < s.head_dim; ++d) {
                            chunk_value[static_cast<std::size_t>(d)] +=
                                e * request.v[kv_index(s, b, kvh, kv, d)];
                        }
                    }

                    if (!(chunk_sum > 0.0f) || !std::isfinite(chunk_sum)) {
                        return false;
                    }

                    if (running_sum == 0.0f) {
                        running_max = chunk_max;
                        running_sum = chunk_sum;
                        running_value = std::move(chunk_value);
                        continue;
                    }

                    const float merged_max = std::max(running_max, chunk_max);
                    const float old_scale = std::exp(running_max - merged_max);
                    const float new_scale = std::exp(chunk_max - merged_max);

                    running_sum = running_sum * old_scale + chunk_sum * new_scale;
                    for (std::int64_t d = 0; d < s.head_dim; ++d) {
                        running_value[static_cast<std::size_t>(d)] =
                            running_value[static_cast<std::size_t>(d)] * old_scale +
                            chunk_value[static_cast<std::size_t>(d)] * new_scale;
                    }
                    running_max = merged_max;
                }

                if (!(running_sum > 0.0f) || !std::isfinite(running_sum)) {
                    return false;
                }

                for (std::int64_t d = 0; d < s.head_dim; ++d) {
                    result.output[q_index(s, b, qh, q, d)] =
                        running_value[static_cast<std::size_t>(d)] / running_sum;
                }
            }
        }
    }

    return true;
}

} // namespace

bool forward(const AttentionRequest & request,
             const SplitAttentionPlan & plan,
             AttentionResult & result) {
    const AttentionShape & s = request.shape;

    if (!plan.valid || request.q == nullptr || request.k == nullptr ||
        request.v == nullptr || s.batch <= 0 || s.query_tokens <= 0 ||
        s.key_value_tokens <= 0 || s.query_heads <= 0 ||
        s.key_value_heads <= 0 || s.head_dim <= 0 ||
        s.query_heads % s.key_value_heads != 0 || plan.chunk_size <= 0) {
        return false;
    }

    const std::size_t output_elements =
        static_cast<std::size_t>(s.batch) *
        static_cast<std::size_t>(s.query_heads) *
        static_cast<std::size_t>(s.query_tokens) *
        static_cast<std::size_t>(s.head_dim);
    result.output.assign(output_elements, 0.0f);

    switch (plan.axis) {
        case SplitAxis::None:
            return execute_complete_kv_domain(
                request, result,
                0, s.batch,
                0, s.query_heads,
                0, s.query_tokens);

        case SplitAxis::Query:
            for (std::int64_t start = 0; start < s.query_tokens; start += plan.chunk_size) {
                const std::int64_t end = std::min(start + plan.chunk_size, s.query_tokens);
                if (!execute_complete_kv_domain(
                        request, result,
                        0, s.batch,
                        0, s.query_heads,
                        start, end)) {
                    return false;
                }
            }
            return true;

        case SplitAxis::KeyValue:
            return execute_kv_split(request, plan, result);

        case SplitAxis::Heads:
            for (std::int64_t start = 0; start < s.query_heads; start += plan.chunk_size) {
                const std::int64_t end = std::min(start + plan.chunk_size, s.query_heads);
                if (!execute_complete_kv_domain(
                        request, result,
                        0, s.batch,
                        start, end,
                        0, s.query_tokens)) {
                    return false;
                }
            }
            return true;

        case SplitAxis::Batch:
            for (std::int64_t start = 0; start < s.batch; start += plan.chunk_size) {
                const std::int64_t end = std::min(start + plan.chunk_size, s.batch);
                if (!execute_complete_kv_domain(
                        request, result,
                        start, end,
                        0, s.query_heads,
                        0, s.query_tokens)) {
                    return false;
                }
            }
            return true;
    }

    return false;
}

} // namespace split_attention::prototype
