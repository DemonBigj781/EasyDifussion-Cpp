#ifndef SD_RUNTIME_TOKEN_MERGE_HPP
#define SD_RUNTIME_TOKEN_MERGE_HPP

#include "core/ggml_extend.hpp"

// Native, deterministic, windowed bipartite token merging. Matching and
// averaging stay in the GGML graph; no hidden-state download or CPU callback.
// Windows bound matching memory and keep argsort within CUDA's 1024-row limit.
class SpatialTokenMerge {
    struct Partition {
        std::vector<int32_t> source, destination;
        std::vector<float> identity, ones;
        explicit Partition(int64_t tokens) : ones(tokens, 1.f) {
            for (int32_t i = 0; i < tokens; ++i)
                (i % 4 == 0 ? destination : source).push_back(i);
            const size_t count = destination.size();
            identity.resize(count * count, 0.f);
            for (size_t i = 0; i < count; ++i) identity[i * count + i] = 1.f;
        }
    };
    std::map<int64_t, Partition> partitions;

    static ggml_tensor* input(GGMLRunnerContext* ctx, ggml_type type,
                              int64_t columns, int64_t rows, const void* data) {
        auto tensor = ggml_new_tensor_2d(ctx->ggml_ctx, type, columns, rows);
        ctx->bind_backend_tensor_data(tensor, data);
        return tensor;
    }
    static ggml_tensor* rows(ggml_context* ctx, ggml_tensor* values, ggml_tensor* ids) {
        return ggml_get_rows(ctx, values, ggml_reshape_1d(ctx, ids, ggml_nelements(ids)));
    }
    static ggml_tensor* append(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int axis) {
        return a == nullptr ? b : ggml_concat(ctx, a, b, axis);
    }

public:
    struct Window {
        int64_t frame, offset, length, kept, destinations, merged;
        ggml_tensor *original, *source_ids, *destination_ids, *keep_ids, *merge_ids, *nearest;
    };
    struct Result {
        ggml_tensor* values;
        std::vector<Window> windows;
    };

    Result merge(GGMLRunnerContext* ctx, ggml_tensor* metric, ggml_tensor* values, float ratio) {
        if (ratio == 0.f) return {values, {}};
        GGML_ASSERT(std::isfinite(ratio) && ratio > 0.f && ratio <= .5f);
        GGML_ASSERT(metric->ne[0] == values->ne[0] && metric->ne[1] == values->ne[1]);
        GGML_ASSERT(values->type == GGML_TYPE_F32 && values->ne[3] == 1);
        auto c = ctx->ggml_ctx;
        const int64_t channels = values->ne[0], tokens = values->ne[1], frames = values->ne[2];
        Result result{nullptr, {}};
        for (int64_t frame = 0; frame < frames; ++frame) {
            ggml_tensor* frame_values = nullptr;
            int64_t reduced_offset = 0;
            for (int64_t offset = 0; offset < tokens; offset += 1024) {
                const int64_t length = std::min<int64_t>(1024, tokens - offset);
                auto original = ggml_view_2d(c, values, channels, length, values->nb[1],
                                             frame * values->nb[2] + offset * values->nb[1]);
                auto local_metric = ggml_view_2d(c, metric, channels, length, metric->nb[1],
                                                 frame * metric->nb[2] + offset * metric->nb[1]);
                auto& partition = partitions.try_emplace(length, length).first->second;
                const int64_t source_count = partition.source.size(), destination_count = partition.destination.size();
                const int64_t merge_count = std::min<int64_t>(source_count, std::floor(length * ratio));
                if (merge_count == 0) {
                    result.windows.push_back({frame, reduced_offset, length, length, 0, 0, original,
                                              nullptr, nullptr, nullptr, nullptr, nullptr});
                    frame_values = append(c, frame_values, original, 1);
                    reduced_offset += length;
                    continue;
                }
                auto sources = ggml_reshape_1d(c, input(ctx, GGML_TYPE_I32, 1, source_count, partition.source.data()), source_count);
                auto destinations = ggml_reshape_1d(c, input(ctx, GGML_TYPE_I32, 1, destination_count, partition.destination.data()), destination_count);
                auto normalized = ggml_l2_norm(c, ggml_cont(c, local_metric), 1e-6f);
                auto scores = ggml_mul_mat(c, rows(c, normalized, destinations), rows(c, normalized, sources));
                ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                auto nearest = ggml_argmax(c, scores);
                auto best_scores = ggml_get_rows(c, ggml_reshape_3d(c, scores, 1, destination_count, source_count),
                                                  ggml_reshape_2d(c, nearest, 1, source_count));
                auto order = ggml_argsort(c, ggml_reshape_1d(c, best_scores, source_count), GGML_SORT_ORDER_DESC);
                auto merge_ids = ggml_view_1d(c, order, merge_count, 0);
                const int64_t kept = source_count - merge_count;
                auto keep_ids = ggml_view_1d(c, order, kept, merge_count * sizeof(int32_t));
                auto selected_destinations = ggml_reshape_1d(c, rows(c, ggml_reshape_2d(c, nearest, 1, source_count), merge_ids), merge_count);
                auto source_values = rows(c, original, sources);
                auto selected = rows(c, source_values, merge_ids);
                auto destination_values = rows(c, original, destinations);
                // One-hot assignment and matrix multiplication avoid the
                // channel-wise quadratic scan in GET_ROWS_BACK's CUDA kernel.
                auto ones = input(ctx, GGML_TYPE_F32, 1, destination_count, partition.ones.data());
                ggml_tensor *sum, *counts;
                if (sd_backend_is(ctx->backend, "Vulkan")) {
                    // Vulkan may convert F32 matmul inputs to F16. Use its
                    // F32 scatter accumulation for feature averages instead.
                    sum = ggml_get_rows_back(c, selected, selected_destinations, destination_values);
                    counts = ggml_get_rows_back(c, input(ctx, GGML_TYPE_F32, 1, merge_count, partition.ones.data()), selected_destinations, ones);
                } else {
                    auto identity = input(ctx, GGML_TYPE_F32, destination_count, destination_count, partition.identity.data());
                    auto weights = ggml_cont(c, ggml_transpose(c, rows(c, identity, selected_destinations)));
                    sum = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, selected)), weights);
                    ggml_mul_mat_set_prec(sum, GGML_PREC_F32);
                    counts = ggml_sum_rows(c, weights);
                }
                counts = ggml_add(c, counts, ones);
                auto averaged = ggml_div(c, ggml_add(c, destination_values, sum), counts);
                auto reduced = ggml_concat(c, rows(c, source_values, keep_ids), averaged, 1);
                result.windows.push_back({frame, reduced_offset, length, kept, destination_count, merge_count,
                                          original, sources, destinations, keep_ids, merge_ids, selected_destinations});
                frame_values = append(c, frame_values, reduced, 1);
                reduced_offset += length - merge_count;
            }
            result.values = append(c, result.values, frame_values, 2);
        }
        return result;
    }

    ggml_tensor* unmerge(GGMLRunnerContext* ctx, const Result& plan, ggml_tensor* attended) const {
        if (plan.windows.empty()) return attended;
        auto c = ctx->ggml_ctx;
        ggml_tensor *result = nullptr, *frame_values = nullptr;
        int64_t current_frame = 0;
        for (const auto& w : plan.windows) {
            if (w.frame != current_frame) {
                result = append(c, result, frame_values, 2);
                frame_values = nullptr;
                current_frame = w.frame;
            }
            const size_t offset = w.frame * attended->nb[2] + w.offset * attended->nb[1];
            auto kept = ggml_view_2d(c, attended, attended->ne[0], w.kept, attended->nb[1], offset);
            if (w.merged == 0) {
                frame_values = append(c, frame_values, kept, 1);
                continue;
            }
            auto destinations = ggml_view_2d(c, attended, attended->ne[0], w.destinations,
                                              attended->nb[1], offset + w.kept * attended->nb[1]);
            auto global_keep = rows(c, ggml_reshape_2d(c, w.source_ids, 1, ggml_nelements(w.source_ids)), w.keep_ids);
            auto global_merge = rows(c, ggml_reshape_2d(c, w.source_ids, 1, ggml_nelements(w.source_ids)), w.merge_ids);
            auto output = ggml_dup(c, w.original);
            output = ggml_set_rows(c, output, kept, ggml_reshape_1d(c, global_keep, w.kept));
            output = ggml_set_rows(c, output, destinations, w.destination_ids);
            output = ggml_set_rows(c, output, rows(c, destinations, w.nearest), ggml_reshape_1d(c, global_merge, w.merged));
            frame_values = append(c, frame_values, output, 1);
        }
        return append(c, result, frame_values, 2);
    }
};
#endif
