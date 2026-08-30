#ifndef __SD_RUNTIME_TEACACHE_HPP__
#define __SD_RUNTIME_TEACACHE_HPP__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.hpp"
#include "runtime/condition_cache_utils.hpp"

// Native condition-boundary implementation of the TeaCache policy described
// by ali-vilab/TeaCache (Apache-2.0). It keeps the full-model residual in host
// memory and uses the official architecture-specific polynomial to accumulate
// relative L1 changes between diffusion inputs. Keeping the probe outside the
// transformer avoids a second partial graph and works with streamed weights.
struct TeaCacheConfig {
    bool enabled          = false;
    float reuse_threshold = 0.2f;
    float start_percent   = 0.0f;
    float end_percent     = 1.0f;
    int total_steps       = 0;
    std::vector<double> coefficients;
    std::string model_variant;
};

struct TeaCacheEntry {
    std::vector<float> previous_input;
    std::vector<float> residual;
    double accumulated_rel_l1 = 0.0;
};

struct TeaCacheState {
    TeaCacheConfig config;
    std::unordered_map<const void*, TeaCacheEntry> entries;
    bool initialized       = false;
    bool step_active       = false;
    bool skip_current_step = false;
    int current_step_index = -1;
    int cache_hits         = 0;
    int model_evaluations  = 0;

    void init(const TeaCacheConfig& cfg) {
        config             = cfg;
        initialized        = cfg.enabled && cfg.reuse_threshold > 0.0f && cfg.total_steps > 1;
        step_active        = false;
        skip_current_step  = false;
        current_step_index = -1;
        cache_hits         = 0;
        model_evaluations  = 0;
        entries.clear();
    }

    bool enabled() const {
        return initialized && config.enabled;
    }

    void begin_step(int step_index, float /*sigma*/) {
        if (!enabled() || step_index == current_step_index) {
            return;
        }
        current_step_index = step_index;
        skip_current_step  = false;
        const float progress = config.total_steps > 1
                                   ? static_cast<float>(step_index) / static_cast<float>(config.total_steps - 1)
                                   : 0.0f;
        step_active = progress >= config.start_percent && progress <= config.end_percent;
    }

    bool is_step_skipped() const {
        return enabled() && step_active && skip_current_step;
    }

    double rescale(double relative_l1) const {
        if (config.coefficients.empty()) {
            return relative_l1;
        }
        double value = 0.0;
        for (double coefficient : config.coefficients) {
            value = value * relative_l1 + coefficient;
        }
        return std::isfinite(value) ? std::max(0.0, value) : relative_l1;
    }

    bool before_condition(const void* condition,
                          const sd::Tensor<float>& input,
                          sd::Tensor<float>* output,
                          float sigma,
                          int step_index) {
        if (!enabled() || condition == nullptr || output == nullptr || step_index < 0) {
            return false;
        }
        if (step_index != current_step_index) {
            begin_step(step_index, sigma);
        }

        TeaCacheEntry& entry = entries[condition];
        const size_t count   = static_cast<size_t>(input.numel());
        const float* data    = input.data();
        bool should_compute  = !step_active || step_index == 0 ||
                              step_index == config.total_steps - 1 ||
                              entry.previous_input.size() != count || entry.residual.empty();

        if (!should_compute) {
            double absolute_change = 0.0;
            double previous_norm   = 0.0;
            for (size_t i = 0; i < count; ++i) {
                absolute_change += std::fabs(static_cast<double>(data[i]) - entry.previous_input[i]);
                previous_norm += std::fabs(static_cast<double>(entry.previous_input[i]));
            }
            const double relative_l1 = previous_norm > 0.0 ? absolute_change / previous_norm
                                                           : (absolute_change == 0.0 ? 0.0 : INFINITY);
            entry.accumulated_rel_l1 += rescale(relative_l1);
            should_compute = !std::isfinite(entry.accumulated_rel_l1) ||
                             entry.accumulated_rel_l1 >= config.reuse_threshold;
            if (should_compute) {
                entry.accumulated_rel_l1 = 0.0;
            }
        } else if (!step_active || step_index == 0 || step_index == config.total_steps - 1) {
            entry.accumulated_rel_l1 = 0.0;
        }

        entry.previous_input.assign(data, data + count);
        if (should_compute) {
            ++model_evaluations;
            return false;
        }

        sd::apply_condition_cache_diff(entry.residual, input, output);
        skip_current_step = true;
        ++cache_hits;
        return true;
    }

    void after_condition(const void* condition,
                         const sd::Tensor<float>& input,
                         const sd::Tensor<float>& output) {
        if (!enabled() || condition == nullptr) {
            return;
        }
        TeaCacheEntry& entry = entries[condition];
        sd::store_condition_cache_diff(&entry.residual, input, output);
    }
};

#endif  // __SD_RUNTIME_TEACACHE_HPP__
