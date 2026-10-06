#ifndef SDKIT_SD15_EPOCH_SAMPLING_HPP
#define SDKIT_SD15_EPOCH_SAMPLING_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

struct SD15EpochSchedule {
    int steps_per_epoch;
    int steps;
    SD15EpochSchedule(size_t images, int repeats, int epochs, int requested_steps) {
        if (images < 1 || images > 100000 || repeats < 1 || repeats > 100000 ||
            epochs < 0 || epochs > 100000 || requested_steps < 1 || requested_steps > 100000)
            throw std::invalid_argument("invalid epoch schedule");
        const auto length = static_cast<uint64_t>(images) * repeats;
        const auto total = epochs ? length * epochs : static_cast<uint64_t>(requested_steps);
        if (length > 100000 || total > 100000)
            throw std::invalid_argument("epoch schedule exceeds 100000 steps");
        steps_per_epoch = static_cast<int>(length);
        steps = static_cast<int>(total);
    }
    bool epoch_end(int step) const { return step > 0 && step % steps_per_epoch == 0; }
};

// Deterministic DDIM (eta=0), epsilon prediction, with sequential CFG passes.
// The caller supplies current model predictions; this function never updates weights.
template<class Predictor>
std::vector<float> sd15_ddim_sample(size_t features, uint32_t seed, int iterations,
                                    float guidance, const std::array<double, 1000>& alpha,
                                    Predictor predict) {
    if (!features || iterations < 2 || iterations > 1000 || !std::isfinite(guidance))
        throw std::invalid_argument("invalid sample settings");
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> latent(features);
    for (auto& value : latent) value = normal(random);
    for (int i = iterations - 1; i >= 0; --i) {
        const int t = i * 999 / (iterations - 1);
        const int previous = (i - 1) * 999 / (iterations - 1);
        auto unconditional = predict(latent, t, false);
        auto conditional = predict(latent, t, true);
        if (conditional.size() != features || unconditional.size() != features)
            throw std::runtime_error("incorrect sample prediction shape");
        const double next_alpha = i > 0 ? alpha[previous] : 1.0;
        for (size_t j = 0; j < features; ++j) {
            const double epsilon = unconditional[j] + guidance * (conditional[j] - unconditional[j]);
            const double clean = (latent[j] - std::sqrt(1.0 - alpha[t]) * epsilon) / std::sqrt(alpha[t]);
            latent[j] = static_cast<float>(std::sqrt(next_alpha) * clean + std::sqrt(1.0 - next_alpha) * epsilon);
            if (!std::isfinite(latent[j])) throw std::runtime_error("non-finite sample latent");
        }
    }
    return latent;
}
#endif
