#ifndef SDKIT_SD15_TRAINING_SCHEDULE_HPP
#define SDKIT_SD15_TRAINING_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

class SD15TrainingSchedule {
    bool cosine_;
    int steps_, warmup_, cycles_;

public:
    SD15TrainingSchedule(const std::string& name, int steps, int warmup, int cycles)
        : cosine_(name == "cosine_with_restarts"), steps_(steps), warmup_(warmup), cycles_(cycles) {
        if (name != "constant" && !cosine_)
            throw std::invalid_argument("unsupported learning rate scheduler");
        if (steps < 1 || warmup < 0 || warmup >= steps || cycles < 1 || cycles > steps - warmup)
            throw std::invalid_argument("invalid scheduler steps, warmup or cycles");
        if (!cosine_ && (warmup != 0 || cycles != 1))
            throw std::invalid_argument("constant scheduling requires zero warmup and one cycle");
    }

    // Zero-based optimizer update index, matching the HF hard-restart schedule.
    float factor(int update) const {
        if (update < 0) throw std::invalid_argument("negative optimizer update index");
        if (!cosine_) return 1.0f;
        if (update < warmup_) return static_cast<float>(update) / warmup_;
        if (update >= steps_) return 0.0f;
        const double progress = static_cast<double>(update - warmup_) / (steps_ - warmup_);
        const double phase = std::fmod(cycles_ * progress, 1.0);
        return static_cast<float>(std::max(0.0, 0.5 * (1.0 + std::cos(std::acos(-1.0) * phase))));
    }
};

#endif
