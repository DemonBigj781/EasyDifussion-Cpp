#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "runtime/hires_steps.h"

int main() {
    try {
        for (int steps = 1; steps <= 100; ++steps) {
            for (float strength : {0.1f, 0.25f, 0.3f, 0.5f, 0.7f, 0.9f, 1.f}) {
                const int scheduler_steps = std::max(1, static_cast<int>(steps / strength));
                const int start = hires_sigma_start(scheduler_steps, steps);
                std::vector<float> sigmas(scheduler_steps + 1);
                for (int i = 0; i <= scheduler_steps; ++i) sigmas[i] = static_cast<float>(scheduler_steps - i);
                if (start < 0 || start >= scheduler_steps) throw std::runtime_error("hires sigma index out of bounds");
                std::vector<float> selected(sigmas.begin() + start, sigmas.end());
                if (selected.size() != static_cast<size_t>(steps + 1) || selected.back() != 0.f || selected.front() != steps) {
                    throw std::runtime_error("hires schedule does not honor requested effective steps");
                }
            }
        }
        if (hires_sigma_start(1, 4) != 0) throw std::runtime_error("short schedule clamp changed");
        std::cout << "PASS 700 hires step/strength combinations, sigma order/terminal zero, one-step boundary and short-schedule clamp\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
