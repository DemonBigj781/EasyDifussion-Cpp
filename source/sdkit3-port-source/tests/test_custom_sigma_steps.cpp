#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "runtime/custom_sigma_steps.h"

static void check(int steps, int high_noise, int count, int expected_steps, int expected_high) {
    int total = steps + std::max(0, high_noise);
    resolve_custom_sigma_steps(count, steps, high_noise, total);
    if (steps != expected_steps || high_noise != expected_high || total != count - 1) {
        throw std::runtime_error("custom sigma step partition mismatch");
    }
}

int main() {
    try {
        check(20, 0, 41, 40, 0);
        check(40, 0, 21, 20, 0);
        check(20, 0, 21, 20, 0);
        check(0, 0, 2, 1, 0);
        check(20, 10, 41, 20, 20);
        check(20, 10, 16, 15, 0);
        check(20, 10, 21, 20, 0);
        check(20, 10, 31, 20, 10);
        check(0, 10, 21, 0, 20);
        check(20, -1, 41, 40, -1);
        check(20, -1, 21, 20, -1);
        check(40, -1, 21, 20, -1);
        for (int steps = 0; steps <= 20; ++steps) {
            for (int high = -1; high <= 10; ++high) {
                for (int count = 2; count <= 42; ++count) {
                    int sample = steps, high_noise = high;
                    int total = sample + std::max(0, high_noise);
                    resolve_custom_sigma_steps(count, sample, high_noise, total);
                    if (total != count - 1 || sample + std::max(0, high_noise) != total || sample < 0 ||
                        (high == -1 && high_noise != -1) || (high == 0 && high_noise != 0)) {
                        throw std::runtime_error("custom sigma partition invariant failed");
                    }
                }
            }
        }
        std::cout << "PASS custom sigma short/long/exact schedules, dual-model partitions, automatic boundary sentinel and 10332 partition checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
