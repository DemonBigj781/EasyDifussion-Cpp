#include "../tools/sd15_epoch_sampling.hpp"
#include <iostream>
#include <limits>

int main() {
    bool good = true;
    SD15EpochSchedule schedule(2, 3, 2, 99);
    good = good && schedule.steps_per_epoch == 6 && schedule.steps == 12;
    for (int step = 1; step <= 12; ++step)
        good = good && schedule.epoch_end(step) == (step == 6 || step == 12);
    SD15EpochSchedule partial(2, 3, 0, 7);
    good = good && partial.steps == 7 && !partial.epoch_end(7);
    for (int repeat : {0, -1, std::numeric_limits<int>::max()}) {
        try { SD15EpochSchedule invalid(2, repeat, 1, 1); good = false; }
        catch (const std::invalid_argument&) {}
    }
    std::array<double, 1000> alpha{};
    double product = 1.0;
    for (int t = 0; t < 1000; ++t) {
        const double b = std::sqrt(.00085) + double(t) / 999 * (std::sqrt(.012) - std::sqrt(.00085));
        alpha[t] = product *= 1 - b * b;
    }
    int calls = 0, last_t = 1000;
    auto predict = [&](const std::vector<float>& latent, int t, bool positive) {
        good = good && t <= last_t;
        last_t = t;
        ++calls;
        return std::vector<float>(latent.size(), positive ? .02f : .01f);
    };
    auto a = sd15_ddim_sample(8, 42, 25, 7.5f, alpha, predict);
    good = good && calls == 50 && last_t == 0;
    last_t = 1000;
    auto b = sd15_ddim_sample(8, 42, 25, 7.5f, alpha, predict);
    good = good && a == b;
    good = good && std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
    try {
        sd15_ddim_sample(8, 42, 25, 7.5f, alpha, [](const auto&, int, bool) {
            return std::vector<float>(8, std::numeric_limits<float>::quiet_NaN());
        });
        good = false;
    } catch (const std::runtime_error&) {}
    std::cout << (good ? "PASS" : "FAIL") << ": epoch boundaries and deterministic finite DDIM sampling\n";
    return good ? 0 : 1;
}
