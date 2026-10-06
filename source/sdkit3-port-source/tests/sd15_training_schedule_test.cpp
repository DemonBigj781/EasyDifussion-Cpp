#include "../tools/sd15_training_schedule.hpp"
#include <iostream>

int main() {
    bool good = true;
    auto check = [&](float actual, float expected) {
        if (std::abs(actual - expected) > 1e-6f) {
            std::cerr << actual << " != " << expected << '\n';
            good = false;
        }
    };
    SD15TrainingSchedule constant("constant", 8, 0, 1);
    SD15TrainingSchedule cosine("cosine_with_restarts", 8, 0, 2);
    for (int i = 0; i <= 8; ++i) check(constant.factor(i), 1);
    check(cosine.factor(0), 1);
    check(cosine.factor(2), .5f);
    check(cosine.factor(3), .1464466094f);
    check(cosine.factor(4), 1);
    check(cosine.factor(6), .5f);
    check(cosine.factor(8), 0);
    SD15TrainingSchedule warmup("cosine_with_restarts", 10, 2, 2);
    check(warmup.factor(0), 0);
    check(warmup.factor(1), .5f);
    for (int i = 0; i <= 8; ++i) check(warmup.factor(i + 2), cosine.factor(i));
    check(SD15TrainingSchedule("cosine_with_restarts", 1, 0, 1).factor(0), 1);
    for (const auto& name : {"typo", "constant", "cosine_with_restarts"}) {
        try { SD15TrainingSchedule invalid(name, 2, 2, 1); good = false; }
        catch (const std::invalid_argument&) {}
    }
    try { SD15TrainingSchedule invalid("constant", 8, 0, 2); good = false; }
    catch (const std::invalid_argument&) {}
    std::cout << (good ? "PASS" : "FAIL") << ": schedule, warmup, restart and terminal boundaries\n";
    return good ? 0 : 1;
}
